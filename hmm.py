import json
import asyncio
import os
import string
import logging

from tqdm.asyncio import tqdm as async_tqdm
from models import APIModel
from utils import (
    compute_reward,
    read_jsonl,
    write_jsonl,
    messages_to_string,
    is_valid_boxed_answer,
)
from omegaconf import DictConfig
import hydra
import copy

SEMAPHORE_LIMIT = 20  # adjust this number as needed
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

logger = logging.getLogger("httpx")
logger.propagate = False

stop_words = ["Wait", "Alternatively", "Hmm", "</think>", "</thought>"]
confidence_class = {
    "Almost no chance": 0,
    "Highly unlikely": 1,
    "Chances are slight": 2,
    "Unlikely": 3,
    "Less than even": 4,
    "Better than even": 5,
    "Likely": 6,
    "Very good chance": 7,
    "Highly likely": 8,
    "Almost certain": 9,
}
end_parentheses = [
    "}",
    "}\n",
    "}\n\n",
    "}\n\n\n",
    "}\n\n\n\n",
    "}\n\n\n\n\n",
    "}\n\n\n\n\n\n",
    " }",
    "} ",
    " }\n",
]

import re


def parse_confidence(text):
    """
    Extracts the last occurring confidence level from a string in the format \confidence{X}.
    Returns the confidence level (X) if found, else returns None.
    """
    matches = re.findall(r"\\confidence\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


async def process_single_example(
    i, d, model, prompt, format, fields, system_prompt, model_config, semaphore
):
    # Acquire semaphore for entire function
    async with semaphore:

        # if system_prompt:
        #     messages = [{"role": "system", "content": system_prompt}]
        # else:
        messages = []
        answer = d["answer"]
        d["answer"] = ""
        query = format.format(**{field: d.get(field, "") for field in fields})
        context = prompt + query
        messages.append({"role": "user", "content": context})
        messages.append({"role": "assistant", "content": "<think>\n"})
        end = False
        last_confidence = "Almost no chance"
        count = 0

        while not end:
            # generate thinking
            try:
                result = await model.agenerate(
                    messages,
                    max_tokens=model_config["max_think"],
                    temperature=model_config["temperature"],
                    top_p=model_config.get("top_p", 1.0),
                    top_k=model_config.get("top_k", 0),
                    min_p=model_config.get("min_p", 0),
                    stop=stop_words,
                    include_stop_str_in_output=True,
                )
            except Exception as e:
                print(f"[ERROR] {i}: {e}")
                return None
            # get last word of the last message
            end_word = (
                result["messages"][-1]["content"].split()[-1]
                if result["messages"][-1]["content"]
                else ""
            )
            messages = result["messages"]

            # in case the last word is end of thinking

            if any(stop in end_word for stop in stop_words):
                # remove the last word from the last message
                messages[-1]["content"] = messages[-1]["content"].rsplit(
                    sep=None, maxsplit=1
                )[0]
            # in case the last word is not in end_word
            else:
                # print a warning (usually it is when generation is too long)
                print(f"[WARNING] {i}: Last word '{end_word}' is not in stop words.")
                # print a warning with token length and max think length
                tokenized = model.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )

                print(
                    f"[WARNING] {i}: Token length is {len(tokenized)}, max think length is {model_config['max_think']}."
                )
                # in this case we can end the thinking
                end = True

            # perform confidence query without altering main message thread
            confidence_input = copy.deepcopy(messages)
            confidence_input[0] = {"role": "system", "content": system_prompt}
            confidence_input[-1][
                "content"
            ] += "\nLet’s evaluate the confidence level of the current reasoning.\n\\confidence{"
            try:
                results = await model.agenerate(
                    confidence_input,
                    max_new_tokens=10,
                    temperature=model_config["temperature"],
                    top_p=model_config.get("top_p", 1.0),
                    top_k=model_config.get("top_k", 0),
                    min_p=model_config.get("min_p", 0),
                    stop=end_parentheses,
                    include_stop_str_in_output=True,
                )
            except Exception as e:
                print(f"[ERROR] {i}: {e}")
                end = True
            # get the confidence level from the last message
            confidence = parse_confidence(results["messages"][-1]["content"])
            if confidence == "Almost certain":
                # if confidence is almost certain, we can end the thinking
                end = True
            else:
                # check if confidence has been improving
                if confidence_class.get(last_confidence, 0) >= confidence_class.get(
                    confidence, 0
                ):
                    count += 1
                    if count > model_config["max_count"]:
                        # if confidence has not been improving for 10 times, we can end the thinking
                        print(
                            f"[WARNING] {i}: Confidence has not been improving for 10 times, ending thinking."
                        )
                        end = True
                    else:
                        messages[-1]["content"] += f"\n{end_word},"
                else:
                    count = 0
                    # if confidence is not almost certain, we can continue thinking
                    messages[-1]["content"] += f"\n{end_word},"
                last_confidence = confidence

        # end of thinking, we can get the final answer
        # append </think> to the last message
        messages[-1]["content"] += "\n</think>"
        try:
            result = await model.agenerate(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                top_p=model_config.get("top_p", 1.0),
                top_k=model_config.get("top_k", 0),
                min_p=model_config.get("min_p", 0),
            )
        except Exception as e:
            input_token_count = len(
                model.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            print(
                f"[WARNING] {i}: Model generation failed. Input token count: {input_token_count}. Max tokens: {model_config['max_tokens']}"
            )
            print(f"[WARNING] {i}: Last message: {messages[-1]['content']}")
            print(
                f"[WARNING] {i}: Tokenized messages: {model.tokenizer.apply_chat_template(messages, tokenize=False)}"
            )
            print(f"[ERROR] {i}: {e}")

        messages = result["messages"]

        return {
            "index": i,
            "answer": answer,
            "messages": messages,
            "question": d["question"],
            "solution": d["solution"],
        }


async def main_async(config: DictConfig):
    max_examples = config["max_examples"]
    model_config = config["model"]
    model = APIModel(
        api_key=model_config["api_key"],
        model=model_config["model_name_or_path"],
        base_url=model_config["base_url"],
        organization=model_config["organization"],
        project=model_config["project"],
        chat_template=model_config.get("chat_template", None),
    )
    system_prompt = config["template"].get("system")

    instruction = config["template"]["instruction"]
    format = config["template"]["format"]

    formatter = string.Formatter()
    fields = [field for _, field, _, _ in formatter.parse(format) if field]

    prompt = instruction

    test_data = read_jsonl(config["data_path"])

    existing_results = []
    if os.path.exists(config["result_path"]):
        existing_results = read_jsonl(config["result_path"])

    existing_indices = [r["index"] for r in existing_results]
    tasks = []
    for i, d in enumerate(test_data[:max_examples]):
        if i in existing_indices:
            continue
        tasks.append(
            process_single_example(
                i,
                d,
                model,
                prompt,
                format,
                fields,
                system_prompt,
                model_config,
                semaphore,
            )
        )

    print(f"Running {len(tasks)} tasks asynchronously...")
    results = await async_tqdm.gather(*tasks)

    all_results = existing_results + [r for r in results if r is not None]
    all_results.sort(key=lambda x: x["index"])
    with open(config["result_path"], "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0

    results = read_jsonl(config["result_path"])
    for r in results:
        output = (
            r["messages"][2]["content"]
            if isinstance(r["messages"], list) and len(r["messages"]) > 2
            else ""
        )

        if not is_valid_boxed_answer(output):
            invalid += 1
            continue

        reward = compute_reward(output, r["answer"])
        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")
    else:
        print("No examples were evaluated.")


@hydra.main(version_base=None, config_path="conf", config_name="baseline")
def main(config: DictConfig):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
