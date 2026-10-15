import asyncio
import copy
import logging

import hydra
from omegaconf import DictConfig

from adaptive_thinking.core.confidence import (
    confidence_class,
    end_parentheses,
    parse_confidence,
)
from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.core.runner import RunOptions, run_experiment

logger = logging.getLogger("httpx")
logger.propagate = False


async def process_single_example(
    i,
    d,
    model,
    prompt,
    format,
    fields,
    system_prompt,
    model_config,
    semaphore,
    *,
    stop_words=("Alternatively", "</think>", "</thought>"),
):
    # Acquire semaphore for entire function
    async with semaphore:
        # if system_prompt:
        #     messages = [{"role": "system", "content": system_prompt}]
        # else:
        confidences = []
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
                    stop=list(stop_words),
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
            # confidence_input[0] = {"role": "system", "content": system_prompt}
            confidence_input.insert(0, {"role": "system", "content": system_prompt})
            confidence_input[-1]["content"] += (
                "\nLet’s evaluate the confidence level of the current reasoning.\n\\confidence{"
            )
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
            token_length = len(model.tokenizer(messages[-1]["content"])["input_ids"])
            confidences.append([token_length, confidence])
            if confidence == "Almost certain":
                # if confidence is almost certain, we can end the thinking
                end = True
            else:
                # check if confidence has been improving
                if confidence_class.get(last_confidence, 0) >= confidence_class.get(
                    confidence, 0
                ):
                    count += 1
                    if count > 999:
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
            "confidences": confidences,
        }


async def main_async(config: DictConfig):
    return await run_experiment(config, process_single_example, options=RunOptions())


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="baseline")
def main(config: DictConfig):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
