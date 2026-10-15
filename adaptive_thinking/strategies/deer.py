import asyncio
import copy
import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from adaptive_thinking.core.confidence import end_parentheses
from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.core.runner import RunOptions, run_experiment

logger = logging.getLogger("httpx")
logger.propagate = False

stop_words = ["Wait", "</think>", "</thought>"]


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
    threshold,
):
    # Acquire semaphore for entire function
    async with semaphore:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        answer = d["answer"]
        d["answer"] = ""
        query = format.format(**{field: d.get(field, "") for field in fields})
        context = prompt + query
        messages.append({"role": "user", "content": context})
        # messages.append({"role": "assistant", "content": "<think>\n"})
        end = False

        while not end:
            # generate thinking
            try:
                result = await model.agenerate(
                    messages,
                    max_tokens=model_config["max_think"],
                    temperature=0.0,
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
                # pass
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
            confidence_input[-1]["content"] += (
                "\n**Final Answer**\nThe answer is \\boxed{"
            )
            try:
                results = await model.agenerate(
                    confidence_input,
                    max_new_tokens=30,
                    temperature=0,
                    top_p=model_config.get("top_p", 1.0),
                    top_k=model_config.get("top_k", 0),
                    min_p=model_config.get("min_p", 0),
                    stop=end_parentheses,
                    include_stop_str_in_output=True,
                    logprobs=1,
                )
            except Exception as e:
                print(f"[ERROR] {i}: {e}")
                end = True
            # get the confidence level from the last message
            # print(results["messages"][-1]["content"])
            logprobs = results["logprobs"].content

            # calculate average logprobs for all tokens except the last one
            sum_logprobs = 0
            for logprob in logprobs[:-1]:
                sum_logprobs += logprob.logprob
            len_logprobs = len(logprobs) - 1
            avg_logprobs = sum_logprobs / len_logprobs
            avg_logprobs = np.exp(avg_logprobs)
            # print(avg_logprobs)
            if avg_logprobs > threshold:
                # if confidence is almost certain, we can end the thinking
                end = True
            else:
                if end_word in ["</think>", "</thought>"]:
                    messages[-1]["content"] += f"\nWait,"
                else:
                    messages[-1]["content"] += f"\n{end_word},"

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
    return await run_experiment(
        config,
        process_single_example,
        options=RunOptions(summary="deer", threshold=True),
    )


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="deer")
def main(config: DictConfig):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
