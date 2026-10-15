import asyncio
import copy
import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from adaptive_thinking.core.confidence import end_parentheses, parse_confidence
from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.core.runner import RunOptions, run_experiment
from adaptive_thinking.evaluation.answers import last_boxed_only_string, remove_boxed
from adaptive_thinking.evaluation.grader import grade_answer

logger = logging.getLogger("httpx")
logger.propagate = False

stop_words = ["Wait", "</think>", "</thought>"]


async def process_single_example(
    i, d, model, prompt, format, fields, system_prompt, model_config, semaphore
):
    # Acquire semaphore for entire function
    async with semaphore:
        messages = []
        answer = d["answer"]
        d["answer"] = ""
        query = format.format(**{field: d.get(field, "") for field in fields})
        context = prompt + query
        messages.append({"role": "user", "content": context})
        split = d.get("split", None)
        content = ""
        total_responses = []
        conf_self = []
        conf_met = []
        accuracies = []
        for s in split:
            content += s + "\n"
            messages.append({"role": "assistant", "content": content})
            # calculate the confidence level with self-assessment
            # perform confidence query without altering main message thread
            confidence_input = copy.deepcopy(messages)
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
            # get the confidence level from the last message
            confidence = parse_confidence(results["messages"][-1]["content"])
            conf_self.append(confidence)
            # calculate the confidence with metric
            # perform confidence query without altering main message thread
            confidence_input = copy.deepcopy(messages)
            confidence_input[-1]["content"] += (
                "\n**Final Answer**\nThe answer is \\boxed{"
            )
            try:
                results = await model.agenerate(
                    confidence_input,
                    max_new_tokens=30,
                    temperature=model_config["temperature"],
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
            logprobs = results["logprobs"].content
            # print(f"[DEBUG] {i}: logprobs: {logprobs}")
            # calculate average logprobs for all tokens except the last one
            sum_logprobs = 0
            for logprob in logprobs[:-1]:
                sum_logprobs += logprob.logprob
            len_logprobs = len(logprobs) - 1
            avg_logprobs = sum_logprobs / len_logprobs
            avg_logprobs = np.exp(avg_logprobs)
            conf_met.append(avg_logprobs)

            # sample responses 5 times using current reasoning trace
            responses = []
            for _ in range(5):
                response_input = copy.deepcopy(messages)
                response_input[-1]["content"] += "\n</think>"
                try:
                    result = await model.agenerate(
                        response_input,
                        max_new_tokens=5000,
                        temperature=model_config["temperature"],
                        top_p=model_config.get("top_p", 1.0),
                        top_k=model_config.get("top_k", 0),
                        min_p=model_config.get("min_p", 0),
                    )
                    responses.append(result["messages"][-1]["content"])
                except Exception as e:
                    print(f"[ERROR] {i}: {e}")

            correct_list = []
            for r in responses:
                try:
                    prediction = remove_boxed(last_boxed_only_string(r))
                except Exception as e:
                    print(f"[ERROR] {i}: {e}")
                    prediction = ""

                correct = grade_answer(prediction, answer)
                correct_list.append(correct)
            # calculate the accuracy of the responses

            acc = np.mean(correct_list)

            total_responses.append(responses)
            accuracies.append(acc)

        # return {
        #     "index": i,
        #     "answer": answer,
        #     "messages": messages,
        #     "question": d["question"],
        #     "solution": d["solution"],
        # }
        print(accuracies, conf_self, conf_met)
        print(f"----------------------------")
        return {
            "index": i,
            "answer": answer,
            "messages": messages,
            "question": d["question"],
            "solution": d["solution"],
            "responses": total_responses,
            "confidence_self": conf_self,
            "confidence_met": conf_met,
            "accuracy": accuracies,
        }


async def main_async(config: DictConfig):
    return await run_experiment(config, process_single_example, options=RunOptions())


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="baseline")
def main(config: DictConfig):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
