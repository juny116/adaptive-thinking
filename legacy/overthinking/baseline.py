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

SEMAPHORE_LIMIT = 20  # adjust this number as needed
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

logger = logging.getLogger("httpx")
logger.propagate = False


async def process_single_example(
    i, d, model, prompt, format, fields, system_prompt, model_config, semaphore
):
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    answer = d["answer"]
    d["answer"] = ""
    query = format.format(**{field: d.get(field, "") for field in fields})
    context = prompt + query
    messages.append({"role": "user", "content": context})

    try:
        async with semaphore:
            result = await model.agenerate(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                top_p=model_config.get("top_p", 1.0),
                top_k=model_config.get("top_k", 0),
                min_p=model_config.get("min_p", 0),
            )
    except Exception as e:
        print(f"[ERROR] {i}: {e}")
        return None

    messages = result["messages"]
    # print(f"[{i}] {messages_to_string(messages)}")
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
    )
    system_prompt = config["template"].get("system")

    instruction = config["template"]["instruction"]
    format = config["template"]["format"]

    formatter = string.Formatter()
    fields = [field for _, field, _, _ in formatter.parse(format) if field]

    prompt = instruction

    test_data = read_jsonl(config["data_path"])
    if config["few-shot_path"]:
        print("Using few-shot learning")
        train_data = read_jsonl(config["few-shot_path"])
        few_shot = ""
        for d in train_data:
            few_shot += format.format(**{field: d.get(field, "") for field in fields})
            few_shot += "\n\n"
        prompt += few_shot

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
