import asyncio
import logging

import hydra
from omegaconf import DictConfig

from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.core.runner import RunOptions, run_experiment

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
    return await run_experiment(
        config,
        process_single_example,
        options=RunOptions(summary="baseline", few_shot=True, chat_template=False),
    )


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="baseline")
def main(config: DictConfig):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
