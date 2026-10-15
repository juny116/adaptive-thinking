"""Shared asynchronous execution, resume handling and result writing."""

import asyncio
import string
from dataclasses import dataclass
from pathlib import Path

from adaptive_thinking.data.jsonl import read_jsonl, write_jsonl
from adaptive_thinking.evaluation import summaries

from .model import APIModel


@dataclass(frozen=True)
class RunOptions:
    summary: str = "legacy"
    few_shot: bool = False
    chat_template: bool = True
    threshold: bool = False


async def run_experiment(
    config, process_example, *, options=RunOptions(), model=None, gather=None
):
    model_config = config["model"]
    if model is None:
        kwargs = dict(
            api_key=model_config["api_key"],
            model=model_config["model_name_or_path"],
            base_url=model_config["base_url"],
            organization=model_config["organization"],
            project=model_config["project"],
        )
        if options.chat_template:
            kwargs["chat_template"] = model_config.get("chat_template")
        model = APIModel(**kwargs)
    if gather is None:
        from tqdm.asyncio import tqdm

        gather = tqdm.gather
    template = config["template"]
    prompt, form = template["instruction"], template["format"]
    fields = [field for _, field, _, _ in string.Formatter().parse(form) if field]
    if options.few_shot and config.get("few-shot_path"):
        print("Using few-shot learning")
        for item in read_jsonl(config["few-shot_path"]):
            prompt += (
                form.format(**{field: item.get(field, "") for field in fields}) + "\n\n"
            )
    data = read_jsonl(config["data_path"])
    result_path = Path(config["result_path"])
    existing = read_jsonl(result_path) if result_path.exists() else []
    completed = {row["index"] for row in existing}
    semaphore = asyncio.Semaphore(config.get("concurrency", 20))
    extra = (config.get("threshold", 0.9),) if options.threshold else ()
    tasks = [
        process_example(
            i,
            item,
            model,
            prompt,
            form,
            fields,
            template.get("system"),
            model_config,
            semaphore,
            *extra,
        )
        for i, item in enumerate(data[: config["max_examples"]])
        if i not in completed
    ]
    print(f"Running {len(tasks)} tasks asynchronously...")
    results = await gather(*tasks)
    rows = sorted(
        existing + [row for row in results if row is not None],
        key=lambda row: row["index"],
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, result_path)
    getattr(summaries, "summarize_" + options.summary)(config, model)
    return rows
