"""Confidence continuation at Wait and Alternatively boundaries."""

import asyncio
from functools import partial

import hydra

from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.core.runner import run_experiment

from .alternative_stop import process_single_example as process_with_stops

process_single_example = partial(
    process_with_stops, stop_words=["Alternatively", "Wait", "</think>", "</thought>"]
)


async def main_async(config):
    return await run_experiment(config, process_single_example)


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="baseline")
def main(config):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
