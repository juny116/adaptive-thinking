import json
import argparse

from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import APIModel
from utils import (
    read_jsonl,
    remove_boxed,
    last_boxed_only_string,
)
from grader import grade_answer, _normalize, normalize_answer
from omegaconf import DictConfig
import hydra
import os
import logging
import string
from transformers import AutoTokenizer
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger("httpx")
logger.propagate = False


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    qwq_tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-reasoning")
    r1_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    data_dir = "final_results"
    datasets = ["math_500", "aime2024", "aime2025", "amc23"]
    models = ["qwen", "distil", "qwq", "qwen-14b", "r1-14b"]
    settings = ["base", "deer", "ours"]

    for dataset in datasets:
        for model in models:
            # Ensure tokenizer is defined for each model
            if model == "qwen":
                tokenizer = qwen_tokenizer
            elif model == "qwq":
                tokenizer = qwq_tokenizer
            elif model == "phi":
                tokenizer = phi_tokenizer
            elif model == "distil":
                tokenizer = r1_tokenizer
            elif model == "qwen-14b":
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
            elif model == "r1-14b":
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            for setting in settings:
                dir_path = os.path.join(data_dir, dataset, model)
                # Ensure results are stored for all settings, even if no files exist
                file_path = os.path.join(dir_path, setting + ".jsonl")
                # for all files in the directory
                correct_list = []
                lengths = []
                invalid_list = []
                eval_data = read_jsonl(file_path)
                for i, d in enumerate(eval_data):
                    content = d["messages"][-1]["content"]
                    token_length = len(tokenizer(content)["input_ids"])
                    prediction = remove_boxed(last_boxed_only_string(content))
                    if prediction is None:
                        prediction = None
                        invalid_list.append(i)
                        correct = False
                        lengths.append(token_length)
                        correct_list.append(correct)
                    else:
                        correct = grade_answer(prediction, d["answer"])
                        lengths.append(token_length)
                        correct_list.append(correct)

                # Calculate average accuracy and token length
                if len(correct_list) > 0:
                    avg_accuracy = sum(correct_list) / len(correct_list)
                else:
                    avg_accuracy = 0.0
                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                print(
                    f"Dataset: {dataset}, Model: {model}, Setting: {setting}, "
                    f"Average Accuracy: {avg_accuracy:.4f}, Average Length: {avg_length:.2f}"
                )


if __name__ == "__main__":
    main()
