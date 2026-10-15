import json

from typing import List
from tqdm import tqdm
from models import APIModel
from utils import (
    read_jsonl,
    remove_boxed,
    last_boxed_only_string,
)
from grader import grade_answer
from omegaconf import DictConfig
import hydra
import os
import logging
import string
from transformers import AutoTokenizer
import re


data_path = "sample_results/aime2025/qwen/base_1.jsonl"
test_data = read_jsonl(data_path)

cnt = 0
new_data = []
for i, d in enumerate(test_data[:100]):
    content = d["messages"][-1]["content"]

    # check if </think> is in the content only once
    if content.count("</think>") != 1:
        print(f"Skipping index {i} due to multiple or no </think> tags.")
        continue
    # parse reasoning part
    reasoning_content = content.split("</think>")[0].strip()
    # split reasoning using Wait

    split_parts = re.split(r"Wait", reasoning_content)
    reasoning_parts = []

    if split_parts:
        reasoning_parts.append(split_parts[0].strip())

    for part in split_parts[1:]:
        if part.strip():
            reasoning_parts.append(f"Wait{part.strip()}")
    cnt += len(reasoning_parts)
    data = {
        "index": i,
        "answer": d["answer"],
        "question": d["question"],
        "solution": d["solution"],
        "split": reasoning_parts,
        "messages": d["messages"],
    }

    new_data.append(data)
print(f"Total split parts: {cnt}")
# Save the new data to a JSONL file
output_path = "ablation/aime2025_split.jsonl"
with open(output_path, "w") as f:
    for item in new_data:
        f.write(json.dumps(item) + "\n")
