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


data_path = "sample_results/math_500/qwq/base.jsonl"
test_data = read_jsonl(data_path)
# tokenizer=  AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
cnt = 0
new_data = []
lengths = []
for i, d in enumerate(test_data[:100]):
    content = d["messages"][-1]["content"]

    # check if </think> is in the content only once
    if content.count("</think>") != 1:
        print(f"Skipping index {i} due to multiple or no </think> tags.")
        continue
    # parse reasoning part
    reasoning_content = content.split("</think>")[0].strip()
    # split reasoning into 300-word chunks while preserving formatting
    import re

    token_pattern = re.compile(r"\S+|\s+")

    tokens = token_pattern.findall(reasoning_content)
    reasoning_parts = []
    current_chunk = []
    word_count = 0

    for token in tokens:
        current_chunk.append(token)
        if token.strip() and not token.isspace():
            word_count += 1
        if word_count >= 300:
            reasoning_parts.append("".join(current_chunk))
            current_chunk = []
            word_count = 0

    if current_chunk:
        reasoning_parts.append("".join(current_chunk))

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
output_path = "ablation/math_300.jsonl"
with open(output_path, "w") as f:
    for item in new_data:
        f.write(json.dumps(item) + "\n")
