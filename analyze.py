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


data_path = "sample_results/aime2024/qwq/very.jsonl"
test_data = read_jsonl(data_path)
# tokenizer=  AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

num_wait = 0
num_alt = 0
confidence_class = {
    "Almost no chance": 0.5,
    "Highly unlikely": 1.5,
    "Chances are slight": 2.5,
    "Unlikely": 3.5,
    "Less than even": 4.5,
    "Better than even": 5.5,
    "Likely": 7,
    "Very good chance": 8.5,
    "Highly likely": 9.25,
    "Almost certain": 9.75,
}
sum_confidence = 0.0
cnt = 0
for i, d in enumerate(test_data[:100]):
    content = d["messages"][-1]["content"]

    # count occurrences of "Wait" and "Alternatively"
    num_wait += content.count("Wait")
    num_alt += content.count("Alternatively")

    confidences = d["confidences"]
    for confidence in confidences:
        conf = confidence[1]
        if conf in confidence_class:
            sum_confidence += confidence_class[conf]
            cnt += 1
print(f"Total 'Wait' occurrences: {num_wait/len(test_data):.2f}")
print(f"Total 'Alternatively' occurrences: {num_alt/len(test_data):.2f}")
print(f"Total confidence entries: {cnt/len(test_data):.2f}")
avg_confidence = sum_confidence / cnt
print(f"Average confidence: {avg_confidence:.2f}")
