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


data_path = "sample_results/olympiad/qwen/fix_len_2.jsonl"
test_data = read_jsonl(data_path)
# tokenizer=  AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)


for i, d in enumerate(test_data):
    answer = d["answer"][0]
    d["answer"] = answer

with open(data_path, "w") as f:
    for d in test_data:
        json.dump(d, f)
        f.write("\n")
