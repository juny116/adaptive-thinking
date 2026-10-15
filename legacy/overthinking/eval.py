import json

from typing import List
from tqdm import tqdm
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

logger = logging.getLogger("httpx")
logger.propagate = False


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_name_or_path"],
    )
    eval_data = read_jsonl(config["data_path"])
    correct_list = []
    invalid_list = []
    length_list = []
    for i, d in enumerate(tqdm(eval_data)):
        # get the token length
        try:
            prediction = remove_boxed(
                last_boxed_only_string(d["messages"][-1]["content"])
            )
        except Exception as e:
            print(f"[ERROR] {i}: {e}")
            invalid_list.append(i)
            prediction = ""
        # print(f"[{i}] {prediction}, {d['answer']}")
        content = d["messages"][-1]["content"]
        length = len(tokenizer(content)["input_ids"])
        length_list.append(length)
        correct = grade_answer(prediction, d["answer"])
        correct_list.append(correct)

    # print accuracy
    accuracy = sum(correct_list) / len(correct_list)
    print(f"Accuracy: {accuracy:.4f}")
    # print number of correct answers
    print(f"Number of correct answers: {sum(correct_list)}")
    # print number of incorrect answers
    print(f"Number of incorrect answers: {len(correct_list) - sum(correct_list)}")
    # print number of invalid answers
    print(f"Number of invalid answers: {len(invalid_list)}")
    # print average length of the answers
    print(f"Average length of the answers: {sum(length_list) / len(length_list):.2f}")


if __name__ == "__main__":
    main()
