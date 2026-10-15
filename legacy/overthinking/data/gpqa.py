import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from datasets import load_dataset


def reconst_instance(d):
    import random

    question = d["Question"]
    correct_answer = str(d["Correct Answer"])
    explanation = d["Explanation"]
    subdomain = d.get("Subdomain", "")
    domain = d.get("High-level domain", "")

    # 모든 선택지를 리스트로 만들기
    all_choices = [
        d["Correct Answer"],
        d["Incorrect Answer 1"],
        d["Incorrect Answer 2"],
        d["Incorrect Answer 3"],
    ]

    # 선택지 순서를 섞기
    random.shuffle(all_choices)

    # 섞인 선택지에서 정답의 위치 찾기
    correct_choice_index = all_choices.index(correct_answer)
    choice_labels = ["A", "B", "C", "D"]
    correct_choice = choice_labels[correct_choice_index]

    # 선택지 딕셔너리 구성
    options = {}
    for i, choice in enumerate(all_choices):
        options[choice_labels[i]] = choice

    return {
        "question": question,
        "answer": correct_choice,  # A, B, C, D 중 정답 레이블
        "answer_text": correct_answer,  # 실제 정답 텍스트
        "options": options,
        "explanation": explanation,
        "subdomain": subdomain,
        "domain": domain,
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="Idavidrein/gpqa")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="gpqa.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset
    data = load_dataset(args.data_path, "gpqa_main", split="train")

    if args.max_examples > 0:
        # np ramdom select max examples
        indices = np.random.choice(range(len(data)), args.max_examples, replace=False)
        # np int to int
        indices = [int(i) for i in indices]
        data = [data[i] for i in indices]

    new_data = []

    for d in data:
        new_data.append(reconst_instance(d))

    # # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in new_data:
            json.dump(d, f)
            f.write("\n")
