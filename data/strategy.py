import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from datasets import load_dataset


def reconst_instance(d):
    question = d["question"]
    facts = d["facts"]
    decomposition = d["decomposition"]
    description = d["description"]
    answer = str(d["answer"])

    options = {}

    return {
        "question": question,
        "answer": answer,
        "options": options,
        "facts": facts,
        "decomposition": decomposition,
        "description": description,
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="wics/strategy-qa")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="strategy_qa.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset
    data = load_dataset(args.data_path, split="test")

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
