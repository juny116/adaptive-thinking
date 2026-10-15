import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from utils import last_boxed_only_string, remove_boxed
from datasets import load_dataset

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="HuggingFaceH4/MATH-500")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="data/math_500.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset

    data = load_dataset(
        args.data_path,
        split="test",
        trust_remote_code=True,
        cache_dir="data",
    )

    if args.max_examples > 0:
        # np ramdom select max examples
        indices = np.random.choice(range(len(data)), args.max_examples, replace=False)
        # np int to int
        indices = [int(i) for i in indices]
        data = [data[i] for i in indices]

    new_data = []
    for d in data:
        new_data.append(
            {
                "question": d["problem"],
                "answer": remove_boxed(last_boxed_only_string(d["solution"])),
                "solution": d["solution"],
            }
        )

    # # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in new_data:
            json.dump(d, f)
            f.write("\n")
