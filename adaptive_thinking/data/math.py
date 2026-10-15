import argparse

import numpy as np
from datasets import load_dataset

from adaptive_thinking.data.jsonl import write_jsonl
from adaptive_thinking.evaluation.answers import last_boxed_only_string, remove_boxed

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="hendrycks/competition_math")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="data/math.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset

    data = load_dataset(
        args.data_path,
        split="train",
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
    write_jsonl(new_data, args.out_file)
