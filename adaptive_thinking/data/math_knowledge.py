import argparse

import numpy as np
from datasets import load_dataset

from adaptive_thinking.data.jsonl import write_jsonl


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
    args.add_argument("--data_path", type=str, default="yilunzhao/KnowledgeMath")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="math_know.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset
    data = load_dataset(args.data_path)
    print(data)
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
    write_jsonl(new_data, args.out_file)
