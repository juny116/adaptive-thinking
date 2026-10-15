"""Split reasoning traces at Wait markers or fixed word counts."""

import argparse
import re

from .jsonl import read_jsonl, write_jsonl


def split_by_wait(text):
    parts = re.split(r"Wait", text)
    return [parts[0].strip()] + ["Wait" + part.strip() for part in parts[1:]]


def split_by_words(text, words=300):
    if words <= 0:
        raise ValueError("words must be positive")
    chunks, current, count = [], [], 0
    for token in re.findall(r"\S+|\s+", text):
        current.append(token)
        if token.strip():
            count += 1
        if count >= words:
            chunks.append("".join(current))
            current, count = [], 0
    if current:
        chunks.append("".join(current))
    return chunks


def split_records(records, mode="wait", words=300, limit=100):
    if mode not in ("wait", "words"):
        raise ValueError("mode must be wait or words")
    for i, record in enumerate(records[:limit]):
        content = record["messages"][-1]["content"]
        if content.count("</think>") != 1:
            continue
        reasoning = content.split("</think>")[0].strip()
        parts = (
            split_by_wait(reasoning)
            if mode == "wait"
            else split_by_words(reasoning, words)
        )
        yield {
            "index": i,
            **{
                key: record[key]
                for key in ("answer", "question", "solution", "messages")
            },
            "split": parts,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("wait", "words"), default="wait")
    parser.add_argument("--words", type=int, default=300)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    rows = list(
        split_records(read_jsonl(args.input), args.mode, args.words, args.limit)
    )
    write_jsonl(rows, args.output)
    print(f"Total split parts: {sum(len(row['split']) for row in rows)}")


if __name__ == "__main__":
    main()
