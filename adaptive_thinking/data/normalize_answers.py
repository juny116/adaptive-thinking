"""Convert list-valued answers into scalar answers in JSONL records."""

import argparse

from .jsonl import read_jsonl, write_jsonl


def normalize_answers(records):
    for record in records:
        record = dict(record)
        answer = record["answer"]
        if isinstance(answer, list):
            if not answer:
                raise ValueError("Cannot normalize an empty answer list")
            record["answer"] = answer[0]
        yield record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = list(normalize_answers(read_jsonl(args.input)))
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
