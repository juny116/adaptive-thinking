"""Summarize continuation markers and confidence labels in generated responses."""

import argparse
import json

from adaptive_thinking.data.jsonl import read_jsonl

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


def summarize(records, limit=100):
    rows = records[:limit]
    wait = alternative = count = 0
    total = 0.0
    for row in rows:
        content = row["messages"][-1]["content"]
        wait += content.count("Wait")
        alternative += content.count("Alternatively")
        for _, label in row["confidences"]:
            if label in confidence_class:
                total += confidence_class[label]
                count += 1
    n = len(rows)
    return {
        "examples": n,
        "wait_per_example": wait / n if n else 0,
        "alternative_per_example": alternative / n if n else 0,
        "confidence_entries_per_example": count / n if n else 0,
        "mean_confidence": total / count if count else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(read_jsonl(args.input), args.limit), indent=2))


if __name__ == "__main__":
    main()
