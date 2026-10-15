from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive_thinking.evaluation.answers import last_boxed_only_string, remove_boxed

if TYPE_CHECKING:
    import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)
