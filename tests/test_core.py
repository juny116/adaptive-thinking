import asyncio
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_thinking.analysis.confidence_statistics import summarize
from adaptive_thinking.core.confidence import parse_confidence
from adaptive_thinking.core.runner import RunOptions, run_experiment
from adaptive_thinking.data.jsonl import read_jsonl, write_jsonl
from adaptive_thinking.data.normalize_answers import normalize_answers
from adaptive_thinking.data.split_reasoning import (
    split_by_wait,
    split_by_words,
    split_records,
)


class DataTests(unittest.TestCase):
    def test_last_confidence(self):
        self.assertEqual(
            parse_confidence(r"\confidence{Unlikely} then \confidence{ Likely }"),
            "Likely",
        )
        self.assertIsNone(parse_confidence("no confidence label"))

    def test_answer_normalization_preserves_scalar(self):
        original = [{"answer": ["42"]}, {"answer": "123"}]
        self.assertEqual(
            list(normalize_answers(original)), [{"answer": "42"}, {"answer": "123"}]
        )
        self.assertEqual(original[0]["answer"], ["42"])
        with self.assertRaises(ValueError):
            list(normalize_answers([{"answer": []}]))

    def test_split_preserves_whitespace(self):
        text = "a  b\nc d\tend"
        parts = split_by_words(text, 2)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part.split()) <= 2 for part in parts))
        self.assertEqual(
            split_by_wait("first Wait second Wait third"),
            ["first", "Waitsecond", "Waitthird"],
        )
        with self.assertRaises(ValueError):
            split_by_words(text, 0)

    def test_split_filters_bad_traces(self):
        row = dict(
            answer="42",
            question="q",
            solution="s",
            messages=[{"content": "<think>a b</think>42"}],
        )
        bad = {**row, "messages": [{"content": "no thinking terminator"}]}
        output = list(split_records([bad, row], mode="words", words=1))
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["index"], 1)
        self.assertEqual(output[0]["messages"], row["messages"])

    def test_statistics_uses_selected_rows(self):
        row = {
            "messages": [{"content": "Wait Alternatively"}],
            "confidences": [[3, "Likely"]],
        }
        result = summarize([row] * 200, limit=100)
        self.assertEqual(result["wait_per_example"], 1)
        self.assertEqual(result["mean_confidence"], 7)
        self.assertIsNone(summarize([])["mean_confidence"])

    def test_jsonl_roundtrip_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_jsonl([{"question": "문제"}], path)
            with path.open("a") as f:
                f.write("\n  \n")
            self.assertEqual(read_jsonl(path), [{"question": "문제"}])

    def test_all_strategy_entrypoints_load_external_config(self):
        from adaptive_thinking.cli import COMMANDS

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "smoke.yaml").write_text("max_examples: 0\n")
            for name in COMMANDS["run"]:
                with self.subTest(strategy=name):
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "adaptive_thinking",
                            "run",
                            name,
                            "--config-dir",
                            directory,
                            "--config-name",
                            "smoke",
                            "--cfg",
                            "job",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertIn("max_examples: 0", proc.stdout)

    def test_cli_without_model_dependencies(self):
        for args in (["--help"], ["list"], ["data", "split_reasoning", "--help"]):
            proc = subprocess.run(
                [sys.executable, "-m", "adaptive_thinking", *args],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = subprocess.run(
            [sys.executable, "-m", "adaptive_thinking", "run", "missing"],
            capture_output=True,
        )
        self.assertNotEqual(proc.returncode, 0)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_failure_sort_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            target = Path(directory) / "output.jsonl"
            write_jsonl([{"question": str(i)} for i in range(4)], source)
            write_jsonl([{"index": 2}], target)
            config = {
                "model": {},
                "template": {"instruction": "Solve ", "format": "{question}"},
                "data_path": str(source),
                "result_path": str(target),
                "max_examples": 4,
            }
            seen = []

            async def process(
                i, item, model, prompt, form, fields, system, model_config, semaphore
            ):
                async with semaphore:
                    seen.append(i)
                    return None if i == 1 else {"index": i}

            with (
                patch("adaptive_thinking.core.runner.summaries.summarize_legacy"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rows = await run_experiment(
                    config, process, model=object(), gather=asyncio.gather
                )
                self.assertEqual([r["index"] for r in rows], [0, 2, 3])
                self.assertEqual(seen, [0, 1, 3])
                seen.clear()
                await run_experiment(
                    config, process, model=object(), gather=asyncio.gather
                )
                self.assertEqual(seen, [1])

    async def test_few_shot_threshold_and_nested_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            few = Path(directory) / "few.jsonl"
            target = Path(directory) / "nested" / "out.jsonl"
            write_jsonl([{"question": "test"}], source)
            write_jsonl([{"question": "example"}], few)
            config = {
                "model": {},
                "template": {
                    "instruction": "Solve ",
                    "format": "{question}: {missing}",
                },
                "data_path": str(source),
                "few-shot_path": str(few),
                "result_path": str(target),
                "max_examples": 1,
                "threshold": 0.75,
            }

            async def process(
                i,
                item,
                model,
                prompt,
                form,
                fields,
                system,
                model_config,
                semaphore,
                threshold,
            ):
                self.assertEqual(prompt, "Solve example: \n\n")
                self.assertEqual(fields, ["question", "missing"])
                self.assertEqual(threshold, 0.75)
                return {"index": i}

            with (
                patch("adaptive_thinking.core.runner.summaries.summarize_legacy"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await run_experiment(
                    config,
                    process,
                    options=RunOptions(few_shot=True, threshold=True),
                    model=object(),
                    gather=asyncio.gather,
                )
            self.assertEqual(read_jsonl(target), [{"index": 0}])


if __name__ == "__main__":
    unittest.main()
