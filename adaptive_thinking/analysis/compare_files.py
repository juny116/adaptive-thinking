import logging
import os

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from adaptive_thinking.core.paths import default_config_dir
from adaptive_thinking.data.jsonl import read_jsonl
from adaptive_thinking.evaluation.answers import last_boxed_only_string, remove_boxed
from adaptive_thinking.evaluation.grader import (
    grade_answer,
)

logger = logging.getLogger("httpx")
logger.propagate = False


@hydra.main(version_base=None, config_path=default_config_dir(), config_name="eval")
def main(config: DictConfig) -> None:

    qwq_tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer", "Qwen/QwQ-32B")
    )

    data_dir = config.get("data_dir", "ablation")
    # Dynamically list subdirectories under ablation as settings
    settings = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    # Compute and print accuracy and token length for each setting
    for setting in settings:
        dir_path = os.path.join(data_dir, setting)
        file_list = [
            f
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jsonl")
        ]
        # Print per-file accuracy and token length
        print(f"Setting {setting} results:")
        for file_name in file_list:
            eval_data = read_jsonl(os.path.join(dir_path, file_name))
            correct_list = []
            lengths = []
            for d in eval_data:
                content = d["messages"][-1]["content"]
                token_length = len(qwq_tokenizer(content)["input_ids"])
                prediction = remove_boxed(last_boxed_only_string(content))
                if prediction is None:
                    correct_list.append(False)
                else:
                    correct_list.append(grade_answer(prediction, d["answer"]))
                lengths.append(token_length)
            if correct_list:
                file_acc = sum(correct_list) / len(correct_list)
                file_len = sum(lengths) / len(lengths)
            else:
                file_acc = 0
                file_len = 0
            print(
                f"  {file_name}: Accuracy: {file_acc:.4f}, Avg Token Length: {file_len:.2f}"
            )
        print()
    return


if __name__ == "__main__":
    main()
