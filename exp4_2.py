import json
import argparse

from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import APIModel
from utils import (
    read_jsonl,
    remove_boxed,
    last_boxed_only_string,
)
from grader import grade_answer, _normalize, normalize_answer
from omegaconf import DictConfig
import hydra
import os
import logging
import string
from transformers import AutoTokenizer
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger("httpx")
logger.propagate = False


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:

    qwq_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    data_dir = "ablation"
    # Path for caching computed results
    results_file = os.path.join(data_dir, "results_by_setting.json")

    # Helpers to save/load results
    def save_results(path, obj):
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)

    def load_results(path):
        with open(path, "r") as f:
            return json.load(f)

    # Dynamically list subdirectories under ablation as settings
    settings = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    # Load or compute results
    if os.path.exists(results_file):
        results = load_results(results_file)
    else:
        results = {}
        for setting in settings:
            dir_path = os.path.join(data_dir, setting)
            file_list = [f for f in os.listdir(dir_path) if f.endswith(".jsonl")]
            results[setting] = {"files": {}}
            accs, lens = [], []
            for fn in file_list:
                eval_data = read_jsonl(os.path.join(dir_path, fn))
                cl, ll = [], []
                for d in eval_data:
                    content = d["messages"][-1]["content"]
                    tlen = len(qwq_tokenizer(content)["input_ids"])
                    pred = remove_boxed(last_boxed_only_string(content))
                    cl.append(
                        False if pred is None else grade_answer(pred, d["answer"])
                    )
                    ll.append(tlen)
                fa = sum(cl) / len(cl) if cl else 0
                fl = sum(ll) / len(ll) if ll else 0
                results[setting]["files"][fn] = {"accuracy": fa, "token_length": fl}
                accs.append(fa)
                lens.append(fl)
            # store averages
            results[setting]["avg_accuracy"] = sum(accs) / len(accs) if accs else 0
            results[setting]["avg_token_length"] = sum(lens) / len(lens) if lens else 0
        save_results(results_file, results)

    # Print per-file metrics and average per setting
    for setting in settings:
        print(f"Setting {setting} results:")
        for fn, m in results[setting]["files"].items():
            print(
                f"  {fn}: Accuracy: {m['accuracy']:.4f}, Avg Token Length: {m['token_length']:.2f}"
            )
        avg = results[setting]
        print(
            f"  >> Average: Accuracy: {avg['avg_accuracy']:.4f}, Token Length: {avg['avg_token_length']:.2f}\n"
        )

    # Plot 2D: average accuracy vs token length per setting
    # Compute averages for plotting
    avg_accs = []
    avg_lens = []
    for setting in settings:
        avg_accs.append(results[setting]["avg_accuracy"])
        avg_lens.append(results[setting]["avg_token_length"])
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(avg_lens, avg_accs)
    for i, setting in enumerate(settings):
        ax.annotate(setting, (avg_lens[i], avg_accs[i]))
    ax.set_xlabel("Average Token Length")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Accuracy vs Token Length per Setting")
    ax.grid(True)
    fig.savefig("accuracy_vs_length.png", bbox_inches="tight")
    plt.close(fig)
    return


if __name__ == "__main__":
    main()
