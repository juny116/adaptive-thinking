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

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    qwq_tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-reasoning")
    r1_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    data_dir = "ablation"
    datasets = ["math_500", "aime2024"]
    models = ["qwen", "distil", "qwq"]
    settings = ["low", "med", "high", "adaptive"]

    # Function to save results to a JSON file
    def save_results(file_path, results):
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

    # Function to load results from a JSON file
    def load_results(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # Path to save/load results
    results_file = os.path.join(data_dir, "calculated_results.json")

    # Load cached results if available
    if os.path.exists(results_file):
        print(f"Loading cached results from {results_file}")
        results = load_results(results_file)
    else:
        results = {}
        for dataset in datasets:
            results[dataset] = {}
            for model in models:
                results[dataset][model] = {}
                # Ensure tokenizer is defined for each model
                if model == "qwen":
                    tokenizer = qwen_tokenizer
                elif model == "qwq":
                    tokenizer = qwq_tokenizer
                elif model == "phi":
                    tokenizer = phi_tokenizer
                elif model == "distil":
                    tokenizer = r1_tokenizer
                else:
                    raise ValueError(f"Unknown model: {model}")
                # 모델별로 'no' 세팅은 qwen에만 포함
                model_settings = settings + ["no"] if model == "qwen" else settings
                for setting in model_settings:
                    dir_path = os.path.join(data_dir, dataset, model, setting)
                    # Ensure results are stored for all settings, even if no files exist
                    if not os.path.exists(dir_path):
                        results[dataset][model][setting] = {
                            "avg_accuracy": 0,
                            "avg_token_length": 0,
                            "avg_unique": 0,
                        }
                        continue

                    # for all files in the directory
                    file_list = [
                        f
                        for f in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, f))
                    ]
                    # Debug: list files found
                    print(
                        f"DEBUG: {dataset}/{model}/{setting} 파일 {len(file_list)}개 발견: {file_list}"
                    )

                    accuracies = []
                    token_lengths = []
                    predictions_list = []
                    for file_name in file_list:
                        correct_list = []
                        invalid_list = []
                        lengths = []
                        predictions = []
                        if file_name.endswith(".jsonl"):
                            file_path = os.path.join(dir_path, file_name)
                            eval_data = read_jsonl(file_path)
                            for i, d in enumerate(tqdm(eval_data, desc=file_name)):
                                content = d["messages"][-1]["content"]
                                token_length = len(tokenizer(content)["input_ids"])
                                prediction = remove_boxed(
                                    last_boxed_only_string(content)
                                )
                                if prediction is None:
                                    prediction = None
                                    invalid_list.append(i)
                                    correct = False
                                    predictions.append(prediction)
                                    lengths.append(token_length)
                                    correct_list.append(correct)
                                else:
                                    correct = grade_answer(prediction, d["answer"])
                                    lengths.append(token_length)
                                    correct_list.append(correct)
                                    # Normalize the prediction
                                    prediction = normalize_answer(prediction)
                                    prediction = _normalize(prediction)
                                    predictions.append(prediction)
                    predictions_list.append(predictions)
                    # Print per-file performance
                    file_accuracy = (
                        sum(correct_list) / len(correct_list) if correct_list else 0
                    )
                    file_avg_len = sum(lengths) / len(lengths) if lengths else 0
                    print(
                        f"{dataset}/{model}/{setting}/{file_name}: accuracy={file_accuracy:.4f}, avg_token_length={file_avg_len:.2f}"
                    )
                    accuracies.append(file_accuracy)
                    token_lengths.append(file_avg_len)

                    avg_accuracy = (
                        sum(accuracies) / len(accuracies) if accuracies else 0
                    )
                    avg_token_length = (
                        sum(token_lengths) / len(token_lengths) if token_lengths else 0
                    )
                    # for each question get average number of unique predictions using predictions_list
                    if predictions_list:
                        # transpose predictions_list: list of lists (per file)
                        num_questions = len(predictions_list[0])
                        unique_counts = [set() for _ in range(num_questions)]
                        for preds in predictions_list:
                            for idx, pred in enumerate(preds):
                                unique_counts[idx].add(pred)
                        avg_unique = sum(len(s) for s in unique_counts) / len(
                            unique_counts
                        )
                    else:
                        avg_unique = 0

                    results[dataset][model][setting] = {
                        "avg_accuracy": avg_accuracy,
                        "avg_token_length": avg_token_length,
                        "avg_unique": avg_unique,
                    }
        save_results(results_file, results)

    # Define colors for each model with more saturated/darker tones
    model_colors = {
        "qwen": "#cc0000",  # Dark Red
        "qwq": "#0033cc",  # Dark Blue
        "distil": "#cc6600",  # Dark Orange
    }

    # Define marker styles for _r and non-_r settings
    marker_styles = {
        "adaptive": "*",  # Asterisk marker for _r settings
        "default": "o",  # Circle marker for non-_r settings
    }

    # Define intensity levels for settings
    setting_intensity = {
        "no": 0.2,
        "low": 0.4,
        "med": 0.7,
        "high": 1.0,
    }

    model_labels = {"qwen": "Qwen3", "qwq": "QwQ", "distil": "R1-distil"}

    # 디버그: 각 dataset/model별로 계산된 세팅 키 출력
    print("Available settings per model:")
    for dataset in datasets:
        for model in models:
            keys = list(results[dataset][model].keys())
            print(f"{dataset}/{model}: {keys}")
    # Use precomputed results for plotting: 하나의 Figure에 2x3 서브플롯
    n_rows, n_cols = len(datasets), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            # 모델별 플롯용 세팅 리스트
            model_settings = settings + ["no"] if model == "qwen" else settings
            for setting in model_settings:
                if setting in results[dataset][model]:
                    avg_acc = results[dataset][model][setting]["avg_accuracy"]
                    avg_len = results[dataset][model][setting]["avg_token_length"]
                    # 색상 및 마커 설정
                    if setting == "conf":
                        conf_colors = {
                            "qwen": "#ff69b4",
                            "qwq": "#7570b3",
                            "distil": "#d95f02",
                        }
                        color = conf_colors.get(model, "#ff69b4")
                    else:
                        base = model_colors.get(model, "#000000")
                        alpha = setting_intensity.get(setting.replace("_r", ""), 1.0)
                        color = mcolors.to_rgba(base, alpha=alpha)
                    marker = (
                        marker_styles["adaptive"]
                        if "adaptive" in setting
                        else marker_styles["default"]
                    )
                    ax.scatter(
                        avg_len,
                        avg_acc,
                        label=(
                            f"{model_labels[model]}-Ours"
                            if setting == "adaptive"
                            else f"{model_labels[model]}-{setting.capitalize()}"
                        ),
                        color=color,
                        marker=marker,
                    )
            # 축 및 제목 설정
            ax.set_xlabel("Avg Token Length", fontsize=12)
            ax.set_ylabel("Avg Accuracy", fontsize=12)
            if dataset == "math_500":
                ax.set_ylim(0.5, 1.0)
            ax.set_title(f"{dataset.upper()} / {model}", fontsize=14)
            # 범례: 아티스트가 있을 때만 표시
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower right", fontsize="small")
            ax.grid(True)
    fig.tight_layout()
    fig.savefig("combined_plot.png", bbox_inches="tight")
    plt.close(fig)

    # Additional plot: One subplot per dataset (2 subplots)
    fig2, axes2 = plt.subplots(1, len(datasets), figsize=(12, 6), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax2 = axes2[0][i]
        for model in models:
            model_settings = ["no"] + settings if model == "qwen" else settings
            for setting in model_settings:
                if setting in results[dataset][model]:
                    avg_acc = results[dataset][model][setting]["avg_accuracy"]
                    avg_len = results[dataset][model][setting]["avg_token_length"]
                    # 색상 및 마커 설정
                    if setting == "conf":
                        conf_colors = {
                            "qwen": "#ff69b4",
                            "qwq": "#7570b3",
                            "distil": "#d95f02",
                        }
                        color = conf_colors.get(model, "#ff69b4")
                    else:
                        base = model_colors.get(model, "#000000")
                        alpha = setting_intensity.get(setting.replace("_r", ""), 1.0)
                        color = mcolors.to_rgba(base, alpha=alpha)
                    marker = (
                        marker_styles["adaptive"]
                        if "adaptive" in setting
                        else marker_styles["default"]
                    )
                    ax2.scatter(
                        avg_len,
                        avg_acc,
                        label=(
                            f"{model_labels[model]}-Ours"
                            if setting == "adaptive"
                            else f"{model_labels[model]}-{setting.capitalize()}"
                        ),
                        color=color,
                        marker=marker,
                    )
        ax2.set_xlabel("Avg Token Length", fontsize=12)
        ax2.set_ylabel("Avg Accuracy", fontsize=12)
        ax2.set_title(f"{dataset.upper()}", fontsize=14)
        # Compute dynamic y-axis limit based on min accuracy
        all_accs = [
            results[dataset][model][setting]["avg_accuracy"]
            for model in models
            for setting in (["no"] + settings if model == "qwen" else settings)
            if setting in results[dataset][model]
        ]
        ymin = min(all_accs) - 0.02 if all_accs else 0.3
        ax2.set_ylim(ymin, 1.0)
        ax2.grid(True)
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(loc="lower right", fontsize="x-small", ncol=2)
    fig2.tight_layout()
    fig2.savefig("per_dataset_plot.png", bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
