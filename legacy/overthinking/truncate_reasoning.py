from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time
import json
from tqdm import tqdm
from typing import List
from answer_processing import (
    read_jsonl,
    remove_boxed,
    last_boxed_only_string,
)

# from omegaconf import DictConfig
# import hydra
import os
import logging
import string
import argparse
from transformers import AutoTokenizer


def main(llm=None, tokenizer=None, sampling_params=None, data=None):
    results = []
    for d in tqdm(data, desc="Processing samples"):
        # 원본 데이터 복사
        result_d = d.copy()
        early_exit_results = []

        # parse the think content
        message = d["messages"]
        content = message[-1]["content"]

        if "<think>" in content and "</think>" in content:
            think_content = content.split("<think>")[1].split("</think>")[0]
        else:
            # if no <think> tag, use the whole content
            think_content = content

        # use 10, 20, 30, 40, 50, 60, 70, 80, 90 100 percent of the think content
        think_length = len(tokenizer(think_content)["input_ids"])
        think_percentages = range(10, 101, 10)

        # 모든 프롬프트를 배치로 준비
        prompts = []
        for percentage in think_percentages:
            length_tokens = int(think_length * percentage / 100)
            # 토큰 단위로 자르기 위해 tokenize 후 truncate
            think_tokens = tokenizer(think_content, add_special_tokens=False)[
                "input_ids"
            ]
            truncated_tokens = think_tokens[:length_tokens]
            truncated_think_content = tokenizer.decode(
                truncated_tokens, skip_special_tokens=True
            )

            # add <think> and </think> tags
            truncated_think_content = f"<think>{truncated_think_content}</think>"
            new_message = message[:-1] + [
                {"role": "assistant", "content": truncated_think_content}
            ]

            prompt = tokenizer.apply_chat_template(
                new_message, tokenize=False, add_generation_prompt=True
            ).strip()
            prompts.append(prompt)

        # vLLM으로 배치 inference 수행
        outputs = llm.generate(prompts, sampling_params)

        # 결과 저장
        for i, (percentage, output) in enumerate(zip(think_percentages, outputs)):
            generated_text = output.outputs[0].text

            early_exit_results.append(
                {
                    "percentage": percentage,
                    "generated_text": generated_text,
                    "truncated_think_length": int(think_length * percentage / 100),
                }
            )

        # early_exit 결과를 원본 데이터에 추가
        result_d["early_exit"] = early_exit_results
        results.append(result_d)

    return results


if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="Overthinking analysis script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name or path",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="chat_templates/qwen3.jinja",
        help="Path to chat template file",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="sample_results/olympiad/qwen-32b",
        help="Base path for input and output files",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.75,
        help="GPU memory utilization ratio",
    )
    args = parser.parse_args()

    # 모델과 토크나이저 설정
    model_name = args.model_name

    # vLLM 모델 초기화
    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.float16,  # 메모리 효율성을 위해
        trust_remote_code=True,
        download_dir="/hf/hub",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # llm = None
    # 토크나이저 초기화 (chat template 사용을 위해)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(args.chat_template, "r") as f:
        custom_template = f.read()
    tokenizer.chat_template = custom_template
    # Sampling parameters 설정
    # sampling_params = SamplingParams(
    #     temperature=0.6,
    #     top_p=0.95,
    #     top_k=20,
    #     min_p=0.0,
    #     max_tokens=30000,
    # )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=30000,
    )

    base_path = args.base_path
    file_path = os.path.join(base_path, "overthinking.jsonl")
    data = read_jsonl(file_path)

    results = main(llm, tokenizer, sampling_params, data)

    # 결과를 새로운 파일에 저장
    output_path = os.path.join(base_path, "overthinking_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n결과가 '{output_path}'에 저장되었습니다.")
    print(f"총 {len(results)}개 샘플, 각각 9개의 early exit 결과 포함")
