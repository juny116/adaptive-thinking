import warnings

warnings.filterwarnings("ignore")

import os
import json
import time
import sys
import string
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from omegaconf import DictConfig
import hydra


def read_jsonl(file_path: str) -> List[Dict]:
    """Read .jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data: List[Dict], file_path: str) -> None:
    """Write data to .jsonl file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_prompt(
    data: Dict, template_instruction: str, template_format: str, fields: List[str]
) -> str:
    """Format question data into a prompt based on template configuration."""
    # Create format dict with all required fields
    format_dict = {}

    # Basic question field
    format_dict["question"] = data.get("question", "")

    # Options - extract from options dict if available
    if "options" in data and isinstance(data["options"], dict):
        options = data["options"]
        format_dict["option_A"] = options.get("A", "")
        format_dict["option_B"] = options.get("B", "")
        format_dict["option_C"] = options.get("C", "")
        format_dict["option_D"] = options.get("D", "")

    # Add any other fields that might be referenced in template
    for field in fields:
        if field not in format_dict:
            format_dict[field] = data.get(field, "")

    # Format the template with available fields
    query = template_format.format(**format_dict)
    return template_instruction + query


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(config: DictConfig):
    # Extract config values
    model_config = config["model"]
    template_config = config["template"]

    print(f"Model: {model_config['model_name_or_path']}")
    print(f"Data: {config['data_path']}")
    print(f"Template: {template_config['template_name']}")
    print(f"Temperature: {model_config['temperature']}")
    print(f"Max tokens: {model_config['max_tokens']}")
    print(f"Num samples: {config['num_samples']}")
    print(f"Batch size: {config['batch_size']}")

    # Create output directory
    os.makedirs(config["output_path"], exist_ok=True)

    # Generate output filename if not specified
    if config["output_filename"] is None:
        model_name = os.path.basename(model_config["model_name_or_path"].rstrip("/"))
        output_filename = f"gpqa_{model_name}_temp{model_config['temperature']}_samples{config['num_samples']}_{template_config['template_name']}.jsonl"
    else:
        output_filename = config["output_filename"]

    output_file = os.path.join(config["output_path"], output_filename)
    print(f"Output will be saved to: {output_file}")

    # Load data
    print("\nLoading GPQA dataset...")
    try:
        questions = read_jsonl(config["data_path"])
        if config.get("max_examples"):
            questions = questions[: config["max_examples"]]
        print(f"Loaded {len(questions)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Parse template format to extract fields
    formatter = string.Formatter()
    fields = [
        field for _, field, _, _ in formatter.parse(template_config["format"]) if field
    ]

    # Initialize model
    print("\nInitializing vLLM model...")
    try:
        # Determine number of GPUs
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            tensor_parallel_size = len([gpu for gpu in available_gpus if gpu.strip()])
        else:
            tensor_parallel_size = 1

        llm = LLM(
            model=model_config["model_name_or_path"],
            tensor_parallel_size=tensor_parallel_size,
            dtype=model_config.get("dtype", "float16"),
            gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.75),
            trust_remote_code=model_config.get("trust_remote_code", True),
            seed=config.get("seed", 42),
            disable_custom_all_reduce=True,
            download_dir="/hf/hub",
        )
        print("Model initialized successfully")

        # Load tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_or_path"])
        print("Tokenizer loaded successfully")

    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # Test generation
    print("\nRunning test generation...")
    try:
        test_outputs = llm.generate(
            ["Hello, how are you?"],
            SamplingParams(max_tokens=10, temperature=model_config["temperature"]),
            use_tqdm=False,
        )
        if test_outputs and test_outputs[0].outputs:
            test_text = test_outputs[0].outputs[0].text
            print(f"Test successful: '{test_text.strip()}'")
        else:
            print("Warning: Test generation returned empty output")
    except Exception as e:
        print(f"Test generation failed: {e}")

    # Prepare prompts
    print("\nPreparing prompts...")
    all_prompts = []
    question_indices = []

    for i, question_data in enumerate(questions):
        formatted_prompt = format_prompt(
            question_data,
            template_config["instruction"],
            template_config["format"],
            fields,
        )

        # Generate multiple samples if requested
        for sample_idx in range(config["num_samples"]):
            all_prompts.append(formatted_prompt)
            question_indices.append((i, sample_idx))

    print(
        f"Prepared {len(all_prompts)} prompts ({len(questions)} questions × {config['num_samples']} samples)"
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
        top_p=model_config.get("top_p", 1.0),
        top_k=model_config.get("top_k", 0),
    )

    # Run inference
    print("\nRunning inference...")
    start_time = time.time()

    try:
        # Process in batches
        all_outputs = []
        for i in tqdm(
            range(0, len(all_prompts), config["batch_size"]), desc="Processing batches"
        ):
            batch_prompts = all_prompts[i : i + config["batch_size"]]
            batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            all_outputs.extend(batch_outputs)

        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

    # Process results and format like sample.jsonl
    print("\nProcessing results...")

    if config["num_samples"] == 1:
        # Single sample case - format like sample.jsonl
        results = []
        for output_idx, output in enumerate(all_outputs):
            question_idx, sample_idx = question_indices[output_idx]
            question_data = questions[question_idx]

            if output.outputs:
                generated_text = output.outputs[0].text.strip()
            else:
                generated_text = ""
                print(f"Warning: Empty output for question {question_idx}")

            # Format like sample.jsonl structure
            result = {
                "index": question_idx,
                "answer": question_data["answer"],
                "messages": [
                    {"role": "user", "content": all_prompts[output_idx]},
                    {"role": "assistant", "content": generated_text},
                ],
                "question": question_data["question"],
                "solution": question_data.get("explanation", ""),
            }
            results.append(result)
    else:
        # Multiple samples case
        grouped_results = {}
        for output_idx, output in enumerate(all_outputs):
            question_idx, sample_idx = question_indices[output_idx]
            question_data = questions[question_idx]

            if output.outputs:
                generated_text = output.outputs[0].text.strip()
            else:
                generated_text = ""
                print(
                    f"Warning: Empty output for question {question_idx}, sample {sample_idx}"
                )

            if question_idx not in grouped_results:
                grouped_results[question_idx] = {
                    "index": question_idx,
                    "answer": question_data["answer"],
                    "messages": [],
                    "question": question_data["question"],
                    "solution": question_data.get("explanation", ""),
                }

            # Add message pair for this sample
            grouped_results[question_idx]["messages"].append(
                [
                    {"role": "user", "content": all_prompts[output_idx]},
                    {"role": "assistant", "content": generated_text},
                ]
            )

        # Convert to list
        results = []
        for question_idx in sorted(grouped_results.keys()):
            results.append(grouped_results[question_idx])

    # Save results
    print(f"\nSaving {len(results)} results to {output_file}...")
    try:
        write_jsonl(results, output_file)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    print(
        f"\nCompleted! Processed {len(questions)} questions with {config['num_samples']} samples each."
    )
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
