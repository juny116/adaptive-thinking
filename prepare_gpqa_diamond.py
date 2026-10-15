"""
Script to prepare GPQA Diamond dataset for evaluation.
Downloads from HuggingFace and converts to the required format.
"""
import json
from datasets import load_dataset
from pathlib import Path


def prepare_gpqa_diamond(output_path="data/gpqa_diamond.jsonl", num_examples=None):
    """
    Download GPQA Diamond dataset and convert to evaluation format.
    
    Args:
        output_path: Where to save the processed dataset
        num_examples: Maximum number of examples to include (None for all)
    """
    print("Downloading GPQA Diamond dataset...")
    dataset = load_dataset("fingertap/GPQA-Diamond", split="test")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    with open(output_path, "w") as f:
        for idx, example in enumerate(dataset):
            if num_examples and idx >= num_examples:
                break
            
            # GPQA Diamond format: question, choice_a, choice_b, choice_c, choice_d, answer
            processed_example = {
                "question": example["question"],
                "choice_a": example.get("choice_a", example.get("A", "")),
                "choice_b": example.get("choice_b", example.get("B", "")),
                "choice_c": example.get("choice_c", example.get("C", "")),
                "choice_d": example.get("choice_d", example.get("D", "")),
                "answer": example.get("answer", ""),
                "solution": example.get("solution", ""),  # if available
            }
            
            f.write(json.dumps(processed_example) + "\n")
            processed_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} examples...")
    
    print(f"✓ Successfully processed {processed_count} examples")
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    prepare_gpqa_diamond()
