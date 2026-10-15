import json
import re
from typing import List, Dict, Optional
from collections import Counter


def read_jsonl(file_path: str) -> List[Dict]:
    """Read .jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    # Look for \boxed{...} pattern
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return the last match

    # Fallback: look for common answer patterns
    # Try to find single letter answers (A, B, C, D)
    letter_pattern = r"\b([ABCD])\b"
    letter_matches = re.findall(letter_pattern, text)
    if letter_matches:
        return letter_matches[-1]

    return None


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract choice letter (A, B, C, D) from text."""
    # First try boxed format
    boxed = extract_boxed_answer(text)
    if boxed and boxed.upper() in ["A", "B", "C", "D"]:
        return boxed.upper()

    # Look for patterns like "The answer is A", "Choice A", etc.
    patterns = [
        r"(?:answer|choice)\s+is\s+([ABCD])",
        r"(?:answer|choice):\s*([ABCD])",
        r"(?:^|\s)([ABCD])(?:\s|$|\.)",
        r"option\s+([ABCD])",
        r"letter\s+([ABCD])",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


def calculate_accuracy(results: List[Dict]) -> Dict:
    """Calculate accuracy metrics."""
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    total = len(results)
    correct = 0
    extraction_success = 0

    for result in results:
        gold = result.get(
            "answer", ""
        ).upper()  # Changed from "gold_answer" to "answer"

        # Handle sample.jsonl format with messages
        if "messages" in result:
            # Get all assistant responses
            assistant_responses = []
            for message_list in result["messages"]:
                for message in message_list:
                    if message["role"] == "assistant":
                        assistant_responses.append(message["content"])

            if assistant_responses:
                if len(assistant_responses) == 1:
                    # Single sample
                    pred = extract_choice_letter(assistant_responses[0])
                    if pred:
                        extraction_success += 1
                        if pred == gold:
                            correct += 1
                else:
                    # Multiple samples - use majority vote
                    predictions = []
                    for response in assistant_responses:
                        pred = extract_choice_letter(response)
                        if pred:
                            predictions.append(pred)

                    if predictions:
                        extraction_success += 1
                        # Majority vote
                        vote_counts = Counter(predictions)
                        majority_pred = vote_counts.most_common(1)[0][0]
                        if majority_pred == gold:
                            correct += 1

        # Handle legacy format (for backward compatibility)
        elif "generated_response" in result:
            pred = extract_choice_letter(result["generated_response"])
            if pred:
                extraction_success += 1
                if pred == gold:
                    correct += 1
        elif "samples" in result:
            # Legacy multiple samples format
            predictions = []
            for sample in result["samples"]:
                pred = extract_choice_letter(sample["generated_response"])
                if pred:
                    predictions.append(pred)

            if predictions:
                extraction_success += 1
                # Majority vote
                vote_counts = Counter(predictions)
                majority_pred = vote_counts.most_common(1)[0][0]
                if majority_pred == gold:
                    correct += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "extraction_success_rate": extraction_success / total if total > 0 else 0.0,
        "extraction_success": extraction_success,
    }


def analyze_results(file_path: str) -> None:
    """Analyze results from GPQA inference."""
    print(f"Analyzing results from: {file_path}")

    try:
        results = read_jsonl(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not results:
        print("No results found in file")
        return

    print(f"Total questions: {len(results)}")

    # Calculate metrics
    metrics = calculate_accuracy(results)

    print(f"\nAccuracy Metrics:")
    print(
        f"  Overall Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})"
    )
    print(
        f"  Answer Extraction Success: {metrics['extraction_success_rate']:.1%} ({metrics['extraction_success']}/{metrics['total']})"
    )

    # Show some examples
    print(f"\nExample predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\nQuestion {i+1}:")
        print(
            f"  Gold: {result.get('answer', 'N/A')}"
        )  # Changed from 'gold_answer' to 'answer'

        # Handle sample.jsonl format with messages
        if "messages" in result:
            assistant_responses = []
            for message_list in result["messages"]:
                for message in message_list:
                    if message["role"] == "assistant":
                        assistant_responses.append(message["content"])

            if len(assistant_responses) == 1:
                pred = extract_choice_letter(assistant_responses[0])
                print(f"  Predicted: {pred}")
                print(f"  Response snippet: {assistant_responses[0][:200]}...")
            else:
                predictions = []
                for response in assistant_responses:
                    pred = extract_choice_letter(response)
                    predictions.append(pred)
                print(f"  Predictions: {predictions}")
                if predictions:
                    vote_counts = Counter(predictions)
                    majority_pred = vote_counts.most_common(1)[0][0]
                    print(f"  Majority vote: {majority_pred}")

        # Handle legacy format
        elif "generated_response" in result:
            pred = extract_choice_letter(result["generated_response"])
            print(f"  Predicted: {pred}")
            print(f"  Response snippet: {result['generated_response'][:200]}...")
        elif "samples" in result:
            predictions = []
            for sample in result["samples"]:
                pred = extract_choice_letter(sample["generated_response"])
                predictions.append(pred)
            print(f"  Predictions: {predictions}")
            if predictions:
                vote_counts = Counter(predictions)
                majority_pred = vote_counts.most_common(1)[0][0]
                print(f"  Majority vote: {majority_pred}")

    # Answer distribution
    all_predictions = []
    all_golds = []

    for result in results:
        gold = result.get(
            "answer", ""
        ).upper()  # Changed from 'gold_answer' to 'answer'
        if gold:
            all_golds.append(gold)

        # Handle sample.jsonl format with messages
        if "messages" in result:
            assistant_responses = []
            for message_list in result["messages"]:
                for message in message_list:
                    if message["role"] == "assistant":
                        assistant_responses.append(message["content"])

            if len(assistant_responses) == 1:
                pred = extract_choice_letter(assistant_responses[0])
                if pred:
                    all_predictions.append(pred)
            else:
                predictions = []
                for response in assistant_responses:
                    pred = extract_choice_letter(response)
                    if pred:
                        predictions.append(pred)
                if predictions:
                    vote_counts = Counter(predictions)
                    majority_pred = vote_counts.most_common(1)[0][0]
                    all_predictions.append(majority_pred)

        # Handle legacy format
        elif "generated_response" in result:
            pred = extract_choice_letter(result["generated_response"])
            if pred:
                all_predictions.append(pred)
        elif "samples" in result:
            predictions = []
            for sample in result["samples"]:
                pred = extract_choice_letter(sample["generated_response"])
                if pred:
                    predictions.append(pred)
            if predictions:
                vote_counts = Counter(predictions)
                majority_pred = vote_counts.most_common(1)[0][0]
                all_predictions.append(majority_pred)

    print(f"\nAnswer Distribution:")
    print(f"  Gold answers: {dict(Counter(all_golds))}")
    print(f"  Predicted answers: {dict(Counter(all_predictions))}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.jsonl>")
        sys.exit(1)

    analyze_results(sys.argv[1])
