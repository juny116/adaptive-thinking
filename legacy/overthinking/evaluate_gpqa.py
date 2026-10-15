import warnings

warnings.filterwarnings("ignore")

import os
import json
import sys
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np


def read_jsonl(file_path: str) -> List[Dict]:
    """Read .jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_answer(text: str) -> str:
    """Extract answer from generated text. Look for patterns like (A), A), A., etc."""
    text = text.strip()

    # Pattern 1: (A), (B), (C), (D)
    pattern1 = r"\(([ABCD])\)"
    matches1 = re.findall(pattern1, text)

    # Pattern 2: A), B), C), D)
    pattern2 = r"([ABCD])\)"
    matches2 = re.findall(pattern2, text)

    # Pattern 3: A., B., C., D.
    pattern3 = r"([ABCD])\."
    matches3 = re.findall(pattern3, text)

    # Pattern 4: Just A, B, C, D at the end
    pattern4 = r"\b([ABCD])\b"
    matches4 = re.findall(pattern4, text)

    # Try patterns in order of specificity
    for matches in [matches1, matches2, matches3, matches4]:
        if matches:
            # Return the last match (most likely the final answer)
            return matches[-1]

    # If no clear pattern found, look for "answer is" patterns
    # Updated patterns to match research standards: answer is \((C)\)|Answer: \((C)\)|answer: \((C)\)|answer \((C)\)
    answer_patterns = [
        r"answer is \(([ABCD])\)",
        r"Answer: \(([ABCD])\)",
        r"answer: \(([ABCD])\)",
        r"answer \(([ABCD])\)",
        r"correct answer is \(([ABCD])\)",
        r"the answer is \(([ABCD])\)",
        # Fallback patterns without parentheses
        r"answer is ([ABCD])",
        r"correct answer is ([ABCD])",
        r"the answer is ([ABCD])",
        r"answer: ([ABCD])",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"


def evaluate_single_sample(result: Dict) -> Dict:
    """Evaluate a single result with one sample."""
    correct_answer = result["answer"]

    if len(result["messages"]) < 2:
        return {
            "correct": False,
            "predicted": "UNKNOWN",
            "correct_answer": correct_answer,
            "confidence": 0.0,
        }

    generated_text = result["messages"][1]["content"]
    predicted_answer = extract_answer(generated_text)

    is_correct = predicted_answer == correct_answer

    return {
        "correct": is_correct,
        "predicted": predicted_answer,
        "correct_answer": correct_answer,
        "confidence": 1.0 if predicted_answer != "UNKNOWN" else 0.0,
    }


def evaluate_multiple_samples(result: Dict) -> Dict:
    """Evaluate a result with multiple samples using majority voting."""
    correct_answer = result["answer"]
    predictions = []

    for message_pair in result["messages"]:
        if len(message_pair) >= 2:
            generated_text = message_pair[1]["content"]
            predicted_answer = extract_answer(generated_text)
            predictions.append(predicted_answer)

    if not predictions:
        return {
            "correct": False,
            "predicted": "UNKNOWN",
            "correct_answer": correct_answer,
            "confidence": 0.0,
            "predictions": [],
            "majority_vote": "UNKNOWN",
        }

    # Count votes
    vote_counts = defaultdict(int)
    for pred in predictions:
        vote_counts[pred] += 1

    # Find majority vote
    if vote_counts:
        majority_vote = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        confidence = vote_counts[majority_vote] / len(predictions)
    else:
        majority_vote = "UNKNOWN"
        confidence = 0.0

    is_correct = majority_vote == correct_answer

    return {
        "correct": is_correct,
        "predicted": majority_vote,
        "correct_answer": correct_answer,
        "confidence": confidence,
        "predictions": predictions,
        "majority_vote": majority_vote,
        "vote_counts": dict(vote_counts),
    }


def calculate_metrics(evaluations: List[Dict]) -> Dict:
    """Calculate overall metrics."""
    total = len(evaluations)
    correct = sum(1 for eval_result in evaluations if eval_result["correct"])

    accuracy = correct / total if total > 0 else 0.0

    # Calculate confidence statistics
    confidences = [eval_result["confidence"] for eval_result in evaluations]
    avg_confidence = np.mean(confidences) if confidences else 0.0

    # Count unknown predictions
    unknown_count = sum(
        1 for eval_result in evaluations if eval_result["predicted"] == "UNKNOWN"
    )
    unknown_rate = unknown_count / total if total > 0 else 0.0

    # Count by answer choice
    answer_distribution = defaultdict(int)
    for eval_result in evaluations:
        answer_distribution[eval_result["predicted"]] += 1

    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "average_confidence": avg_confidence,
        "unknown_predictions": unknown_count,
        "unknown_rate": unknown_rate,
        "answer_distribution": dict(answer_distribution),
    }


def print_detailed_results(
    evaluations: List[Dict], metrics: Dict, show_errors: bool = False
):
    """Print detailed evaluation results."""
    print("=" * 60)
    print("GPQA EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Average Confidence: {metrics['average_confidence']:.3f}")
    print(
        f"Unknown Predictions: {metrics['unknown_predictions']} ({metrics['unknown_rate']*100:.1f}%)"
    )

    print("\nAnswer Distribution:")
    for answer, count in sorted(metrics["answer_distribution"].items()):
        percentage = count / metrics["total_questions"] * 100
        print(f"  {answer}: {count} ({percentage:.1f}%)")

    if show_errors:
        print("\nIncorrect Predictions:")
        print("-" * 40)
        for i, eval_result in enumerate(evaluations):
            if not eval_result["correct"]:
                print(f"Question {i+1}:")
                print(f"  Correct: {eval_result['correct_answer']}")
                print(f"  Predicted: {eval_result['predicted']}")
                if "predictions" in eval_result:
                    print(f"  All predictions: {eval_result['predictions']}")
                    print(f"  Vote counts: {eval_result['vote_counts']}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPQA generation results")
    parser.add_argument(
        "input_file", help="Path to the generated results file (.jsonl)"
    )
    parser.add_argument(
        "--output", "-o", help="Path to save evaluation results (optional)"
    )
    parser.add_argument(
        "--show-errors", action="store_true", help="Show detailed error analysis"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input_file}")
    try:
        results = read_jsonl(args.input_file)
        print(f"Loaded {len(results)} results")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Determine evaluation type based on structure
    if not results:
        print("No results found in the file")
        sys.exit(1)

    sample_result = results[0]
    has_multiple_samples = (
        "messages" in sample_result
        and len(sample_result["messages"]) > 0
        and isinstance(sample_result["messages"][0], list)
    )

    if has_multiple_samples:
        print("Detected multiple samples per question - using majority voting")
        evaluate_func = evaluate_multiple_samples
    else:
        print("Detected single sample per question - using direct evaluation")
        evaluate_func = evaluate_single_sample

    # Evaluate all results
    print("Evaluating results...")
    evaluations = []
    for result in tqdm(results, desc="Evaluating"):
        evaluation = evaluate_func(result)
        evaluations.append(evaluation)

    # Calculate metrics
    metrics = calculate_metrics(evaluations)

    # Print results
    if not args.quiet:
        print_detailed_results(evaluations, metrics, args.show_errors)
    else:
        print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")

    # Save results if requested
    if args.output:
        output_data = {"metrics": metrics, "evaluations": evaluations}

        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")


if __name__ == "__main__":
    main()
