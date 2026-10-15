"""Result summaries used by the generation strategies."""

from adaptive_thinking.data.jsonl import read_jsonl

from .answers import compute_reward, is_valid_boxed_answer
from .gpqa import compute_gpqa_reward, is_valid_choice_answer


def summarize_legacy(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0

    results = read_jsonl(config["result_path"])
    for r in results:
        output = (
            r["messages"][2]["content"]
            if isinstance(r["messages"], list) and len(r["messages"]) > 2
            else ""
        )

        if not is_valid_boxed_answer(output):
            invalid += 1
            continue

        reward = compute_reward(output, r["answer"])
        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")
    else:
        print("No examples were evaluated.")


def summarize_baseline(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0
    token_lengths = []  # Track token lengths for valid answers

    results = read_jsonl(config["result_path"])
    for r in results:
        # Get the last assistant message
        output = ""
        for msg in reversed(r["messages"]):
            if msg.get("role") == "assistant":
                output = msg.get("content", "")
                break

        if data_type == "math":
            if not is_valid_boxed_answer(output):
                invalid += 1
                continue
            reward = compute_reward(output, r["answer"])
        elif data_type == "gpqa":
            if not is_valid_choice_answer(output):
                invalid += 1
                continue
            reward = compute_gpqa_reward(output, r["answer"])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
            # Track token length for correct answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)
        else:
            incorrect += 1
            # Track token length for incorrect answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")

        # Calculate and print average token length
        if token_lengths:
            avg_token_length = sum(token_lengths) / len(token_lengths)
            max_token_length = max(token_lengths)
            min_token_length = min(token_lengths)
            print(f"\nToken Statistics (for valid answers):")
            print(f"  Average token length: {avg_token_length:.2f}")
            print(f"  Max token length: {max_token_length}")
            print(f"  Min token length: {min_token_length}")
    else:
        print("No examples were evaluated.")


def summarize_adaptive(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0
    invalid_samples = []  # Track invalid samples for debugging
    token_lengths = []  # Track token lengths for valid answers

    results = read_jsonl(config["result_path"])
    for r in results:
        # Get the last assistant message (not hardcoded index 2)
        output = ""
        for msg in reversed(r["messages"]):
            if msg.get("role") == "assistant":
                output = msg.get("content", "")
                break

        if not output:
            invalid += 1
            invalid_samples.append(
                {
                    "index": r["index"],
                    "reason": "no_assistant_message",
                    "num_messages": len(r["messages"]),
                }
            )
            continue

        if data_type == "math":
            if not is_valid_boxed_answer(output):
                invalid += 1
                invalid_samples.append(
                    {
                        "index": r["index"],
                        "reason": "no_boxed_answer",
                        "output_last_200": output[-200:] if output else "EMPTY",
                    }
                )
                continue

            reward = compute_reward(output, r["answer"])
            if reward is None:
                invalid += 1
                invalid_samples.append(
                    {"index": r["index"], "reason": "parsing_failed", "output": output}
                )
            elif reward:
                correct += 1
                # Track token length for correct answers
                token_len = len(model.tokenizer(output)["input_ids"])
                token_lengths.append(token_len)
            else:
                incorrect += 1
                # Track token length for incorrect answers
                token_len = len(model.tokenizer(output)["input_ids"])
                token_lengths.append(token_len)
        elif data_type == "gpqa":
            if not is_valid_choice_answer(output):
                invalid += 1
                invalid_samples.append(
                    {
                        "index": r["index"],
                        "reason": "no_valid_choice",
                        "output_last_300": output[-300:] if output else "EMPTY",
                        "correct_answer": r["answer"],
                    }
                )
                continue

            reward = compute_gpqa_reward(output, r["answer"])
            if reward is None:
                invalid += 1
                invalid_samples.append(
                    {
                        "index": r["index"],
                        "reason": "parsing_failed",
                        "output": output,
                        "correct_answer": r["answer"],
                    }
                )
            elif reward:
                correct += 1
                # Track token length for correct answers
                token_len = len(model.tokenizer(output)["input_ids"])
                token_lengths.append(token_len)
            else:
                incorrect += 1
                # Track token length for incorrect answers
                token_len = len(model.tokenizer(output)["input_ids"])
                token_lengths.append(token_len)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")

        # Calculate and print average token length
        if token_lengths:
            avg_token_length = sum(token_lengths) / len(token_lengths)
            max_token_length = max(token_lengths)
            min_token_length = min(token_lengths)
            print(f"\nToken Statistics (for valid answers):")
            print(f"  Average token length: {avg_token_length:.2f}")
            print(f"  Max token length: {max_token_length}")
            print(f"  Min token length: {min_token_length}")

        if invalid_samples and len(invalid_samples) <= 10:
            print(f"\n[DEBUG] Invalid samples (first {len(invalid_samples)}):")
            for sample in invalid_samples[:10]:
                print(f"  Index {sample['index']}: {sample}")
    else:
        print("No examples were evaluated.")


def summarize_deer(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0
    token_lengths = []  # Track token lengths for valid answers

    results = read_jsonl(config["result_path"])
    for r in results:
        # Find the final assistant message
        output = ""
        if isinstance(r["messages"], list) and len(r["messages"]) > 0:
            # Get the last assistant message (final answer)
            for msg in reversed(r["messages"]):
                if msg.get("role") == "assistant":
                    output = msg.get("content", "")
                    break

        if data_type == "math":
            if not is_valid_boxed_answer(output):
                invalid += 1
                continue
            reward = compute_reward(output, r["answer"])
        elif data_type == "gpqa":
            if not is_valid_choice_answer(output):
                invalid += 1
                continue
            reward = compute_gpqa_reward(output, r["answer"])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
            # Track token length for correct answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)
        else:
            incorrect += 1
            # Track token length for incorrect answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")

        # Calculate and print average token length
        if token_lengths:
            avg_token_length = sum(token_lengths) / len(token_lengths)
            max_token_length = max(token_lengths)
            min_token_length = min(token_lengths)
            print(f"\nToken Statistics (for valid answers):")
            print(f"  Average token length: {avg_token_length:.2f}")
            print(f"  Max token length: {max_token_length}")
            print(f"  Min token length: {min_token_length}")
    else:
        print("No examples were evaluated.")


def summarize_fixed(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0
    token_lengths = []  # Track token lengths for valid answers

    results = read_jsonl(config["result_path"])
    for r in results:
        # Get the last assistant message (not hardcoded index 2)
        output = ""
        for msg in reversed(r["messages"]):
            if msg.get("role") == "assistant":
                output = msg.get("content", "")
                break

        if not is_valid_boxed_answer(output):
            invalid += 1
            continue

        reward = compute_reward(output, r["answer"])
        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
            # Track token length for correct answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)
        else:
            incorrect += 1
            # Track token length for incorrect answers
            token_len = len(model.tokenizer(output)["input_ids"])
            token_lengths.append(token_len)

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")

        # Calculate and print average token length
        if token_lengths:
            avg_token_length = sum(token_lengths) / len(token_lengths)
            max_token_length = max(token_lengths)
            min_token_length = min(token_lengths)
            print(f"\nToken Statistics (for valid answers):")
            print(f"  Average token length: {avg_token_length:.2f}")
            print(f"  Max token length: {max_token_length}")
            print(f"  Min token length: {min_token_length}")
    else:
        print("No examples were evaluated.")


def summarize_thinking(config, model):
    data_type = config.get("data_type", "math")
    print("Evaluating answers from result file...")
    correct = 0
    incorrect = 0
    invalid = 0

    results = read_jsonl(config["result_path"])
    for r in results:
        # Find the final assistant message
        output = ""
        if isinstance(r["messages"], list) and len(r["messages"]) > 0:
            # Get the last assistant message (final answer)
            for msg in reversed(r["messages"]):
                if msg.get("role") == "assistant":
                    output = msg.get("content", "")
                    break

        if data_type == "math":
            if not is_valid_boxed_answer(output):
                invalid += 1
                continue
            reward = compute_reward(output, r["answer"])
        elif data_type == "gpqa":
            if not is_valid_choice_answer(output):
                invalid += 1
                continue
            reward = compute_gpqa_reward(output, r["answer"])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if reward is None:
            invalid += 1
        elif reward:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect + invalid
    if total > 0:
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Invalid: {invalid}")
        print(f"Accuracy: {correct / total:.4f}")
    else:
        print("No examples were evaluated.")
