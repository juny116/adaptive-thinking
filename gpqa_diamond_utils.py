"""
Utilities for GPQA Diamond evaluation - MCQA format with A/B/C/D choices
"""
import re


CHOICE_LETTERS = ["A", "B", "C", "D"]
ANSWER_RE = re.compile(r"[Tt]he answer is \(?([A-D])\)?")


def extract_choice_letter(generated: str) -> str:
    """
    Extract the choice letter (A, B, C, or D) from generated text.
    
    Looks for patterns like:
    - "The answer is A"
    - "The answer is (A)"
    - "the answer is a"
    
    Returns:
        The letter choice (A, B, C, D) or None if not found
    """
    # Find all matches and use the LAST one (in case of multiple "The answer is" statements)
    matches = ANSWER_RE.findall(generated)
    if matches:
        return matches[-1].upper()  # Return the last match
    
    # Fallback: look for any standalone letter at the end
    lines = generated.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.upper() in CHOICE_LETTERS:
            return line.upper()
    
    return None


def compute_gpqa_reward(generated: str, correct_answer: str) -> int:
    """
    Compute reward for GPQA Diamond answers.
    
    Args:
        generated: The model-generated answer text
        correct_answer: The correct answer (should be a letter A/B/C/D)
    
    Returns:
        1 if correct, 0 if incorrect, None if parsing fails
    """
    extracted = extract_choice_letter(generated)
    if extracted is None:
        return None
    
    # Normalize correct answer to uppercase letter
    correct = correct_answer.strip().upper()
    
    if extracted == correct:
        return 1
    else:
        return 0


def is_valid_choice_answer(generated: str) -> bool:
    """
    Check if the generated answer contains a valid choice selection.
    
    Returns:
        True if a valid choice letter (A-D) was extracted, False otherwise
    """
    return extract_choice_letter(generated) is not None
