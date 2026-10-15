"""Confidence labels, stop delimiters and parser."""

import re

confidence_class = {
    "Almost no chance": 0,
    "Highly unlikely": 1,
    "Chances are slight": 2,
    "Unlikely": 3,
    "Less than even": 4,
    "Better than even": 5,
    "Likely": 6,
    "Very good chance": 7,
    "Highly likely": 8,
    "Almost certain": 9,
}

end_parentheses = [
    "}",
    "}\n",
    "}\n\n",
    "}\n\n\n",
    "}\n\n\n\n",
    "}\n\n\n\n\n",
    "}\n\n\n\n\n\n",
    " }",
    "} ",
    " }\n",
]


def parse_confidence(text):
    """
    Extracts the last occurring confidence level from a string in the format \\confidence{X}.
    Returns the confidence level (X) if found, else returns None.
    """
    matches = re.findall(r"\\confidence\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None
