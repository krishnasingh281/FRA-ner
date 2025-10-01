import re
from statistics import median

COMMON_FIXES = [
    (r"\s+", " "),        # Normalize whitespace
    (r"(?i)narne", "Name"), # Example: fix a common OCR error for "Name"
    (r"0f", "of"),
]

def apply_common_fixes(text: str) -> str:
    """Applies a list of regular expression fixes to the text."""
    for pattern, replacement in COMMON_FIXES:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def group_words_into_lines(words: list, y_tolerance: int = 10) -> list:
    """Groups individual words into lines based on their vertical position."""
    if not words:
        return []

    # Calculate the center y-coordinate for each word
    for w in words:
        ys = [p[1] for p in w["bbox"]]
        w["cy"] = sum(ys) / len(ys) if ys else 0

    words.sort(key=lambda w: (w["cy"], w["bbox"][0][0]))

    lines = []
    current_line_words = [words[0]]

    for word in words[1:]:
        median_y_current_line = median([w["cy"] for w in current_line_words])
        if abs(word["cy"] - median_y_current_line) <= y_tolerance:
            current_line_words.append(word)
        else:
            # Finalize the current line and start a new one
            current_line_words.sort(key=lambda w: w["bbox"][0][0])
            lines.append({
                "text": " ".join([w["text"] for w in current_line_words]),
                "words": current_line_words
            })
            current_line_words = [word]
    
    # Add the last line
    current_line_words.sort(key=lambda w: w["bbox"][0][0])
    lines.append({
        "text": " ".join([w["text"] for w in current_line_words]),
        "words": current_line_words
    })
    
    return lines