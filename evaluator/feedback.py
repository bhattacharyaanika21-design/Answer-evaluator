"""Turns the evaluation signals into short, constructive feedback for the student."""
from __future__ import annotations


def generate_feedback(score: float, max_marks: float, similarity: float, coverage: float,
                      completeness: float, concepts: list[dict], word_count: int,
                      reference_word_count: int) -> str:
    pct = score / max_marks if max_marks else 0.0
    found = [c for c in concepts if c["found"]]
    missing = sorted((c for c in concepts if not c["found"]), key=lambda c: -c["weight"])
    lines: list[str] = []

    if word_count < 3:
        return "The answer is too short to evaluate. Write a full explanation in your own words."

    if pct >= 0.85:
        lines.append("Excellent answer: it explains the idea clearly and covers almost every key point.")
    elif pct >= 0.65:
        lines.append("Good answer: the main idea is correct, but a few important points are missing.")
    elif pct >= 0.40:
        lines.append("Partially correct: the answer touches on the topic but leaves out several key points.")
    elif similarity < 0.15 and not found:
        lines.append("The answer does not appear to address this question. Re-read the question and try again.")
    else:
        lines.append("The answer needs significant improvement: most of the expected content is missing or incorrect.")

    if concepts:
        lines.append(f"You covered {len(found)} of {len(concepts)} key concepts.")
    if missing and pct < 0.95:
        names = ", ".join(c["name"].lower() for c in missing[:3])
        lines.append(f"To improve, explain: {names}.")

    # Keyword stuffing: many concepts named, but little explanation around them.
    if coverage >= 0.5 and similarity < 0.45 and word_count < 0.35 * reference_word_count:
        lines.append("You list the right terms but do not explain them; add a sentence on what each one means.")
    elif completeness < 0.35 and word_count < 0.3 * reference_word_count and pct < 0.85:
        lines.append("The answer is brief compared with what is expected; develop each point further.")

    return " ".join(lines)
