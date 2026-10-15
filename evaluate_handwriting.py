"""Read a photo of a handwritten answer, convert it to text with TrOCR, and mark it.

    python evaluate_handwriting.py path/to/photo.jpg Q1

The second argument is the question ID from data/questions.json (Q1 to Q6).
Leave it out to only convert the handwriting to text.
The first run downloads the TrOCR model (about 1.3 GB).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from evaluator.ocr import HandwritingOCR, segment_lines


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = Path(sys.argv[1])
    question_id = sys.argv[2] if len(sys.argv) > 2 else None
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    image = Image.open(image_path)
    print(f"Found {len(segment_lines(image))} line(s) of handwriting.")
    print("Loading TrOCR (the first run downloads the model)...")
    text, lines = HandwritingOCR().transcribe(image)

    print("\nRecognised text, line by line:")
    for i, line in enumerate(lines, 1):
        print(f"  {i}. {line}")
    print(f"\nFull answer:\n  {text}\n")

    if question_id:
        from evaluator import AnswerEvaluator

        ev = AnswerEvaluator()
        if question_id not in ev.questions:
            sys.exit(f"Unknown question ID {question_id}. Choose from: {', '.join(ev.questions)}")
        res = ev.evaluate_question(question_id, text)
        print(f"Question: {ev.questions[question_id]['question']}")
        print(f"Score: {res.score:g} / {res.max_marks:g}")
        print(f"Similarity {res.semantic_similarity:.0%}  |  Concepts {res.concept_coverage:.0%}  |  "
              f"Completeness {res.completeness:.0%}")
        for c in res.concepts:
            print(f"  [{'x' if c['found'] else ' '}] {c['name']}")
        print(f"\nFeedback: {res.feedback}")


if __name__ == "__main__":
    main()
