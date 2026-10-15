"""Scoring engines that turn evaluation features into marks.

* RuleScorer    - transparent weighted formula (similarity / coverage / completeness).
* LearnedScorer - a regression model trained on teacher-marked answers (train.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "semantic_similarity",   # rescaled cosine(student, reference), 0..1
    "concept_coverage",      # weighted share of rubric concepts found, 0..1
    "completeness",          # reference-sentence coverage blended with length, 0..1
    "length_ratio",          # student words / reference words, capped at 1.5
    "raw_cosine",            # unscaled cosine similarity
    "keyword_only_share",    # share of concept weight found only via keywords, 0..1
]


class RuleScorer:
    name = "rule"

    def __init__(self, w_similarity: float = 0.5, w_coverage: float = 0.3, w_completeness: float = 0.2):
        total = w_similarity + w_coverage + w_completeness
        self.w = (w_similarity / total, w_coverage / total, w_completeness / total)

    def predict(self, features: dict, max_marks: float) -> float:
        s = (self.w[0] * features["semantic_similarity"]
             + self.w[1] * features["concept_coverage"]
             + self.w[2] * features["completeness"])
        return float(np.clip(s, 0, 1) * max_marks)


class LearnedScorer:
    """Wraps a scikit-learn regressor trained to predict the fraction of marks (0..1)."""

    name = "learned"

    def __init__(self, model, backend: str, model_type: str = ""):
        self.model = model
        self.backend = backend
        self.model_type = model_type

    def predict(self, features: dict, max_marks: float) -> float:
        x = np.array([[features[f] for f in FEATURE_NAMES]])
        return float(np.clip(self.model.predict(x)[0], 0, 1) * max_marks)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "backend": self.backend, "model_type": self.model_type,
                     "features": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedScorer":
        import joblib

        blob = joblib.load(path)
        if blob.get("features") != FEATURE_NAMES:
            raise ValueError("Saved model was trained with a different feature set; retrain it.")
        return cls(blob["model"], blob["backend"], blob.get("model_type", ""))
