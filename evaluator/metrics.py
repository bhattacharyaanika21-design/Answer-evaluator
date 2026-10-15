"""Agreement metrics between predicted and human (teacher) scores."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def regression_metrics(human, predicted) -> dict:
    h, p = np.asarray(human, float), np.asarray(predicted, float)
    err = p - h
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "pearson": float(pearsonr(h, p)[0]),
        "spearman": float(spearmanr(h, p)[0]),
        # Quadratic weighted kappa on whole marks, the standard essay-scoring metric
        "qwk": float(cohen_kappa_score(np.rint(h).astype(int), np.rint(p).astype(int), weights="quadratic")),
        "within_1_mark": float(np.mean(np.abs(err) <= 1.0)),
    }


def concept_detection_metrics(gold: list[set], detected: list[set]) -> dict:
    tp = sum(len(g & d) for g, d in zip(gold, detected))
    fp = sum(len(d - g) for g, d in zip(gold, detected))
    fn = sum(len(g - d) for g, d in zip(gold, detected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
