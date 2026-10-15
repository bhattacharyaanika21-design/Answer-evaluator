"""Train the learned scoring model on teacher-marked answers.

Validation uses leave-one-question-out cross-validation: the model is always tested
on a question it never saw during training, which is how it will be used in practice.

    python train.py                  # Sentence-BERT (falls back to TF-IDF if unavailable)
    python train.py --backend tfidf  # lexical baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluator import AnswerEvaluator
from evaluator.embedder import DEFAULT_SBERT_MODEL
from evaluator.metrics import regression_metrics
from evaluator.scorer import FEATURE_NAMES, LearnedScorer

ROOT = Path(__file__).resolve().parent

CANDIDATES = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                           learning_rate=0.05, random_state=42),
}


def build_feature_table(ev: AnswerEvaluator, answers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in answers.iterrows():
        q = ev.questions[r.question_id]
        res = ev.evaluate_question(r.question_id, r.student_answer)  # rule score as baseline
        detected = ";".join(c["id"] for c in res.concepts if c["found"])
        rows.append({"answer_id": r.answer_id, "question_id": r.question_id,
                     "human_score": r.human_score, "max_marks": q["max_marks"],
                     "rule_score": res.score, "detected_concepts": detected,
                     "gold_concepts": r.concepts_present if isinstance(r.concepts_present, str) else "",
                     **{f: res.features[f] for f in FEATURE_NAMES}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "sbert", "tfidf"])
    ap.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL,
                    help="Hugging Face name or local path, e.g. models/sbert-finetuned")
    ap.add_argument("--data", default=str(ROOT / "data" / "student_answers.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    ev = AnswerEvaluator(backend=args.backend, scoring="rule", sbert_model=args.sbert_model)
    print(f"Embedding backend: {ev.backend}")
    answers = pd.read_csv(args.data)
    table = build_feature_table(ev, answers)

    X = table[FEATURE_NAMES].to_numpy()
    y = (table.human_score / table.max_marks).to_numpy()
    groups = table.question_id.to_numpy()

    summary = {"backend": ev.backend, "n_answers": len(table),
               "n_questions": int(table.question_id.nunique()), "models": {}}
    summary["models"]["rule"] = regression_metrics(table.human_score, table.rule_score)

    best_name, best_mae = None, np.inf
    for name, factory in CANDIDATES.items():
        preds = np.zeros(len(y))
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            model = factory().fit(X[train_idx], y[train_idx])
            preds[test_idx] = np.clip(model.predict(X[test_idx]), 0, 1)
        table[f"{name}_score"] = np.round(preds * table.max_marks * 2) / 2
        m = regression_metrics(table.human_score, table[f"{name}_score"])
        summary["models"][name] = m
        print(f"{name:18s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  Pearson={m['pearson']:.3f}")
        if m["mae"] < best_mae:
            best_name, best_mae = name, m["mae"]

    r = summary["models"]["rule"]
    print(f"{'rule (baseline)':18s} MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  Pearson={r['pearson']:.3f}")
    summary["best_model"] = best_name

    # Final model is trained on every answer and saved for the app
    final = CANDIDATES[best_name]().fit(X, y)
    LearnedScorer(final, ev.backend, best_name).save(ROOT / "models" / "scorer.joblib")
    if best_name == "ridge":
        coefs = final[-1].coef_
        summary["ridge_coefficients"] = dict(zip(FEATURE_NAMES, map(float, coefs)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "cv_predictions.csv", index=False)
    with open(out / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best model: {best_name}. Saved models/scorer.joblib and {out}/cv_predictions.csv")


if __name__ == "__main__":
    main()
