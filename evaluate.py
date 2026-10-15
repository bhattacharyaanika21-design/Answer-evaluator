"""Evaluate the system against teacher marks and produce charts for the report.

Run after train.py:
    python evaluate.py
Outputs results/metrics.json and results/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluator.metrics import concept_detection_metrics, regression_metrics

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

# Colours (validated categorical slots) and neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "text.color": INK,
})

LABELS = {"rule": "Fixed-weight formula", "ridge": "Ridge regression",
          "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}


def as_set(s) -> set:
    return set(s.split(";")) if isinstance(s, str) and s else set()


def scatter(ax, human, pred, title, color):
    ax.plot([0, 10], [0, 10], color=INK_2, lw=1, ls="--", zorder=1)
    ax.scatter(human, pred, s=46, color=color, edgecolor="white", linewidth=1.5, zorder=2)
    ax.set(xlim=(-0.5, 10.5), ylim=(-0.5, 10.5), xlabel="Teacher score", ylabel="Predicted score")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    m = regression_metrics(human, pred)
    ax.text(0.2, 9.9, f"MAE {m['mae']:.2f}   r {m['pearson']:.2f}", va="top", color=INK_2, fontsize=10)


def main() -> None:
    t = pd.read_csv(RES / "cv_predictions.csv")
    summary = json.loads((RES / "training_summary.json").read_text())
    best = summary["best_model"]

    metrics = {"backend": summary["backend"], "n_answers": len(t), "best_model": best,
               "scoring": {name: regression_metrics(t.human_score, t[f"{name}_score"])
                           for name in ["rule", *[k for k in LABELS if k != "rule"]]
                           if f"{name}_score" in t}}

    gold = [as_set(s) for s in t.gold_concepts]
    det = [as_set(s) for s in t.detected_concepts]
    metrics["concept_detection"] = concept_detection_metrics(gold, det)
    metrics["concept_detection_by_question"] = {
        q: concept_detection_metrics([gold[i] for i in idx], [det[i] for i in idx])
        for q, idx in t.groupby("question_id").indices.items()}

    t["abs_err_best"] = (t[f"{best}_score"] - t.human_score).abs()
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 1. Predicted vs teacher scores
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    scatter(axes[0], t.human_score, t.rule_score, "Fixed-weight formula", ORANGE)
    scatter(axes[1], t.human_score, t[f"{best}_score"], f"Learned scorer ({LABELS[best].lower()})", BLUE)
    axes[1].set_ylabel("")
    fig.savefig(RES / "predicted_vs_teacher.png")
    plt.close(fig)

    # 2. MAE by scoring method (lower is better)
    names = list(metrics["scoring"])
    maes = [metrics["scoring"][n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [BLUE if n == best else "#a9c8ee" for n in names]
    bars = ax.barh([LABELS[n] for n in names], maes, color=colors, height=0.55)
    for b, v in zip(bars, maes):
        ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error (marks out of 10, lower is better)")
    fig.savefig(RES / "mae_by_model.png")
    plt.close(fig)

    # 3. Concept detection recall/precision per question
    qs = sorted(metrics["concept_detection_by_question"])
    prec = [metrics["concept_detection_by_question"][q]["precision"] for q in qs]
    rec = [metrics["concept_detection_by_question"][q]["recall"] for q in qs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(qs))
    ax.bar([i - 0.19 for i in x], prec, width=0.36, color=BLUE, label="Precision")
    ax.bar([i + 0.19 for i in x], rec, width=0.36, color=AQUA, label="Recall")
    ax.set_xticks(list(x), qs)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.set_ylabel("Score")
    fig.savefig(RES / "concept_detection.png")
    plt.close(fig)

    # 4. Worked examples (used by the presentation)
    from evaluator import AnswerEvaluator

    ev = AnswerEvaluator(backend=metrics["backend"])
    answers = pd.read_csv(ROOT / "data" / "student_answers.csv").set_index("answer_id")
    examples = {}
    for aid in ["A04", "A05", "C04"]:
        r = answers.loc[aid]
        res = ev.evaluate_question(r.question_id, r.student_answer)
        examples[aid] = {"question": ev.questions[r.question_id]["question"], "answer": r.student_answer,
                         "teacher_score": float(r.human_score), **res.to_dict(),
                         # out-of-fold score: model never saw this question during training
                         "cv_score": float(t.set_index("answer_id").loc[aid, f"{best}_score"])}
    (RES / "examples.json").write_text(json.dumps(examples, indent=2))

    s = metrics["scoring"][best]
    c = metrics["concept_detection"]
    print(f"Backend: {metrics['backend']}  |  best scorer: {best}")
    print(f"MAE {s['mae']:.2f}  RMSE {s['rmse']:.2f}  Pearson {s['pearson']:.3f}  "
          f"Spearman {s['spearman']:.3f}  QWK {s['qwk']:.3f}  within 1 mark {s['within_1_mark']:.0%}")
    print(f"Concept detection: precision {c['precision']:.2f}  recall {c['recall']:.2f}  F1 {c['f1']:.2f}")
    worst = t.sort_values("abs_err_best", ascending=False).head(5)
    print("\nLargest errors:")
    print(worst[["answer_id", "human_score", f"{best}_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
