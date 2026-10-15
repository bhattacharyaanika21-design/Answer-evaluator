"""Streamlit front end for the Automated Answer Evaluator.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluator import AnswerEvaluator

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI Answer Evaluator", page_icon="📝", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading language model...")
def get_evaluator(backend: str, scoring: str) -> AnswerEvaluator:
    return AnswerEvaluator(backend=backend, scoring=scoring)


@st.cache_resource(show_spinner="Loading handwriting model (first time only)...")
def get_ocr():
    from evaluator.ocr import HandwritingOCR

    return HandwritingOCR()


def concepts_to_text(concepts: list[dict]) -> str:
    return "\n".join(f"{c['name']} | {', '.join(c.get('keywords', []))} | {c.get('weight', 1)}"
                     for c in concepts)


def text_to_concepts(text: str) -> list[dict]:
    """Each line: Concept name | keyword 1, keyword 2 | weight   (keywords and weight optional)."""
    concepts = []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        parts = [p.strip() for p in line.split("|")]
        weight = 1.0
        if len(parts) > 2:
            try:
                weight = float(parts[2])
            except ValueError:
                pass
        keywords = [k.strip() for k in parts[1].split(",") if k.strip()] if len(parts) > 1 else []
        concepts.append({"id": f"c{i + 1}", "name": parts[0], "keywords": keywords, "weight": weight})
    return concepts


def show_result(res) -> None:
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"<div style='font-size:3.2rem;font-weight:700;line-height:1'>{res.score:g}"
                    f"<span style='font-size:1.4rem;color:gray'> / {res.max_marks:g}</span></div>",
                    unsafe_allow_html=True)
        st.caption(f"Scoring: {res.scoring_method} · model: {res.backend}")
    with right:
        for label, value in [("Semantic similarity", res.semantic_similarity),
                             ("Concept coverage", res.concept_coverage),
                             ("Completeness", res.completeness)]:
            st.write(f"**{label}** — {value:.0%}")
            st.progress(min(max(value, 0.0), 1.0))

    st.subheader("Concepts")
    rows = [{"": "✅" if c["found"] else "❌", "Concept": c["name"], "Weight": c["weight"],
             "Detected by": c["method"] or "—", "Evidence": c["evidence"] or ""} for c in res.concepts]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No rubric concepts defined, so concept coverage falls back to semantic similarity.")

    st.subheader("Feedback")
    st.success(res.feedback) if res.percentage >= 65 else st.warning(res.feedback)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Language model", ["auto", "sbert", "tfidf"],
                           help="sbert = Sentence-BERT transformer; tfidf = lexical baseline; "
                                "auto = Sentence-BERT if available")
    scoring = st.selectbox("Scoring engine", ["auto", "learned", "rule"],
                           help="learned = model trained on teacher marks (run train.py); "
                                "rule = 50% similarity + 30% concepts + 20% completeness")
    try:
        ev = get_evaluator(backend, scoring)
        st.success(f"Model: **{ev.backend}** · scoring: **{ev.scoring}**")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

st.title("📝 AI Answer Evaluation System")
st.caption("Scores descriptive answers against a reference answer and marking rubric, "
           "using transformer embeddings and concept coverage.")

tab_eval, tab_batch, tab_perf = st.tabs(["Evaluate an answer", "Batch evaluation", "Model performance"])

# ---------------------------------------------------------------- single answer
with tab_eval:
    options = {f"{q['id']} · {q['question']}": qid for qid, q in ev.questions.items()}
    choice = st.selectbox("Question", ["✏️ Custom question"] + list(options))
    qid = options.get(choice, "custom")
    q = ev.questions.get(qid, {})

    question = st.text_input("Question text", q.get("question", ""), key=f"q_{qid}")
    c1, c2 = st.columns(2)
    with c1:
        reference = st.text_area("Reference answer", q.get("reference_answer", ""), height=200,
                                 key=f"ref_{qid}")
        max_marks = st.number_input("Maximum marks", 1.0, 100.0, float(q.get("max_marks", 10)), 1.0,
                                    key=f"max_{qid}")
    with c2:
        rubric = st.text_area("Rubric concepts (name | keywords | weight)",
                              concepts_to_text(q.get("concepts", [])), height=200, key=f"rubric_{qid}",
                              help="One concept per line, e.g.  Circular wait | circular wait, cycle | 1")

    mode = st.radio("Student answer", ["Type answer", "Upload handwritten answer"], horizontal=True)
    student = ""
    if mode == "Type answer":
        student = st.text_area("Student answer", height=150, placeholder="Type or paste the answer...")
    else:
        img_file = st.file_uploader("Photo or scan of the answer", type=["png", "jpg", "jpeg"])
        if img_file:
            from PIL import Image

            image = Image.open(img_file)
            st.image(image, caption="Uploaded answer", width="stretch")
            try:
                text, lines = get_ocr().transcribe(image)
                student = st.text_area("Recognised text (correct any OCR mistakes)", text, height=150)
            except ImportError:
                st.error("Handwriting recognition needs torch and transformers: "
                         "pip install -r requirements-ocr.txt")

    if st.button("Evaluate", type="primary", disabled=not (reference.strip() and student.strip())):
        res = ev.evaluate(reference, student, text_to_concepts(rubric), max_marks)
        st.divider()
        show_result(res)
        st.download_button("Download result (JSON)", json.dumps(res.to_dict(), indent=2),
                           "evaluation.json", "application/json")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with columns **question_id** and **student_answer** "
             "(question IDs from the question bank). An optional **human_score** column "
             "lets you compare the system with the teacher's marks.")
    up = st.file_uploader("Answers CSV", type="csv", key="batch")
    if up:
        df = pd.read_csv(up)
        missing = {"question_id", "student_answer"} - set(df.columns)
        unknown = set(df.get("question_id", pd.Series(dtype=str))) - set(ev.questions)
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif unknown:
            st.error(f"Unknown question IDs: {', '.join(map(str, unknown))}")
        else:
            out = []
            bar = st.progress(0.0)
            for i, r in enumerate(df.itertuples()):
                res = ev.evaluate_question(r.question_id, str(r.student_answer))
                out.append({"predicted_score": res.score,
                            "similarity": round(res.semantic_similarity, 3),
                            "concept_coverage": round(res.concept_coverage, 3),
                            "missing_concepts": "; ".join(c["name"] for c in res.concepts if not c["found"]),
                            "feedback": res.feedback})
                bar.progress((i + 1) / len(df))
            result = pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)
            if "human_score" in result:
                from evaluator.metrics import regression_metrics

                m = regression_metrics(result.human_score, result.predicted_score)
                cols = st.columns(4)
                for col, (k, label) in zip(cols, [("mae", "MAE"), ("rmse", "RMSE"),
                                                  ("pearson", "Pearson r"), ("qwk", "QWK")]):
                    col.metric(label, f"{m[k]:.2f}")
            st.dataframe(result, width="stretch")
            buf = io.StringIO()
            result.to_csv(buf, index=False)
            st.download_button("Download results CSV", buf.getvalue(), "evaluated_answers.csv", "text/csv")

# ---------------------------------------------------------------- performance
with tab_perf:
    mpath = ROOT / "results" / "metrics.json"
    if not mpath.exists():
        st.info("Run `python train.py` and then `python evaluate.py` to generate results.")
    else:
        m = json.loads(mpath.read_text())
        best = m["scoring"][m["best_model"]]
        st.write(f"Leave-one-question-out cross-validation on {m['n_answers']} teacher-marked answers "
                 f"· language model: **{m['backend']}** · scorer: **{m['best_model']}**")
        cols = st.columns(5)
        for col, (k, label, fmt) in zip(cols, [("mae", "MAE", "{:.2f}"), ("rmse", "RMSE", "{:.2f}"),
                                                ("pearson", "Pearson r", "{:.2f}"),
                                                ("spearman", "Spearman ρ", "{:.2f}"),
                                                ("within_1_mark", "Within 1 mark", "{:.0%}")]):
            col.metric(label, fmt.format(best[k]))
        for img in ["predicted_vs_teacher.png", "mae_by_model.png", "concept_detection.png"]:
            if (ROOT / "results" / img).exists():
                st.image(str(ROOT / "results" / img), width="stretch")
