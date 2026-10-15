"""The AnswerEvaluator: compares a student answer with a reference answer and rubric."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .embedder import DEFAULT_SBERT_MODEL, Embedder
from .feedback import generate_feedback
from .scorer import LearnedScorer, RuleScorer
from .text_utils import contains_phrase, split_sentences, stem_tokens, word_count

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "questions.json"
DEFAULT_MODEL = ROOT / "models" / "scorer.joblib"


@dataclass
class Concept:
    id: str
    name: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class EvaluationResult:
    score: float
    max_marks: float
    semantic_similarity: float
    concept_coverage: float
    completeness: float
    concepts: list[dict]
    feedback: str
    features: dict
    scoring_method: str
    backend: str

    @property
    def percentage(self) -> float:
        return 100 * self.score / self.max_marks if self.max_marks else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["percentage"] = self.percentage
        return d


def load_question_bank(path: str | Path = DEFAULT_QUESTIONS) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}


class AnswerEvaluator:
    """Scores descriptive answers.

    Parameters
    ----------
    backend: "auto" | "sbert" | "tfidf"
    scoring: "auto" (learned model if one exists for this backend, else rule),
             "rule" or "learned".
    """

    def __init__(self, backend: str = "auto", scoring: str = "auto",
                 model_path: str | Path = DEFAULT_MODEL, sbert_model: str = DEFAULT_SBERT_MODEL,
                 weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
                 question_bank: str | Path = DEFAULT_QUESTIONS):
        self.questions = load_question_bank(question_bank) if Path(question_bank).exists() else {}
        corpus = [q["reference_answer"] for q in self.questions.values()]
        self.embedder = Embedder(backend, sbert_model, background_corpus=corpus)
        self.rule_scorer = RuleScorer(*weights)
        self.learned_scorer = None
        if scoring in {"auto", "learned"} and Path(model_path).exists():
            candidate = LearnedScorer.load(model_path)
            if candidate.backend == self.embedder.backend:
                self.learned_scorer = candidate
        if scoring == "learned" and self.learned_scorer is None:
            raise FileNotFoundError(
                f"No learned model for backend '{self.embedder.backend}' at {model_path}. Run train.py first.")
        self.scoring = "rule" if scoring == "rule" else ("learned" if self.learned_scorer else "rule")

    @property
    def backend(self) -> str:
        return self.embedder.backend

    # ------------------------------------------------------------------ #
    def evaluate_question(self, question_id: str, student_answer: str) -> EvaluationResult:
        q = self.questions[question_id]
        return self.evaluate(q["reference_answer"], student_answer, q.get("concepts", []),
                             q.get("max_marks", 10))

    def evaluate(self, reference_answer: str, student_answer: str,
                 concepts: list[dict | Concept] | None = None, max_marks: float = 10) -> EvaluationResult:
        concepts = [c if isinstance(c, Concept) else Concept(**{k: c[k] for k in c if k in Concept.__dataclass_fields__})
                    for c in (concepts or [])]
        features, concept_results = self.extract_features(reference_answer, student_answer, concepts)
        scorer = self.learned_scorer if self.scoring == "learned" else self.rule_scorer
        score = 0.0 if features["word_count"] == 0 else scorer.predict(features, max_marks)
        score = round(score * 2) / 2 if max_marks >= 5 else round(score, 1)  # half-mark steps

        feedback = generate_feedback(score, max_marks, features["semantic_similarity"],
                                     features["concept_coverage"], features["completeness"],
                                     concept_results, features["word_count"],
                                     features["reference_word_count"])
        return EvaluationResult(
            score=score, max_marks=max_marks,
            semantic_similarity=features["semantic_similarity"],
            concept_coverage=features["concept_coverage"],
            completeness=features["completeness"],
            concepts=concept_results, feedback=feedback, features=features,
            scoring_method=scorer.name, backend=self.backend)

    # ------------------------------------------------------------------ #
    def extract_features(self, reference: str, student: str, concepts: list[Concept]):
        prof = self.embedder.profile
        ref_sents = split_sentences(reference)
        stu_sents = split_sentences(student) if student.strip() else [""]
        concept_texts = [c.name for c in concepts]

        # One encoding call for everything keeps the TF-IDF vocabulary consistent
        texts = [reference, student] + ref_sents + stu_sents + concept_texts
        emb = self.embedder.encode(texts)
        i = 2
        ref_vec, stu_vec = emb[0], emb[1]
        ref_sent_vecs = emb[i:i + len(ref_sents)]; i += len(ref_sents)
        stu_sent_vecs = emb[i:i + len(stu_sents)]; i += len(stu_sents)
        concept_vecs = emb[i:]

        raw_cos = float(np.dot(ref_vec, stu_vec)) if student.strip() else 0.0
        similarity = self.embedder.scale(raw_cos)

        # --- concept detection: keyword match, then semantic match --------
        stems = stem_tokens(student)
        sent_sims = concept_vecs @ stu_sent_vecs.T if len(concepts) else np.zeros((0, 0))
        results, total_w, found_w, kw_only_w = [], 0.0, 0.0, 0.0
        for k, c in enumerate(concepts):
            total_w += c.weight
            hit = next((kw for kw in [c.name] + c.keywords if contains_phrase(stems, kw)), None)
            method, evidence, conf = None, None, 0.0
            if hit:
                method, evidence, conf = "keyword", hit, 1.0
            elif prof["concept_threshold"] is not None and student.strip():
                j = int(np.argmax(sent_sims[k]))
                conf = float(sent_sims[k, j])
                if conf >= prof["concept_threshold"]:
                    method, evidence = "semantic", stu_sents[j]
            found = method is not None
            if found:
                found_w += c.weight
                if method == "keyword":
                    kw_only_w += c.weight
            results.append({"id": c.id, "name": c.name, "weight": c.weight, "found": found,
                            "method": method, "evidence": evidence, "confidence": round(conf, 3)})
        coverage = found_w / total_w if total_w else similarity

        # --- completeness: how much of the reference is covered -----------
        if student.strip():
            per_ref = (ref_sent_vecs @ stu_sent_vecs.T).max(axis=1)
            sent_cov = float(np.mean([self.embedder.scale(s) for s in per_ref]))
        else:
            sent_cov = 0.0
        n_ref, n_stu = word_count(reference), word_count(student)
        length_ratio = min(n_stu / n_ref, 1.5) if n_ref else 0.0
        completeness = 0.7 * sent_cov + 0.3 * min(n_stu / (0.6 * n_ref), 1.0) if n_ref else 0.0

        features = {
            "semantic_similarity": similarity,
            "concept_coverage": coverage,
            "completeness": float(completeness),
            "length_ratio": float(length_ratio),
            "raw_cosine": raw_cos,
            "keyword_only_share": kw_only_w / total_w if total_w else 0.0,
            "word_count": n_stu,
            "reference_word_count": n_ref,
        }
        return features, results
