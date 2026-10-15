"""Run with:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import AnswerEvaluator  # noqa: E402
from evaluator.text_utils import contains_phrase, split_sentences, stem_tokens  # noqa: E402

EV = AnswerEvaluator(backend="tfidf", scoring="rule")


def test_phrase_matching_handles_inflections_and_hyphens():
    stems = stem_tokens("TCP is connection-oriented and processes are waiting for resources")
    assert contains_phrase(stems, "connection oriented")
    assert contains_phrase(stems, "waiting for resource")
    assert not contains_phrase(stems, "circular wait")


def test_sentence_split():
    assert len(split_sentences("First point here. Second point here; third point.")) == 3


def test_good_answer_beats_partial_and_off_topic():
    good = EV.evaluate_question("Q1", (
        "Deadlock is when processes wait forever for resources held by each other. It needs mutual "
        "exclusion, hold and wait, no preemption and circular wait, and is handled by prevention, "
        "avoidance or detection."))
    partial = EV.evaluate_question("Q1", "Deadlock occurs when processes wait for resources held by other processes.")
    off = EV.evaluate_question("Q1", "Paging divides memory into fixed size pages.")
    assert good.score > partial.score > off.score
    assert good.concept_coverage == 1.0


def test_empty_answer_scores_zero():
    assert EV.evaluate_question("Q2", "").score == 0


def test_custom_question_without_rubric():
    res = EV.evaluate("Photosynthesis converts light energy into chemical energy in plants.",
                      "Plants use photosynthesis to turn light energy into chemical energy.", None, 5)
    assert 0 <= res.score <= 5 and res.concepts == []
