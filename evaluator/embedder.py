"""Text embedding backends.

* ``sbert``  - pretrained Sentence-BERT transformer (the deep-learning model).
* ``tfidf``  - lexical TF-IDF vectors, a lightweight fallback/baseline that needs
               no model download. Useful for testing and for ablation studies.
* ``auto``   - try Sentence-BERT, fall back to TF-IDF if it cannot be loaded.

Every backend returns L2-normalised vectors, so a dot product is the cosine
similarity.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from .text_utils import stem_tokens

log = logging.getLogger(__name__)

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Calibration of each backend. Raw cosine similarities live on different scales
# for different models, so they are rescaled to 0..1 with a floor and ceiling.
# concept_threshold: min cosine between a concept and a student sentence for a
#   semantic (non-keyword) concept match. None disables semantic matching.
# sentence_threshold: min cosine for a reference sentence to count as "covered".
PROFILES = {
    "sbert": {"sim_floor": 0.25, "sim_ceil": 0.90, "concept_threshold": 0.55, "sentence_threshold": 0.60},
    "tfidf": {"sim_floor": 0.00, "sim_ceil": 0.45, "concept_threshold": None, "sentence_threshold": 0.25},
}


class Embedder:
    def __init__(self, backend: str = "auto", model_name: str = DEFAULT_SBERT_MODEL,
                 background_corpus: Iterable[str] | None = None):
        self.model_name = model_name
        self.background_corpus = list(background_corpus or [])
        self._model = None
        if backend not in {"auto", "sbert", "tfidf"}:
            raise ValueError(f"Unknown backend: {backend}")
        if backend in {"auto", "sbert"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                backend = "sbert"
            except Exception as exc:  # ImportError, network errors, ...
                if backend == "sbert":
                    raise
                log.warning("Sentence-BERT unavailable (%s); using TF-IDF fallback.", exc)
                backend = "tfidf"
        self.backend = backend
        self.profile = PROFILES[backend]

    # ------------------------------------------------------------------ #
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalised vectors (one row per text)."""
        if not texts:
            return np.zeros((0, 1))
        if self.backend == "sbert":
            return np.asarray(self._model.encode(texts, normalize_embeddings=True,
                                                 show_progress_bar=False))
        return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize as l2

        from sklearn.feature_extraction.text import TfidfVectorizer

        fit_corpus = self.background_corpus + texts
        word = TfidfVectorizer(analyzer=lambda t: _word_ngrams(t), sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                               lowercase=True)
        word.fit(fit_corpus)
        char.fit(fit_corpus)
        mat = hstack([l2(word.transform(texts)), 0.5 * l2(char.transform(texts))])
        return l2(mat).toarray()

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        return float(np.dot(va, vb))

    def scale(self, cosine: float) -> float:
        """Map a raw cosine to 0..1 using the backend's calibration."""
        lo, hi = self.profile["sim_floor"], self.profile["sim_ceil"]
        return float(np.clip((cosine - lo) / (hi - lo), 0.0, 1.0))


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "and",
    "or", "it", "its", "that", "this", "for", "by", "with", "as", "at", "from", "which", "when",
    "so", "such", "can", "has", "have", "their", "they", "each", "into", "than", "then", "also",
}


def _word_ngrams(text: str) -> list[str]:
    toks = [t for t in stem_tokens(text) if t not in _STOP]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
