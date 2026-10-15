"""Text preprocessing helpers: normalisation, sentence splitting and stemming."""
from __future__ import annotations

import re
from functools import lru_cache

try:  # NLTK's Porter stemmer needs no downloaded data
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    def _stem(token: str) -> str:
        return _stemmer.stem(token)

except ImportError:  # pragma: no cover - minimal fallback

    def _stem(token: str) -> str:
        for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "ly", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_NON_WORD = re.compile(r"[^a-z0-9\s]")


def normalize(text: str) -> str:
    """Lower-case, turn hyphens/slashes into spaces and drop punctuation."""
    text = (text or "").lower()
    text = re.sub(r"[-/_]", " ", text)
    text = text.replace("'s", "")
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


@lru_cache(maxsize=50_000)
def stem(token: str) -> str:
    return _stem(token)


def stem_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text)]


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences / clauses. Always returns at least one item."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    parts = [p for p in parts if len(tokenize(p)) >= 2] or [text.strip() or ""]
    return parts


def word_count(text: str) -> int:
    return len(tokenize(text))


def contains_phrase(text_stems: list[str], phrase: str) -> bool:
    """True if the stemmed phrase appears as a contiguous token run in the text."""
    phrase_stems = stem_tokens(phrase)
    n = len(phrase_stems)
    if n == 0:
        return False
    for i in range(len(text_stems) - n + 1):
        if text_stems[i : i + n] == phrase_stems:
            return True
    return False
