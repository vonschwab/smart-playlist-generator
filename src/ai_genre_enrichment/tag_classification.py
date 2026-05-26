"""Deterministic first-pass classification for source-provided release tags."""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .genre_vocabulary import GenreVocabulary

_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "data" / "genre_vocabulary.yaml"

_vocab: GenreVocabulary | None = None
_vocab_lock = threading.Lock()


def _get_vocabulary() -> GenreVocabulary:
    global _vocab
    if _vocab is None:
        with _vocab_lock:
            if _vocab is None:
                _vocab = GenreVocabulary(_DEFAULT_YAML_PATH)
    return _vocab


def set_vocabulary(vocab: GenreVocabulary) -> None:
    """Override the module-level vocabulary (for testing or library-tier enrichment)."""
    global _vocab
    _vocab = vocab


def reset_vocabulary() -> None:
    """Reset the module-level vocabulary to force reload from the YAML file."""
    global _vocab
    _vocab = None


_LABEL_OR_ORG_REASON = "Artist, label, or related-entity tag, not a genre."
_CATEGORY_REASONS = {
    "descriptor": "Descriptor tag, not a genre.",
    "instrument": "Instrument tag, not a genre.",
    "place": "Place tag, not a genre.",
    "format": "Release format tag, not a genre.",
    "mood_function": "Mood or listening-function tag, not a genre.",
    "label_or_org": _LABEL_OR_ORG_REASON,
}
_CATEGORY_CONFIDENCES = {
    "mood_function": 0.9,
}


@dataclass(frozen=True)
class SourceTagClassification:
    raw_tag: str
    normalized_tag: str
    classification: str
    confidence: float
    reason: str


def classify_source_tag(raw_tag: str) -> SourceTagClassification:
    """Classify a source tag without collapsing precise source vocabulary."""
    vocab = _get_vocabulary()
    normalized = normalize_source_tag(raw_tag)

    genre_result = vocab.classify_genre(normalized)
    if genre_result is not None:
        return SourceTagClassification(
            raw_tag=raw_tag,
            normalized_tag=genre_result.genre,  # use the resolved form from vocab
            classification="genre_style",
            confidence=genre_result.confidence,
            reason=genre_result.reason,
        )

    non_genre = vocab.classify_non_genre(normalized)
    if non_genre is not None:
        resolved = vocab.resolve_alias(normalized)
        reason = _CATEGORY_REASONS.get(non_genre, f"{non_genre} tag, not a genre.")
        confidence = _CATEGORY_CONFIDENCES.get(non_genre, 0.95)
        # label_or_org maps to "descriptor" externally: downstream consumers only see
        # the six output categories (genre_style, descriptor, instrument, place, format, mood_function).
        classification = "descriptor" if non_genre == "label_or_org" else non_genre
        return SourceTagClassification(raw_tag, resolved, classification, confidence, reason)

    return SourceTagClassification(
        raw_tag=raw_tag,
        normalized_tag=vocab.resolve_alias(normalized),
        classification="review_only",
        confidence=0.5,
        reason="Unknown source tag requires adjudication before use.",
    )


def normalize_source_tag(raw_tag: str) -> str:
    """Normalize source tags while preserving source-specific genre distinctions."""
    text = unicodedata.normalize("NFKD", raw_tag.strip().casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()
