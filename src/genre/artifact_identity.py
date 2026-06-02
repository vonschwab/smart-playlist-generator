"""Deterministic identity checks for sparse genre artifacts and dense sidecars."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DENSE_SIDECAR_SCHEMA_VERSION = "dense-genre-sidecar-v2"

_ARTIFACT_KEYS = ("track_ids", "genre_vocab", "X_genre_raw")
_SIDECAR_KEYS = ("track_ids", "genre_vocab", "X_genre_dense", "genre_emb", "emb_config")
_HASH_CHUNK_SIZE = 1024 * 1024


def _missing_key_reason(container: Any, keys: Iterable[str], *, label: str) -> str | None:
    missing = [key for key in keys if key not in container]
    if missing:
        return f"{label} missing required keys: {', '.join(missing)}"
    return None


def _update_text_values(digest: Any, values: np.ndarray) -> None:
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def genre_artifact_identity(
    track_ids: np.ndarray,
    genre_vocab: np.ndarray,
    X_genre_raw: np.ndarray,
) -> str:
    """Hash the sparse genre inputs consumed by dense embedding generation."""
    track_ids = np.asarray(track_ids)
    genre_vocab = np.asarray(genre_vocab)
    X_genre_raw = np.asarray(X_genre_raw)
    if track_ids.ndim != 1:
        raise ValueError("track_ids must be one-dimensional")
    if genre_vocab.ndim != 1:
        raise ValueError("genre_vocab must be one-dimensional")
    if X_genre_raw.ndim != 2:
        raise ValueError("X_genre_raw must be two-dimensional")
    if X_genre_raw.shape != (track_ids.size, genre_vocab.size):
        raise ValueError("X_genre_raw shape must align with track_ids and genre_vocab")

    digest = hashlib.sha256()
    digest.update(b"dense-genre-sparse-identity-v1\0")
    _update_text_values(digest, track_ids)
    _update_text_values(digest, genre_vocab)
    contiguous = np.ascontiguousarray(X_genre_raw)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    raw = memoryview(contiguous).cast("B")
    for start in range(0, raw.nbytes, _HASH_CHUNK_SIZE):
        digest.update(raw[start:start + _HASH_CHUNK_SIZE])
    return digest.hexdigest()


def dense_sidecar_mismatch_reason(*, artifact: Any, sidecar: Any) -> str | None:
    """Return why a dense sidecar cannot be used with a sparse artifact."""
    reason = _missing_key_reason(artifact, _ARTIFACT_KEYS, label="artifact")
    if reason is not None:
        return reason
    reason = _missing_key_reason(sidecar, _SIDECAR_KEYS, label="sidecar")
    if reason is not None:
        return reason
    if not np.array_equal(sidecar["track_ids"], artifact["track_ids"]):
        return "track_ids mismatch"
    if not np.array_equal(sidecar["genre_vocab"], artifact["genre_vocab"]):
        return "vocabulary mismatch"
    try:
        config = sidecar["emb_config"].item()
    except Exception:
        return "emb_config metadata is invalid"
    if not isinstance(config, dict):
        return "emb_config metadata is invalid"
    if config.get("schema_version") != DENSE_SIDECAR_SCHEMA_VERSION:
        return "schema version mismatch"
    try:
        current = genre_artifact_identity(
            artifact["track_ids"],
            artifact["genre_vocab"],
            artifact["X_genre_raw"],
        )
    except (TypeError, ValueError) as exc:
        return f"artifact sparse genre inputs are invalid: {exc}"
    if config.get("sparse_genre_identity") != current:
        return "sparse genre identity mismatch"
    return None


def dense_sidecar_mismatch_reason_from_paths(
    *,
    artifact_path: str | Path,
    sidecar_path: str | Path,
) -> str | None:
    """Load artifact paths and return their dense-sidecar mismatch reason."""
    with np.load(Path(artifact_path), allow_pickle=True) as artifact:
        with np.load(Path(sidecar_path), allow_pickle=True) as sidecar:
            return dense_sidecar_mismatch_reason(artifact=artifact, sidecar=sidecar)
