"""
PMI-SVD genre embedding — Phase 2 of the genre subsystem redesign.

Computes a dense per-genre embedding from the weighted co-occurrence statistics
in the track library, using Positive PMI + truncated SVD (the standard
distributional semantics approach).

Public API:
    build_genre_matrix(db_path, *, enrichment_db_path, normalize) -> (X, vocab, support)
    train_pmi_svd(X, dim, smoothing, random_state) -> (V, dim) embedding

All functions are pure (no I/O) except build_genre_matrix.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)

# Tiered source weights (raw genres, matching existing artifact builder convention)
WEIGHT_TRACK = 1.0
WEIGHT_ALBUM = 0.8
WEIGHT_ARTIST = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_genre_matrix(
    db_path: str | Path,
    *,
    enrichment_db_path: str | Path | None = None,
    normalize_genres: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Load all tracks from metadata.db and return a weighted genre matrix.

    Returns:
        X:       (N, V) float32 matrix — X[t, g] = max weight for genre g in track t
        vocab:   V genre strings (sorted)
        support: (V,) int — number of tracks where each genre is present (weight > 0)

    Weights:
      - track_genres:  WEIGHT_TRACK
      - album_genres:  WEIGHT_ALBUM
      - artist_genres: WEIGHT_ARTIST
      - enriched (from sidecar): confidence value from enriched_genres table (replaces raw
        weights when a release has enrichment data)
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        track_ids, raw_genres = _load_raw_genres(conn, normalize_genres=normalize_genres)
    finally:
        conn.close()

    enriched_map: dict[str, list[tuple[str, float]]] = {}
    if enrichment_db_path:
        enrichment_db_path = Path(enrichment_db_path)
        if enrichment_db_path.exists():
            enriched_map = _load_enriched_genres(enrichment_db_path, db_path)

    # Merge enriched over raw (replace per-track when enrichment available)
    merged: list[list[tuple[str, float]]] = []
    for tid, raw in zip(track_ids, raw_genres):
        if tid in enriched_map:
            merged.append(enriched_map[tid])
        else:
            merged.append(raw)

    # Build vocab from all active tokens
    all_tokens: set[str] = set()
    for genres in merged:
        all_tokens.update(g for g, _ in genres if g)
    vocab = sorted(all_tokens)
    vocab_index = {g: i for i, g in enumerate(vocab)}

    N = len(track_ids)
    V = len(vocab)
    X = np.zeros((N, V), dtype=np.float32)
    for t, genres in enumerate(merged):
        for g, w in genres:
            j = vocab_index.get(g)
            if j is not None and w > X[t, j]:
                X[t, j] = float(w)

    support = (X > 0).sum(axis=0).astype(np.int32)
    logger.info("build_genre_matrix: N=%d tracks, V=%d genres", N, V)
    return X, vocab, support


def _load_raw_genres(
    conn: sqlite3.Connection,
    *,
    normalize_genres: bool,
) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """
    Return (track_ids, genre_lists) from metadata.db.

    genre_lists[i] = [(normalized_genre, weight), ...] for track_ids[i].
    """
    try:
        from src.genre.normalize_unified import normalize_genre_token as _norm
    except ImportError:
        _norm = lambda x: x.strip().lower()  # noqa: E731

    def _norm_token(raw: str) -> str | None:
        if not raw or not raw.strip():
            return None
        return (_norm(raw) or raw.strip().lower()) if normalize_genres else raw.strip().lower()

    # Load all track IDs
    track_ids = [str(r["track_id"]) for r in conn.execute("SELECT track_id FROM tracks ORDER BY track_id")]
    album_id_for_track: dict[str, str | None] = {
        str(r["track_id"]): r["album_id"] for r in conn.execute("SELECT track_id, album_id FROM tracks")
    }
    artist_for_track: dict[str, str] = {
        str(r["track_id"]): str(r["artist"] or "") for r in conn.execute("SELECT track_id, artist FROM tracks")
    }

    # Build lookup tables: track_genres, album_genres, artist_genres
    track_genre_map: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT track_id, genre, weight FROM track_genres"):
        tid = str(r["track_id"])
        raw = r["genre"]
        w = float(r["weight"] or WEIGHT_TRACK)
        tok = _norm_token(raw)
        if tok:
            track_genre_map.setdefault(tid, {})
            if w > track_genre_map[tid].get(tok, 0):
                track_genre_map[tid][tok] = w

    album_genre_map: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT album_id, genre FROM album_genres"):
        aid = str(r["album_id"])
        tok = _norm_token(r["genre"])
        if tok:
            album_genre_map.setdefault(aid, {})[tok] = WEIGHT_ALBUM

    artist_genre_map: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT artist, genre FROM artist_genres"):
        artist = str(r["artist"])
        tok = _norm_token(r["genre"])
        if tok:
            artist_genre_map.setdefault(artist, {})[tok] = WEIGHT_ARTIST

    genre_lists: list[list[tuple[str, float]]] = []
    for tid in track_ids:
        merged: dict[str, float] = {}
        # track layer
        for g, w in track_genre_map.get(tid, {}).items():
            merged[g] = max(merged.get(g, 0), w)
        # album layer
        aid = album_id_for_track.get(tid)
        if aid:
            for g, w in album_genre_map.get(aid, {}).items():
                merged[g] = max(merged.get(g, 0), w)
        # artist layer
        artist = artist_for_track.get(tid, "")
        for g, w in artist_genre_map.get(artist, {}).items():
            merged[g] = max(merged.get(g, 0), w)
        genre_lists.append(list(merged.items()))

    return track_ids, genre_lists


def _load_enriched_genres(
    enrichment_db: Path,
    metadata_db: Path,
) -> dict[str, list[tuple[str, float]]]:
    """
    Return {track_id: [(genre, confidence), ...]} for tracks whose album has enriched data.
    Confidence comes from enriched_genres.confidence.
    """
    from src.ai_genre_enrichment.normalization import make_release_key, normalize_release_artist, normalize_release_name

    # Load release keys from enrichment DB
    econn = sqlite3.connect(f"file:{enrichment_db.resolve().as_posix()}?mode=ro", uri=True)
    econn.row_factory = sqlite3.Row
    release_genres: dict[str, list[tuple[str, float]]] = {}
    try:
        for r in econn.execute(
            "SELECT release_key, genre, confidence FROM enriched_genres WHERE status IS NULL OR status != 'rejected'"
        ):
            rk = str(r["release_key"])
            release_genres.setdefault(rk, []).append((str(r["genre"]), float(r["confidence"] or 1.0)))
    finally:
        econn.close()

    if not release_genres:
        return {}

    # Map (artist, album) → release_key for tracks in metadata DB
    mconn = sqlite3.connect(f"file:{metadata_db.resolve().as_posix()}?mode=ro", uri=True)
    mconn.row_factory = sqlite3.Row
    result: dict[str, list[tuple[str, float]]] = {}
    try:
        for r in mconn.execute("SELECT track_id, artist, album FROM tracks"):
            artist = normalize_release_artist(str(r["artist"] or ""))
            album = normalize_release_name(str(r["album"] or ""))
            rk = make_release_key(artist, album)
            if rk in release_genres:
                result[str(r["track_id"])] = release_genres[rk]
    finally:
        mconn.close()

    logger.info("Enriched genres: %d tracks have enrichment data", len(result))
    return result


# ---------------------------------------------------------------------------
# PMI-SVD core
# ---------------------------------------------------------------------------

def _remove_top_components(embedding: np.ndarray, k: int) -> np.ndarray:
    """All-but-the-top (Mu & Viswanath, 2018): mean-center and project out the
    top-``k`` principal components.

    PPMI is non-negative, so by Perron-Frobenius its dominant singular direction
    is ~all-positive: every genre vector inherits a large shared component that
    inflates all pairwise cosines and collapses the usable dynamic range. Removing
    the mean + top-k PCs restores it without disturbing the relative structure.
    """
    if k <= 0:
        return embedding
    mu = embedding.mean(axis=0, keepdims=True)
    centered = embedding - mu
    # Principal directions of the centered embedding (Vt rows are unit PCs).
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    k = min(k, Vt.shape[0])
    top = Vt[:k]                               # (k, dim)
    centered = centered - (centered @ top.T) @ top
    return centered


def train_pmi_svd(
    X: np.ndarray,
    *,
    dim: int = 64,
    smoothing: float = 1.0,
    random_state: int = 42,
    remove_top_components: int = 0,
) -> np.ndarray:
    """
    Train a genre embedding from a weighted genre co-occurrence matrix.

    Args:
        X:           (N, V) float32 track-genre weight matrix (0 = absent)
        dim:         embedding dimension
        smoothing:   additive smoothing α on the co-occurrence counts
        random_state: random seed for randomized_svd
        remove_top_components: if > 0, apply all-but-the-top post-processing
            (mean-center + drop this many leading PCs) before L2-normalization.
            Defaults to 0 (legacy behavior). Production builds use 2 to remove
            the PPMI shared-direction anisotropy. See ``_remove_top_components``.

    Returns:
        embedding:   (V, dim) float32, rows L2-normalized
    """
    X = np.asarray(X, dtype=np.float64)
    N, V = X.shape
    if V < dim:
        raise ValueError(f"vocab size {V} < embedding dim {dim}; reduce dim")

    # Weighted co-occurrence: cooc[i,j] = Σ_t X[t,i] * X[t,j]
    cooc = X.T @ X  # (V, V)

    # Additive smoothing PMI
    total = cooc.sum()
    denom = total + smoothing * (V * V)
    P = (cooc + smoothing) / denom          # joint probability (V, V)
    p_marginal = P.sum(axis=1)              # (V,) row marginals

    # PMI = log(P_ij / (p_i * p_j))
    # Use outer product of marginals for the denominator
    outer = np.outer(p_marginal, p_marginal)  # (V, V)
    # Avoid log(0): where P == 0 (after smoothing, never happens) or outer == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(P / np.maximum(outer, 1e-300))

    ppmi = np.maximum(pmi, 0.0)            # positive PMI

    # Truncated SVD
    U, S, _Vt = randomized_svd(ppmi, n_components=dim, random_state=random_state)
    # Absorb singular values into the row vectors (GloVe-style)
    embedding = U * np.sqrt(S)             # (V, dim)

    # All-but-the-top: strip the dominant shared direction (anisotropy) so the
    # downstream cosine has real dynamic range.
    embedding = _remove_top_components(embedding, remove_top_components)

    # L2-normalize rows
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-12)

    return embedding.astype(np.float32)


def project_tracks(
    X_genre_raw: np.ndarray,
    genre_emb: np.ndarray,
    *,
    idf: np.ndarray | None = None,
) -> np.ndarray:
    """Project an (N, V) sparse genre matrix into the (V, dim) dense space.

    Each track vector is the (optionally IDF-weighted) weighted average of its
    genre embeddings, then L2-normalized. Tracks with no genres get a zero vector.

    Args:
        X_genre_raw: (N, V) track-genre weight matrix.
        genre_emb:   (V, dim) genre embedding.
        idf:         optional (V,) per-genre weights. When given, genre columns
            are scaled by ``idf`` before projection so hub genres (e.g. "rock",
            present in ~half the library) do not dominate a track's position.

    Returns:
        (N, dim) float32, L2-normalized rows (zero rows for genre-less tracks).
    """
    X = X_genre_raw
    if idf is not None:
        X = X_genre_raw * np.asarray(idf, dtype=X_genre_raw.dtype)[None, :]
    projected = X @ genre_emb  # (N, dim)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    # Zero out rows whose ORIGINAL (unweighted) genre vector was all-zero.
    has_genre = (X_genre_raw > 0).any(axis=1)
    projected = projected / norms
    projected[~has_genre] = 0.0
    return projected.astype(np.float32)
