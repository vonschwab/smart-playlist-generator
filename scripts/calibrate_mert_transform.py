"""Calibrate post-processing transforms for MERT-v1-95M embeddings.

Phase 2 of the MERT sonic embedding plan
(docs/superpowers/plans/2026-06-11-mert-sonic-embedding.md).

Given a merged MERT sidecar NPZ (from Phase 1), this script:
  1. Builds a stratified ~2k evaluation subset (all diagnostic seeds +
     tower-error-artist tracks + random stratified by artist).
  2. Fits four candidate post-processing transforms on the subset
     (center_l2, whiten_l2, center_pca128, whiten_pca256).
  3. Evaluates each transform using:
     - Taxonomy coherence: mean genre-maxsim of top-6 neighbors for the
       8 diagnostic seeds (higher = better).
     - Spot-check: top-10 neighbors of each seed must not include any
       tower-error artists.
     - Cosine spread: p5 / p50 / p95 of pairwise cosines (wider = better
       discrimination).
  4. Prints a comparison table (including a towers baseline row) and the
     top-6 neighbor lists for each seed under each transform.
  5. Persists fitted transform parameters to mert_transform_calibration.npz.

Usage (defaults point to standard paths):
    python scripts/calibrate_mert_transform.py \\
        --sidecar data/artifacts/beat3tower_32k/mert_sidecar.npz \\
        --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz \\
        --subset-size 2000 \\
        --out data/artifacts/beat3tower_32k/mert_transform_calibration.npz

SAFETY: artifact and sidecar are read-only. metadata.db is opened in URI
read-only mode (SELECT only). The only file written is --out.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same artist-identity primitive the generator uses for seed-artist exclusion.
# We exclude the seed artist's own catalogue from the candidate pool so the
# eval mirrors real generation (principle 10/11: identity is computed; seed
# artist disallowed). Note: this is normalize-key parity with the generator's
# default — it does NOT resolve aliases (e.g. AFX↔Aphex Twin), matching the
# generator's own default behaviour.
from src.string_utils import normalize_artist_key  # noqa: E402

# ---------------------------------------------------------------------------
# Hardcoded constants from the spec
# ---------------------------------------------------------------------------

ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
SIDECAR = ROOT / "data/artifacts/beat3tower_32k/mert_sidecar.npz"
DB = ROOT / "data/metadata.db"
DEFAULT_OUT = ROOT / "data/artifacts/beat3tower_32k/mert_transform_calibration.npz"

# 8 diagnostic seeds (from C:\tmp\diag_mert.py)
DIAGNOSTIC_SEEDS: list[str] = [
    "e6759c60c1e976afafeab74b1eba8f94",  # Yeah Yeah Yeahs - Date With the Night
    "9d8f63a08b07cab0b5ebb55ba626d183",  # St. Vincent - Fast Slow Disco
    "acdca2d0ab79d5f3e843efdc1387369e",  # Caroline Rose - Bikini
    "f9890507d87189c8fadcca57ea4d3569",  # Arctic Monkeys - I Bet You Look Good on the Dancefloor
    "9b1fdfe28ecd7502ca299599b196033d",  # Sleigh Bells - Crown on the Ground
    "2daee06634a60134c5c4e8f1af43da1e",  # We Are Scientists - Nobody Move, Nobody Get Hurt
    "84b3329c46c059208724b75e7aacb0cd",  # Courtney Barnett - Pedestrian At Best
    "9ab264e12dac7b9a23077fde990d515a",  # Julia Jacklin - Pressure to Party
]

# Tower-error artists (case-insensitive substring match against artist name)
TOWER_ERROR_ARTISTS: list[str] = [
    "metallica",
    "glen campbell",
    "jay-z",
    "squarepusher",
    "autechre",
    "james brown",
    "the beatles",
    "busta rhymes",
    "de la soul",
    "mariah carey",
    "aaliyah",
]

# PCA component counts for the two PCA-based transforms (production values)
PCA_K_CENTER = 128
PCA_K_WHITEN = 256

# Default max tracks per artist during stratified sampling
DEFAULT_MAX_PER_ARTIST = 10

# Number of cosine-spread pairs to sample (speed vs accuracy)
MAX_SPREAD_PAIRS = 5_000

# Transforms saved to the output NPZ (usable in the fold pipeline)
LABELS_SAVE = ["center_l2", "whiten_l2", "center_pca128", "whiten_pca256",
               "power_l2_050", "power_l2_075"]

# Additional transforms evaluated in the table but NOT persisted (no fold path yet)
LABELS_EVAL_ONLY = ["umap_128", "umap_64"]

# Layer-selection labels (only run when --layers-sidecar is provided)
LAYER_SUBSETS = {
    "layers_all":    list(range(13)),     # sanity check: should ≈ center_l2
    "layers_upper4": list(range(9, 13)),  # layers 9-12
    "layers_top2":   list(range(11, 13)), # layers 11-12
    "layers_last1":  [12],               # final layer only
    "layers_mid4":   list(range(5, 9)),  # layers 5-8
    "layers_lower4": list(range(1, 5)),  # layers 1-4
}


# ---------------------------------------------------------------------------
# Sidecar / artifact loading helpers
# ---------------------------------------------------------------------------


def _load_sidecar(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (track_ids, emb_mid float32[N, D])."""
    z = np.load(path, allow_pickle=True)
    tids = [str(t) for t in z["track_ids"]]
    emb = np.asarray(z["emb_mid"], np.float32)
    return tids, emb


def _load_artifact_genre(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (track_ids, track_artists, track_titles, X_genre_raw, X_sonic_tower_weighted).

    All arrays aligned to artifact track_ids order.
    """
    z = np.load(path, allow_pickle=True)
    tids = [str(t) for t in z["track_ids"]]
    artists = np.array([str(a) for a in z["track_artists"]], dtype=object)
    titles = np.array([str(t) for t in z["track_titles"]], dtype=object)
    Xg = np.asarray(z["X_genre_raw"], np.float32)
    Xs = np.asarray(z["X_sonic_tower_weighted"], np.float32)
    return tids, artists, titles, Xg, Xs


def _load_artist_from_db(
    track_ids: list[str], db_path: Optional[Path]
) -> dict[str, str]:
    """Return {track_id: artist} for the given ids using metadata.db (read-only).

    Falls back to empty dict if db_path is None (synthetic tests).
    """
    if db_path is None or not Path(db_path).exists():
        return {}
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT track_id, artist FROM tracks"
        ).fetchall()
    finally:
        con.close()
    return {str(t): str(a) for t, a in rows}


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def build_subset(
    *,
    sidecar_path: Path,
    artifact_path: Path,
    db_path: Optional[Path],
    seed_ids: list[str],
    error_artist_ids: list[str],
    subset_size: int,
    rng: np.random.Generator,
    max_per_artist: int = DEFAULT_MAX_PER_ARTIST,
) -> list[str]:
    """Return a list of track_ids for the evaluation subset.

    Selection order:
      1. Diagnostic seeds (all that appear in the sidecar).
      2. Tower-error-artist tracks from the sidecar (deterministic from
         artifact artist strings; DB lookup not required for this).
      3. Random stratified remainder up to *subset_size*, capped at
         *max_per_artist* per artist across the full result.

    Inputs are read-only; no DB is required (error_artist_ids supplied by
    caller for the synthetic test path; real caller derives them from the
    artifact's track_artists).
    """
    sidecar_tids, _ = _load_sidecar(sidecar_path)
    sidecar_set = set(sidecar_tids)

    # Load artist info from artifact (used for stratification)
    art_z = np.load(artifact_path, allow_pickle=True)
    art_tids = [str(t) for t in art_z["track_ids"]]
    art_artists = [str(a) for a in art_z["track_artists"]]
    tid2artist = {t: a for t, a in zip(art_tids, art_artists)}

    # --- Phase 1: seeds -------------------------------------------------
    selected: list[str] = []
    selected_set: set[str] = set()

    for sid in seed_ids:
        if sid in sidecar_set and sid not in selected_set:
            selected.append(sid)
            selected_set.add(sid)

    # --- Phase 2: tower-error-artist tracks --------------------------------
    error_artist_set = {a.lower() for a in error_artist_ids}
    # Also scan the artifact artists directly if we have them
    if error_artist_set:
        for tid in sidecar_tids:
            if tid in selected_set:
                continue
            artist = tid2artist.get(tid, "").lower()
            if any(ea in artist for ea in error_artist_set):
                selected.append(tid)
                selected_set.add(tid)

    # --- Phase 3: stratified random remainder ----------------------------
    # Build per-artist counts already in selection
    artist_counts: Counter[str] = Counter(
        tid2artist.get(t, "") for t in selected
    )

    # Remaining sidecar candidates not yet selected
    candidates = [t for t in sidecar_tids if t not in selected_set]
    rng.shuffle(candidates)  # type: ignore[arg-type]

    for tid in candidates:
        if len(selected) >= subset_size:
            break
        artist = tid2artist.get(tid, "")
        if artist_counts[artist] >= max_per_artist:
            continue
        selected.append(tid)
        selected_set.add(tid)
        artist_counts[artist] += 1

    return selected


# ---------------------------------------------------------------------------
# Transform fitting
# ---------------------------------------------------------------------------


def fit_transform(
    label: str,
    X: np.ndarray,
    *,
    pca_k: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit a named transform on X and return (params_dict, X_transformed).

    Labels:
      - ``center_l2``:    subtract mean → L2-normalize
      - ``whiten_l2``:    subtract mean → divide by std (clamped ≥ 1e-4) → L2-normalize
      - ``center_pca128``: subtract mean → PCA(k) → L2-normalize
      - ``whiten_pca256``: subtract mean → divide by std → PCA(k) → L2-normalize

    *pca_k* overrides the default PCA dimension (used by tests with small data).
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, np.float32)
    params: dict[str, np.ndarray] = {}

    if label == "center_l2":
        mean = X.mean(axis=0)
        params[f"{label}_mean"] = mean
        Xc = X - mean
        Xt = _l2_normalize(Xc)

    elif label == "whiten_l2":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-4, 1e-4, std)
        params[f"{label}_mean"] = mean
        params[f"{label}_std"] = std
        Xc = (X - mean) / std
        Xt = _l2_normalize(Xc)

    elif label == "center_pca128":
        k = pca_k if pca_k is not None else PCA_K_CENTER
        k = min(k, X.shape[0] - 1, X.shape[1])
        mean = X.mean(axis=0)
        params[f"{label}_mean"] = mean
        Xc = X - mean
        pca = PCA(n_components=k, random_state=0)
        Xp = pca.fit_transform(Xc).astype(np.float32)
        params[f"{label}_components"] = pca.components_.astype(np.float32)
        Xt = _l2_normalize(Xp)

    elif label == "whiten_pca256":
        k = pca_k if pca_k is not None else PCA_K_WHITEN
        k = min(k, X.shape[0] - 1, X.shape[1])
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-4, 1e-4, std)
        params[f"{label}_mean"] = mean
        params[f"{label}_std"] = std
        Xw = (X - mean) / std
        pca = PCA(n_components=k, random_state=0)
        Xp = pca.fit_transform(Xw).astype(np.float32)
        params[f"{label}_components"] = pca.components_.astype(np.float32)
        Xt = _l2_normalize(Xp)

    elif label in ("power_l2_050", "power_l2_075"):
        alpha = 0.50 if label == "power_l2_050" else 0.75
        params[f"{label}_alpha"] = np.array(alpha, dtype=np.float32)
        Xp = np.sign(X) * np.abs(X) ** alpha
        Xt = _l2_normalize(Xp)

    elif label in ("umap_128", "umap_64"):
        try:
            import umap as umap_lib  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("umap-learn required: pip install umap-learn") from exc
        k = 128 if label == "umap_128" else 64
        print(f"    fitting UMAP(n_components={k}, metric='cosine') on {X.shape[0]} tracks...",
              flush=True)
        reducer = umap_lib.UMAP(n_components=k, random_state=0, metric="cosine",
                                n_neighbors=15, min_dist=0.1, verbose=False)
        Xt_raw = reducer.fit_transform(X.astype(np.float64)).astype(np.float32)
        Xt = _l2_normalize(Xt_raw)
        # UMAP model is not NPZ-serializable; params is intentionally empty.
        # If UMAP wins, persist via joblib in the fold script.

    else:
        raise ValueError(f"unknown transform label: {label!r}")

    return params, Xt


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize; zero rows become zero (not NaN)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms < 1e-9, 1.0, norms)


def fit_and_save_transforms(
    X: np.ndarray,
    out_path: Path,
    *,
    pca_k_center: Optional[int] = None,
    pca_k_whiten: Optional[int] = None,
    only_labels: Optional[list[str]] = None,
) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    """Fit the transforms on X, save params to *out_path*, return results.

    If *only_labels* is given, only those transforms are fit/saved (useful for
    focused experimental runs that skip the slow UMAP fit on large pools).

    Returns {label: (params, X_transformed)} for each label.
    """
    results: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    all_params: dict[str, np.ndarray] = {}

    all_labels = LABELS_SAVE + LABELS_EVAL_ONLY
    if only_labels is not None:
        all_labels = [lbl for lbl in all_labels if lbl in only_labels]
    for label in all_labels:
        kwargs: dict = {}
        if label == "center_pca128" and pca_k_center is not None:
            kwargs["pca_k"] = pca_k_center
        elif label == "whiten_pca256" and pca_k_whiten is not None:
            kwargs["pca_k"] = pca_k_whiten
        params, Xt = fit_transform(label, X, **kwargs)
        results[label] = (params, Xt)
        if label in LABELS_SAVE:  # only persist saveable transforms
            all_params.update(params)

    np.savez(out_path, **all_params)  # type: ignore[arg-type]
    return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_coherence(
    seed_indices: list[int],
    X_emb: np.ndarray,
    X_genre_raw: np.ndarray,
    k_neighbors: int = 6,
    artist_keys: Optional[np.ndarray] = None,
) -> float:
    """Mean genre-maxsim between each seed and its top-k neighbors in X_emb.

    Genre-maxsim(seed, neighbor) = cosine similarity of their X_genre_raw rows
    (L2-normalized).

    If *artist_keys* (one normalized key per row, aligned to X_emb) is given,
    every track sharing the seed's artist key is excluded from that seed's
    neighbor pool — mirroring the generator's seed-artist exclusion so the
    metric measures cross-catalogue discovery, not intra-catalogue self-match.

    Returns NaN if no seed indices are provided.
    """
    if not seed_indices:
        return float("nan")

    Xg = np.asarray(X_genre_raw, np.float32)
    g_norms = np.linalg.norm(Xg, axis=1, keepdims=True)
    Xg_norm = Xg / np.where(g_norms < 1e-9, 1.0, g_norms)

    X = np.asarray(X_emb, np.float32)
    # already expected to be L2-normalized; re-normalize defensively
    e_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(e_norms < 1e-9, 1.0, e_norms)

    n = X_norm.shape[0]
    coherences: list[float] = []
    for si in seed_indices:
        sims = X_norm @ X_norm[si]
        sims[si] = -2.0  # exclude self
        if artist_keys is not None:
            sims[artist_keys == artist_keys[si]] = -2.0  # exclude seed artist catalogue
        top_k = int(min(k_neighbors, n - 1))
        neighbor_idxs = np.argpartition(-sims, top_k)[:top_k]
        for ni in neighbor_idxs:
            coherences.append(float(Xg_norm[si] @ Xg_norm[ni]))

    return float(np.mean(coherences)) if coherences else float("nan")


def _spot_check_clean(
    seed_indices: list[int],
    X_emb: np.ndarray,
    error_indices: set[int],
    k_neighbors: int = 10,
) -> tuple[bool, list[int]]:
    """Cleaner version of spot-check that avoids the walrus-operator glitch."""
    if not seed_indices:
        return True, []

    X = np.asarray(X_emb, np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms < 1e-9, 1.0, norms)

    n = X_norm.shape[0]
    failures: list[int] = []
    for pos, si in enumerate(seed_indices):
        sims = X_norm @ X_norm[si]
        sims[si] = -2.0
        top_k = int(min(k_neighbors, n - 1))
        top_idxs = set(np.argpartition(-sims, top_k)[:top_k].tolist())
        if top_idxs & error_indices:
            failures.append(pos)

    return len(failures) == 0, failures


def compute_cosine_spread(
    X_emb: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = MAX_SPREAD_PAIRS,
) -> tuple[float, float, float]:
    """Return (p5, p50, p95) of sampled pairwise off-diagonal cosines."""
    X = np.asarray(X_emb, np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms < 1e-9, 1.0, norms)

    n = X_norm.shape[0]
    max_pairs = n * (n - 1) // 2
    actual_pairs = min(n_pairs, max_pairs)

    if actual_pairs <= 0:
        return 0.0, 0.0, 0.0

    # Sample random (i, j) pairs with i < j
    cosines: list[float] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = actual_pairs * 5

    while len(cosines) < actual_pairs and attempts < max_attempts:
        idx = rng.integers(0, n, size=(actual_pairs - len(cosines)) * 2)
        for k in range(0, len(idx) - 1, 2):
            i, j = int(idx[k]), int(idx[k + 1])
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            cosines.append(float(X_norm[i] @ X_norm[j]))
            if len(cosines) >= actual_pairs:
                break
        attempts += 1

    if not cosines:
        return 0.0, 0.0, 0.0

    arr = np.array(cosines, dtype=np.float32)
    return (
        float(np.percentile(arr, 5)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
    )


def _top_k_neighbors(
    seed_idx: int,
    X_emb: np.ndarray,
    k: int = 6,
    artist_keys: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return indices of top-k nearest neighbors to *seed_idx* (excluding self).

    If *artist_keys* is given, all tracks sharing the seed's artist key are
    excluded too (seed-artist exclusion, matching the generator).
    """
    X = np.asarray(X_emb, np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms < 1e-9, 1.0, norms)

    sims = X_norm @ X_norm[seed_idx]
    sims[seed_idx] = -2.0
    if artist_keys is not None:
        sims[artist_keys == artist_keys[seed_idx]] = -2.0
    actual_k = min(k, X_norm.shape[0] - 1)
    if actual_k <= 0:
        return np.array([], dtype=int)
    top_k = np.argpartition(-sims, actual_k)[:actual_k]
    return top_k[np.argsort(-sims[top_k])]


def _top_genres(Xg_row: np.ndarray, vocab: np.ndarray, k: int = 3) -> list[str]:
    """Return the top-k genre labels for a raw genre vector."""
    nz = np.nonzero(Xg_row > 0.05)[0]
    if len(nz) == 0:
        return ["(none)"]
    top = nz[np.argsort(-Xg_row[nz])][:k]
    return [str(vocab[int(j)]) for j in top]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_layer_subsets(
    layers_path: Path,
    id2art_idx: dict[str, int],
    art_artists: np.ndarray,
    art_titles: np.ndarray,
    Xg_full: np.ndarray,
    genre_vocab: np.ndarray,
) -> list[dict]:
    """Evaluate LAYER_SUBSETS from a per-layer sidecar (small targeted extraction).

    The per-layer sidecar stores emb_mid_layers: (N, 13, 768).  Each layer subset
    is mean-pooled → center → L2, then evaluated on its own N-track pool (which
    is typically just the 20 calibration seeds, so the neighbor lists are the
    primary evidence).

    Returns a list of row dicts in the same shape as the main comparison table,
    plus a 'neighbor_details' key for display.
    """
    z = np.load(layers_path, allow_pickle=True)
    layer_tids = [str(t) for t in z["track_ids"]]
    emb_layers = np.asarray(z["emb_mid_layers"], np.float32)  # (N, 13, 768)
    n_tracks, n_layers, _ = emb_layers.shape

    # Map track_ids → artifact positions for genre/artist lookup
    layer_art_idxs = [id2art_idx[t] for t in layer_tids if t in id2art_idx]
    resolved_tids = [t for t in layer_tids if t in id2art_idx]
    if not resolved_tids:
        return []

    emb_layers = emb_layers[[layer_tids.index(t) for t in resolved_tids]]
    Xg_sub = Xg_full[layer_art_idxs]
    all_idxs = list(range(len(resolved_tids)))  # all tracks = seeds in this pool

    rows: list[dict] = []
    for subset_name, layer_indices in LAYER_SUBSETS.items():
        valid = [i for i in layer_indices if i < n_layers]
        if not valid:
            continue
        X_sub = emb_layers[:, valid, :].mean(axis=1)  # (N, 768)
        mean = X_sub.mean(axis=0)
        Xc = X_sub - mean
        Xt = _l2_normalize(Xc)

        coherence = compute_coherence(all_idxs, Xt, Xg_sub, k_neighbors=min(5, len(all_idxs) - 1))
        cos_p5, cos_p50, cos_p95 = compute_cosine_spread(Xt, np.random.default_rng(42))

        # Per-seed neighbor detail
        details = []
        for si in all_idxs:
            ap = layer_art_idxs[si]
            s_artist = str(art_artists[ap])
            s_title = str(art_titles[ap])
            k = min(5, len(all_idxs) - 1)
            top_k = _top_k_neighbors(si, Xt, k=k)
            lines = []
            X_norm = _l2_normalize(Xt)
            for ni in top_k:
                nap = layer_art_idxs[ni]
                sim = float(X_norm[si] @ X_norm[ni])
                n_genres = _top_genres(Xg_sub[ni], genre_vocab)
                lines.append(
                    f"    {sim:+.3f}  {str(art_artists[nap])[:20]:20s}"
                    f" - {str(art_titles[nap])[:24]:24s}  {n_genres}"
                )
            details.append((s_artist, s_title, lines, 0.0))

        rows.append({
            "label": subset_name,
            "coherence": coherence,
            "spot_check": "N/A (layers)",
            "cos_p5": cos_p5,
            "cos_p50": cos_p50,
            "cos_p95": cos_p95,
            "dims": 768,
            "neighbor_details": details,
        })
    return rows


def _format_spot_check(passed: bool, n_seeds: int, failures: list[int]) -> str:
    if n_seeds == 0:
        return "N/A (0 seeds)"
    if passed:
        return f"PASS ({n_seeds}/{n_seeds})"
    n_pass = n_seeds - len(failures)
    return f"FAIL ({n_pass}/{n_seeds})"


def run_calibration(
    sidecar_path: Path,
    artifact_path: Path,
    db_path: Optional[Path],
    out_path: Path,
    subset_size: int,
    seed_ids: Optional[list[str]] = None,
    layers_sidecar_path: Optional[Path] = None,
    max_per_artist: int = DEFAULT_MAX_PER_ARTIST,
    only_transforms: Optional[list[str]] = None,
) -> None:
    """Full calibration pipeline — fits, evaluates, prints table, writes NPZ.

    *seed_ids* overrides DIAGNOSTIC_SEEDS when provided.  Custom seeds disable
    the tower-error spot-check (the check is only meaningful for the original
    indie-rock diagnostic seeds where Metallica/Jay-Z are clear failures).
    """
    rng = np.random.default_rng(42)

    custom_seeds = seed_ids is not None
    effective_seeds = seed_ids if custom_seeds else DIAGNOSTIC_SEEDS

    print(f"Loading sidecar: {sidecar_path}", flush=True)
    sidecar_tids, emb_mid_full = _load_sidecar(sidecar_path)
    print(f"  {len(sidecar_tids)} tracks in sidecar, emb dim={emb_mid_full.shape[1]}", flush=True)

    print(f"Loading artifact: {artifact_path}", flush=True)
    art_tids, art_artists, art_titles, Xg_full, Xs_towers_full = _load_artifact_genre(artifact_path)
    id2art_idx = {t: i for i, t in enumerate(art_tids)}
    art_vocab_z = np.load(artifact_path, allow_pickle=True)
    genre_vocab = np.array([str(v) for v in art_vocab_z["genre_vocab"]], dtype=object)

    # Build sidecar→artifact index mapping
    sidecar_art_idxs: list[int] = []
    sidecar_tids_resolved: list[str] = []
    for tid in sidecar_tids:
        if tid in id2art_idx:
            sidecar_art_idxs.append(id2art_idx[tid])
            sidecar_tids_resolved.append(tid)

    # Emit artist info from DB (optional; falls back to artifact artist strings)
    db_artist_map = _load_artist_from_db(sidecar_tids, db_path)

    # Build error-artist track_ids from the artifact (only used by built-in spot-check)
    error_artist_ids_resolved: list[str] = []
    error_lower = {a.lower() for a in TOWER_ERROR_ARTISTS}
    for tid, art_idx in zip(sidecar_tids_resolved, sidecar_art_idxs):
        artist_l = str(art_artists[art_idx]).lower()
        if any(ea in artist_l for ea in error_lower):
            error_artist_ids_resolved.append(tid)

    # Resolve seeds that appear in the sidecar
    sidecar_set = set(sidecar_tids_resolved)
    seed_ids_resolved = [s for s in effective_seeds if s in sidecar_set]
    n_total = len(effective_seeds)
    seed_label = "custom" if custom_seeds else "diagnostic"
    print(
        f"  Seeds resolved: {len(seed_ids_resolved)}/{n_total} ({seed_label})  "
        f"Tower-error tracks in sidecar: {len(error_artist_ids_resolved)}",
        flush=True,
    )

    # Build subset
    print(f"\nBuilding stratified subset (target size={subset_size})...", flush=True)
    subset_ids = build_subset(
        sidecar_path=sidecar_path,
        artifact_path=artifact_path,
        db_path=db_path,
        seed_ids=effective_seeds,
        error_artist_ids=TOWER_ERROR_ARTISTS if not custom_seeds else [],
        subset_size=subset_size,
        rng=rng,
        max_per_artist=max_per_artist,
    )
    print(f"  Subset size: {len(subset_ids)}", flush=True)

    # Map subset to arrays
    sid2sidecar_pos = {t: i for i, t in enumerate(sidecar_tids)}
    subset_sidecar_pos: list[int] = []
    subset_art_idxs: list[int] = []
    for tid in subset_ids:
        sp = sid2sidecar_pos.get(tid)
        ap = id2art_idx.get(tid)
        if sp is not None and ap is not None:
            subset_sidecar_pos.append(sp)
            subset_art_idxs.append(ap)

    # Normalized artist key per subset row (for seed-artist exclusion, parity
    # with the generator's identity-based pool gating).
    subset_artist_keys = np.array(
        [normalize_artist_key(str(art_artists[ap])) for ap in subset_art_idxs],
        dtype=object,
    )

    emb_subset = emb_mid_full[subset_sidecar_pos]        # (M, 768)
    Xg_subset = Xg_full[subset_art_idxs]                  # (M, G)
    Xs_subset = Xs_towers_full[subset_art_idxs]           # (M, 162)
    art_idx_in_subset = {art_idx: pos for pos, art_idx in enumerate(subset_art_idxs)}

    # Seed indices within subset
    seed_subset_idxs: list[int] = []
    seed_display: list[tuple[str, str]] = []  # (artist, title) for display
    for sid in effective_seeds:
        ap = id2art_idx.get(sid)
        if ap is not None and ap in art_idx_in_subset:
            seed_subset_idxs.append(art_idx_in_subset[ap])
            seed_display.append((str(art_artists[ap]), str(art_titles[ap])))

    # Error-artist indices within subset
    error_subset_idxs: set[int] = set()
    for pos, ap in enumerate(subset_art_idxs):
        artist_l = str(art_artists[ap]).lower()
        if any(ea in artist_l for ea in error_lower):
            error_subset_idxs.add(pos)

    print(
        f"  Seeds in subset: {len(seed_subset_idxs)}  "
        f"Error-artist tracks in subset: {len(error_subset_idxs)}",
        flush=True,
    )
    print(f"  emb_subset shape: {emb_subset.shape}", flush=True)

    # Fit all four transforms
    print("\nFitting transforms...", flush=True)
    transform_results = fit_and_save_transforms(
        emb_subset, out_path, only_labels=only_transforms
    )
    print(f"  Transform params saved to: {out_path}", flush=True)

    # Also prepare towers baseline (L2-normalize tower vectors)
    Xs_norm = _l2_normalize(Xs_subset)

    # Evaluate each transform
    LABELS_EVAL = LABELS_SAVE + LABELS_EVAL_ONLY
    if only_transforms is not None:
        LABELS_EVAL = [lbl for lbl in LABELS_EVAL if lbl in only_transforms]

    rows: list[dict] = []
    neighbor_details: dict[str, list[tuple[str, str, list[str], float]]] = {}
    # neighbor_details[label] = [(seed_artist, seed_title, [neighbor lines], sim), ...]

    for label in LABELS_EVAL:
        _params, Xt = transform_results[label]
        dims = Xt.shape[1]

        coherence = compute_coherence(
            seed_indices=seed_subset_idxs,
            X_emb=Xt,
            X_genre_raw=Xg_subset,
            k_neighbors=6,
            artist_keys=subset_artist_keys,
        )

        if custom_seeds:
            spot_str = "N/A (custom)"
        else:
            passed, fail_positions = _spot_check_clean(
                seed_indices=seed_subset_idxs,
                X_emb=Xt,
                error_indices=error_subset_idxs,
                k_neighbors=10,
            )
            spot_str = _format_spot_check(passed, len(seed_subset_idxs), fail_positions)

        cos_p5, cos_p50, cos_p95 = compute_cosine_spread(Xt, rng)

        rows.append({
            "label": label,
            "coherence": coherence,
            "spot_check": spot_str,
            "cos_p5": cos_p5,
            "cos_p50": cos_p50,
            "cos_p95": cos_p95,
            "dims": dims,
        })

        # Per-seed neighbor details
        details: list[tuple[str, str, list[str], float]] = []
        for si, (s_artist, s_title) in zip(seed_subset_idxs, seed_display):
            top6 = _top_k_neighbors(si, Xt, k=6, artist_keys=subset_artist_keys)
            neighbor_lines: list[str] = []
            for ni in top6:
                ap = subset_art_idxs[ni]
                n_artist = str(art_artists[ap])
                n_title = str(art_titles[ap])
                n_genres = _top_genres(Xg_subset[ni], genre_vocab)

                # compute sim
                X_norm_local = _l2_normalize(Xt)
                sim_val = float(X_norm_local[si] @ X_norm_local[ni])
                neighbor_lines.append(
                    f"    {sim_val:+.3f}  {n_artist[:20]:20s} - {n_title[:24]:24s}  {n_genres}"
                )
            details.append((s_artist, s_title, neighbor_lines, 0.0))
        neighbor_details[label] = details

    # Towers baseline row
    coherence_towers = compute_coherence(
        seed_indices=seed_subset_idxs,
        X_emb=Xs_norm,
        X_genre_raw=Xg_subset,
        k_neighbors=6,
        artist_keys=subset_artist_keys,
    )
    cos_p5_t, cos_p50_t, cos_p95_t = compute_cosine_spread(Xs_norm, rng)
    rows.append({
        "label": "towers_baseline",
        "coherence": coherence_towers,
        "spot_check": "—",
        "cos_p5": cos_p5_t,
        "cos_p50": cos_p50_t,
        "cos_p95": cos_p95_t,
        "dims": Xs_norm.shape[1],
    })

    # --- print comparison table -------------------------------------------
    print()
    print("=" * 90)
    print("MERT transform calibration — comparison table")
    print("=" * 90)
    hdr = f"{'transform':<18}  {'coherence':>10}  {'spot-check':>14}  {'cos-p5':>7}  {'cos-p50':>7}  {'cos-p95':>7}  {'dims':>5}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        coh = f"{r['coherence']:.3f}" if not (isinstance(r['coherence'], float) and r['coherence'] != r['coherence']) else "  N/A"
        print(
            f"{r['label']:<18}  {coh:>10}  {r['spot_check']:>14}  "
            f"{r['cos_p5']:>+7.3f}  {r['cos_p50']:>+7.3f}  {r['cos_p95']:>+7.3f}  {r['dims']:>5}"
        )
    print()

    # --- print per-seed neighbor lists ------------------------------------
    if seed_subset_idxs:
        for label in LABELS_EVAL:
            print(f"\n{'='*70}")
            print(f"Neighbors by transform: {label}")
            print(f"{'='*70}")
            for s_artist, s_title, neighbor_lines, _ in neighbor_details[label]:
                print(f"\n  SEED: {s_artist} - {s_title}")
                for line in neighbor_lines:
                    print(line)
    else:
        print("(No diagnostic seeds in subset — neighbor detail skipped.)")
        print("This is expected for the 50-track smoke sidecar which does not")
        print("contain the 8 diagnostic seeds (YYY, St. Vincent, etc.).")
        print()
        print("For the seeds-present evaluation, run Phase 3 (full extraction)")
        print("and re-run this script against the full sidecar.")

    # --- layer selection section (only when --layers-sidecar provided) -------
    if layers_sidecar_path is not None and layers_sidecar_path.exists():
        print(f"\n{'='*90}")
        print("Layer selection — comparison table (pool = seeds only)")
        print("=" * 90)
        layer_rows = evaluate_layer_subsets(
            layers_sidecar_path, id2art_idx,
            art_artists, art_titles, Xg_full, genre_vocab,
        )
        for r in layer_rows:
            coh = f"{r['coherence']:.3f}" if r['coherence'] == r['coherence'] else "  N/A"
            print(
                f"{r['label']:<20}  {coh:>10}  {r['spot_check']:>14}  "
                f"{r['cos_p5']:>+7.3f}  {r['cos_p50']:>+7.3f}  {r['cos_p95']:>+7.3f}  {r['dims']:>5}"
            )
        # Neighbor lists for first 4 seeds
        for r in layer_rows:
            print(f"\n{'='*70}")
            print(f"Layer neighbors: {r['label']}")
            print("="*70)
            for s_artist, s_title, lines, _ in r["neighbor_details"][:4]:
                print(f"\n  SEED: {s_artist} - {s_title}")
                for line in lines:
                    print(line)

    print(f"\nCalibration params written to: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sidecar",
        type=Path,
        default=SIDECAR,
        help=f"Merged MERT sidecar NPZ (default: {SIDECAR})",
    )
    ap.add_argument(
        "--artifact",
        type=Path,
        default=ARTIFACT,
        help=f"Artifact NPZ with genre+sonic matrices (default: {ARTIFACT})",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=DB,
        help=f"metadata.db for artist lookup (default: {DB}; optional)",
    )
    ap.add_argument(
        "--subset-size",
        type=int,
        default=2000,
        help="Target evaluation subset size (default: 2000)",
    )
    ap.add_argument(
        "--max-per-artist",
        type=int,
        default=DEFAULT_MAX_PER_ARTIST,
        help=(
            f"Max tracks per artist in the eval pool (default: {DEFAULT_MAX_PER_ARTIST}). "
            "Set high (e.g. 1000000) together with a large --subset-size to rank seeds "
            "against the entire scanned library with no per-artist cap."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output NPZ for transform params (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--seeds",
        type=Path,
        default=None,
        help=(
            "Optional file of seed track IDs (one per line) to use instead of the "
            "8 hardcoded diagnostic seeds. Tower-error spot-check is skipped for "
            "custom seeds; use the neighbor lists for manual quality assessment."
        ),
    )
    ap.add_argument(
        "--transforms",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of transforms to fit/evaluate (e.g. 'whiten_l2'). "
            "Skips fitting the rest — notably the slow UMAP fit on large pools. "
            "The towers baseline is always included. Default: all transforms."
        ),
    )
    ap.add_argument(
        "--layers-sidecar",
        type=Path,
        default=None,
        help=(
            "Per-layer sidecar NPZ (from extract_mert_sidecar.py --layers-out). "
            "When provided, appends a layer-selection comparison section to the output."
        ),
    )
    args = ap.parse_args()

    custom_seed_ids: Optional[list[str]] = None
    if args.seeds is not None:
        custom_seed_ids = [
            line.strip() for line in args.seeds.read_text().splitlines() if line.strip()
        ]

    run_calibration(
        sidecar_path=args.sidecar,
        artifact_path=args.artifact,
        db_path=args.db,
        out_path=args.out,
        subset_size=args.subset_size,
        seed_ids=custom_seed_ids,
        layers_sidecar_path=args.layers_sidecar,
        max_per_artist=args.max_per_artist,
        only_transforms=(
            [t.strip() for t in args.transforms.split(",") if t.strip()]
            if args.transforms
            else None
        ),
    )


if __name__ == "__main__":
    main()
