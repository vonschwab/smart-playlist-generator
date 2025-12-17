"""
Artist-aware candidate pool builder for playlist generation experiments.

Loads Step 1 artifacts (data_matrices_step1.npz), builds hybrid
embeddings (sonic PCA + smoothed genre PCA), computes seed similarities,
and assembles an artist-diverse candidate pool based on mode-specific
caps and similarity floors.
"""

import argparse
import logging
import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DEFAULT_ARTIFACT = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"

logger = logging.getLogger(__name__)


def _fit_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """Standardize X and fit PCA to the requested component count."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    embeddings = pca.fit_transform(X_scaled)
    return embeddings, scaler, pca


def _nonzero_genres(row: np.ndarray, genre_vocab: Sequence[str], limit: int = 10) -> List[str]:
    """Return up to `limit` genre names where the genre vector is non-zero."""
    nz = np.nonzero(row)[0]
    return [str(genre_vocab[i]) for i in nz[:limit]]


def _mode_presets(mode: str, playlist_length: int) -> Dict[str, float]:
    """Return mode-specific knobs for artist caps and similarity filtering."""
    mode = mode.lower()
    if mode not in {"narrow", "dynamic", "discover"}:
        raise ValueError(f"Unknown mode: {mode}")

    max_artist_fraction_final = {
        "narrow": 0.20,
        "dynamic": 0.125,
        "discover": 0.05,
    }[mode]

    similarity_floor = {
        "narrow": 0.35,
        "dynamic": 0.30,
        "discover": 0.25,
    }[mode]

    max_pool_size = {
        "narrow": 800,
        "dynamic": 1200,
        "discover": 2000,
    }[mode]

    target_artists = {
        "narrow": max(math.ceil(playlist_length / 2), 12),
        "dynamic": max(math.ceil(0.75 * playlist_length), 16),
        "discover": min(playlist_length, 24),
    }[mode]

    max_per_artist_final = math.ceil(playlist_length * max_artist_fraction_final)

    if mode == "narrow":
        candidate_per_artist = max(3, min(2 * max_per_artist_final, 8))
    elif mode == "dynamic":
        candidate_per_artist = max(3, min(2 * max_per_artist_final, 6))
    else:
        candidate_per_artist = max(2, min(2 * max_per_artist_final, 4))

    seed_candidate_cap = candidate_per_artist + 2

    return {
        "max_artist_fraction_final": max_artist_fraction_final,
        "similarity_floor": similarity_floor,
        "max_pool_size": max_pool_size,
        "target_artists": target_artists,
        "max_per_artist_final": max_per_artist_final,
        "candidate_per_artist": candidate_per_artist,
        "seed_candidate_cap": seed_candidate_cap,
    }


def _safe_str(value: np.ndarray, idx: int) -> str:
    """Ensure bytes from np.load are converted to str."""
    item = value[idx]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="ignore")
    return str(item)


def _build_hybrid_embedding(
    X_sonic: np.ndarray,
    X_genre: np.ndarray,
    n_components_sonic: int,
    n_components_genre: int,
    w_sonic: float,
    w_genre: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, PCA]:
    """Fit PCA on sonic/genre matrices and return the hybrid embedding."""
    n_comp_sonic = min(n_components_sonic, X_sonic.shape[1], X_sonic.shape[0])
    n_comp_genre = min(n_components_genre, X_genre.shape[1], X_genre.shape[0])
    if n_comp_sonic < n_components_sonic or n_comp_genre < n_components_genre:
        logger.info(
            "Adjusted n_components: sonic=%d, genre=%d (requested sonic=%d, genre=%d)",
            n_comp_sonic,
            n_comp_genre,
            n_components_sonic,
            n_components_genre,
        )

    E_sonic, _, pca_sonic = _fit_pca(X_sonic, n_comp_sonic)
    E_genre, _, pca_genre = _fit_pca(X_genre, n_comp_genre)

    hybrid = np.concatenate([w_sonic * E_sonic, w_genre * E_genre], axis=1)
    return hybrid, E_sonic, E_genre, pca_sonic, pca_genre


def _compute_seed_sims(hybrid: np.ndarray, seed_idx: int) -> np.ndarray:
    """Cosine similarity of every track to the seed track."""
    sims = cosine_similarity(hybrid[seed_idx : seed_idx + 1], hybrid)[0]
    sims[seed_idx] = -1.0
    return sims


def _group_by_artist(indices: Sequence[int], artist_keys: Sequence[str]) -> Dict[str, List[int]]:
    """Group candidate indices by artist_key."""
    groups: Dict[str, List[int]] = {}
    for idx in indices:
        key = artist_keys[idx]
        groups.setdefault(key, []).append(idx)
    return groups


def _dedupe_preserve_order(indices: Sequence[int]) -> List[int]:
    """Deduplicate indices while preserving the first occurrence."""
    seen = set()
    deduped = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


def _top_artists_summary(
    pool_indices: Sequence[int],
    artist_keys: Sequence[str],
    artist_names: Sequence[str],
    sims: np.ndarray,
    top_n: int = 10,
) -> List[Tuple[str, str, int, float, float]]:
    """Aggregate counts and similarity stats per artist for reporting."""
    stats: Dict[str, Dict[str, float]] = {}
    for idx in pool_indices:
        key = artist_keys[idx]
        stats.setdefault(
            key,
            {"count": 0, "max_sim": -1.0, "sum_sim": 0.0, "name": _safe_str(artist_names, idx)},
        )
        stats[key]["count"] += 1
        stats[key]["sum_sim"] += sims[idx]
        stats[key]["max_sim"] = max(stats[key]["max_sim"], sims[idx])

    rows = []
    for key, payload in stats.items():
        count = int(payload["count"])
        max_sim = float(payload["max_sim"])
        mean_sim = float(payload["sum_sim"] / count) if count else 0.0
        name = payload["name"]
        rows.append((key, name, count, max_sim, mean_sim))

    rows.sort(key=lambda r: (-r[2], -r[3]))
    return rows[:top_n]


def build_candidate_pool(
    seed_idx: int,
    hybrid: np.ndarray,
    seed_sims: np.ndarray,
    artist_keys: Sequence[str],
    mode_settings: Dict[str, float],
) -> Tuple[List[int], List[int], Dict[str, List[int]]]:
    """Construct the artist-aware candidate pool."""
    similarity_floor = mode_settings["similarity_floor"]
    max_pool_size = mode_settings["max_pool_size"]
    target_artists = mode_settings["target_artists"]
    candidate_per_artist = mode_settings["candidate_per_artist"]
    seed_candidate_cap = mode_settings["seed_candidate_cap"]

    seed_artist_key = artist_keys[seed_idx]

    eligible = [
        i
        for i, sim in enumerate(seed_sims)
        if i != seed_idx and sim >= similarity_floor
    ]
    artist_groups = _group_by_artist(eligible, artist_keys)

    artist_rank = []
    for artist_key, idxs in artist_groups.items():
        best_sim = max(seed_sims[i] for i in idxs)
        artist_rank.append((artist_key, best_sim, idxs))
    artist_rank.sort(key=lambda t: -t[1])

    pool_indices: List[int] = []
    pool_artist_set = set()

    for artist_key, _, idxs in artist_rank:
        per_artist_cap = seed_candidate_cap if artist_key == seed_artist_key else candidate_per_artist
        sorted_idxs = sorted(idxs, key=lambda i: -seed_sims[i])
        take = sorted_idxs[:per_artist_cap]
        for idx in take:
            if len(pool_indices) >= max_pool_size and len(pool_artist_set) >= target_artists:
                break
            pool_indices.append(idx)
        pool_artist_set.add(artist_key)
        if len(pool_indices) >= max_pool_size and len(pool_artist_set) >= target_artists:
            break

    pool_indices = _dedupe_preserve_order(pool_indices)
    return pool_indices, list(pool_artist_set), artist_groups


def main():
    parser = argparse.ArgumentParser(description="Artist-aware candidate pool lab.")
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT,
        help=f"Path to Step 1 artifact (default: {DEFAULT_ARTIFACT})",
    )
    parser.add_argument("--seed-track-id", required=True, help="Seed track_id to anchor the pool.")
    parser.add_argument("--playlist-length", type=int, default=25, help="Intended final playlist length (default: 25).")
    parser.add_argument(
        "--mode",
        choices=["narrow", "dynamic", "discover"],
        default="narrow",
        help="Mode presets controlling caps/floors (default: narrow).",
    )
    parser.add_argument("--w-sonic", type=float, default=0.6, help="Weight for sonic embedding (default: 0.6).")
    parser.add_argument("--w-genre", type=float, default=0.4, help="Weight for genre embedding (default: 0.4).")
    parser.add_argument(
        "--n-components-sonic",
        type=int,
        default=32,
        help="PCA components for sonic space (default: 32).",
    )
    parser.add_argument(
        "--n-components-genre",
        type=int,
        default=32,
        help="PCA components for genre space (default: 32).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for tie-breaking (default: 42).")
    parser.add_argument(
        "--save-pool-npz",
        help="Optional path to save the candidate pool npz (pool_indices, seed sims, metadata).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.random_seed)

    try:
        data = np.load(args.artifact_path, allow_pickle=True)
    except FileNotFoundError:
        print(
            f"Artifact not found at {args.artifact_path}. "
            "Run data_matrix_lab.py with --save-artifacts to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_sonic = data["X_sonic"]
    X_genre_smoothed = data["X_genre_smoothed"] if "X_genre_smoothed" in data else data["X_genre"]
    X_genre_raw = data["X_genre_raw"] if "X_genre_raw" in data else data["X_genre"]
    track_ids = data["track_ids"]
    artist_names = data["artist_names"]
    track_titles = data["track_titles"]
    artist_keys = data["artist_keys"] if "artist_keys" in data else np.array([""] * len(track_ids))
    genre_vocab = data["genre_vocab"]

    def _normalize_artist_key(raw: str) -> str:
        txt = (str(raw) if raw is not None else "").strip().lower()
        if txt in {"", "unknown", "nan"}:
            return ""
        return txt

    # Fallback for bad/missing artist_keys
    missing_mask = np.array(
        [
            (_normalize_artist_key(k) == "")
            for k in artist_keys
        ]
    )
    missing_frac = missing_mask.mean() if len(missing_mask) else 0.0
    artist_keys_effective = np.array([_normalize_artist_key(k) for k in artist_keys], dtype=object)
    if missing_frac > 0.02:
        logger.warning(
            "Artist keys missing for %.2f%% of tracks; falling back to normalized track_artists/track_ids for grouping",
            missing_frac * 100,
        )
        for idx, is_missing in enumerate(missing_mask):
            if not is_missing:
                continue
            fallback = _normalize_artist_key(artist_names[idx])
            if fallback:
                artist_keys_effective[idx] = fallback
            else:
                artist_keys_effective[idx] = f"unknown:{_safe_str(track_ids, idx)}"
    else:
        artist_keys_effective = artist_keys

    # Build hybrid embedding
    hybrid, E_sonic, E_genre, pca_sonic, pca_genre = _build_hybrid_embedding(
        X_sonic,
        X_genre_smoothed,
        args.n_components_sonic,
        args.n_components_genre,
        args.w_sonic,
        args.w_genre,
    )

    seed_matches = np.where(track_ids == args.seed_track_id)[0]
    if len(seed_matches) == 0:
        print(f"Seed track_id not found: {args.seed_track_id}", file=sys.stderr)
        sys.exit(1)
    seed_idx = int(seed_matches[0])

    seed_sims = _compute_seed_sims(hybrid, seed_idx)

    presets = _mode_presets(args.mode, args.playlist_length)
    pool_indices, pool_artist_list, artist_groups = build_candidate_pool(
        seed_idx, hybrid, seed_sims, artist_keys_effective, presets
    )

    if args.save_pool_npz:
        out_path = Path(args.save_pool_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "mode": args.mode,
            "playlist_length": args.playlist_length,
            "w_sonic": args.w_sonic,
            "w_genre": args.w_genre,
            "n_components_sonic": args.n_components_sonic,
            "n_components_genre": args.n_components_genre,
            "random_seed": args.random_seed,
            "presets": presets,
            "seed_track_id": args.seed_track_id,
        }
        np.savez(
            out_path,
            pool_indices=np.array(pool_indices, dtype=int),
            pool_track_ids=track_ids[np.array(pool_indices, dtype=int)],
            pool_seed_sim=seed_sims[np.array(pool_indices, dtype=int)],
            pool_artist_keys=artist_keys_effective[np.array(pool_indices, dtype=int)],
            seed_track_id=args.seed_track_id,
            params_json=json.dumps(params),
        )
        logger.info("Saved candidate pool to %s", out_path)

    # Diagnostics
    seed_genres = _nonzero_genres(X_genre_raw[seed_idx], genre_vocab, limit=10)
    seed_artist = _safe_str(artist_names, seed_idx)
    seed_title = _safe_str(track_titles, seed_idx)
    seed_artist_key = _safe_str(artist_keys, seed_idx)
    seed_artist_key_effective = _safe_str(artist_keys_effective, seed_idx)

    print("=== Seed Summary ===")
    print(f"Seed track_id : {args.seed_track_id}")
    print(f"Seed artist   : {seed_artist}")
    print(f"Seed title    : {seed_title}")
    print(f"Seed artist_key (orig): {seed_artist_key}")
    print(f"Seed artist_key (effective): {seed_artist_key_effective}")
    print(f"Seed genres   : {seed_genres}")
    print()

    print("=== Mode & Config ===")
    print(f"Mode: {args.mode}")
    print(f"Playlist length (L): {args.playlist_length}")
    print(f"max_artist_fraction_final: {presets['max_artist_fraction_final']}")
    print(f"max_per_artist_final: {presets['max_per_artist_final']}")
    print(f"similarity_floor: {presets['similarity_floor']}")
    print(f"candidate_per_artist: {presets['candidate_per_artist']}")
    print(f"seed_candidate_cap: {presets['seed_candidate_cap']}")
    print(f"target_artists: {presets['target_artists']}")
    print(f"max_pool_size: {presets['max_pool_size']}")
    print()

    print("=== Pool Summary ===")
    print(f"Candidate pool size: {len(pool_indices)}")
    print(f"Distinct artists   : {len(pool_artist_list)}")
    print(f"Eligible artists   : {len(artist_groups)}")
    print()

    top_artists = _top_artists_summary(pool_indices, artist_keys_effective, artist_names, seed_sims, top_n=12)
    if top_artists:
        header = f"{'artist_key':<20} | {'artist':<30} | {'tracks':>6} | {'max_sim':>7} | {'mean_sim':>7}"
        print("Top artists in pool:")
        print(header)
        print("-" * len(header))
        for key, name, count, max_sim, mean_sim in top_artists:
            print(f"{key:<20} | {str(name)[:30]:<30} | {count:>6} | {max_sim:>7.3f} | {mean_sim:>7.3f}")
        print()

    # Preview top tracks by similarity
    preview_count = min(15, len(pool_indices))
    if preview_count:
        print("Preview (sorted by seed similarity):")
    header = f"{'rank':>4} | {'sim':>6} | {'track_id':<32} | {'artist  title':<60} | {'artist_key_eff':<20}"
    print(header)
    print("-" * len(header))
    sorted_pool = sorted(pool_indices, key=lambda i: -seed_sims[i])
    for rank, idx in enumerate(sorted_pool[:preview_count], start=1):
        track_id = _safe_str(track_ids, idx)
        artist = _safe_str(artist_names, idx)
        title = _safe_str(track_titles, idx)
        artist_key = _safe_str(artist_keys_effective, idx)
        print(
            f"{rank:>4} | {seed_sims[idx]:>6.3f} | {track_id:<32} | "
            f"{(artist + '  ' + title)[:60]:<60} | {artist_key:<20}"
        )


if __name__ == "__main__":
    main()
