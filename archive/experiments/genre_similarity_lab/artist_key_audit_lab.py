"""
Audit artist_keys coverage inside data_matrices_step1.npz artifacts.

Reports missing fractions, bucket sizes, and examples to help diagnose
why artist grouping may collapse in downstream labs.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_ARTIFACT = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"

logger = logging.getLogger(__name__)


def _as_str(value) -> str:
    """Safe string conversion for numpy scalars/bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("cp1252", errors="ignore")
    return str(value)


def _is_missing(key: str) -> bool:
    """Decide if an artist_key is missing/invalid."""
    key = key.strip().lower()
    return key in {"", "unknown", "nan", "none"}


def _most_common_display(indices: List[int], artist_names: np.ndarray) -> str:
    """Most frequent display artist for a bucket of indices."""
    ctr = Counter(_as_str(artist_names[i]) for i in indices)
    if not ctr:
        return ""
    return ctr.most_common(1)[0][0]


def audit(
    artifact_path: str,
    top_n: int,
    seed_track_id: str,
    fail_if_missing_frac: float,
) -> int:
    """Run audit; return exit code."""
    try:
        data = np.load(artifact_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Artifact not found: {artifact_path}", file=sys.stderr)
        return 1

    if "track_ids" not in data or "track_titles" not in data or "artist_keys" not in data:
        print("Artifact missing required arrays (track_ids/track_titles/artist_keys).", file=sys.stderr)
        return 1

    track_ids = data["track_ids"]
    track_titles = data["track_titles"]
    if "track_artists" in data:
        track_artists = data["track_artists"]
    elif "artist_names" in data:
        track_artists = data["artist_names"]
    else:
        print("Artifact missing track_artists/artist_names array.", file=sys.stderr)
        return 1
    artist_keys = data["artist_keys"]

    n = len(track_ids)
    missing_indices = []
    bucket: Dict[str, List[int]] = defaultdict(list)
    for idx in range(n):
        key = _as_str(artist_keys[idx])
        if _is_missing(key):
            missing_indices.append(idx)
        bucket[key].append(idx)

    n_missing = len(missing_indices)
    missing_frac = n_missing / n if n else 0.0
    largest_key, largest_size = max(((k, len(v)) for k, v in bucket.items()), key=lambda t: t[1])
    unique_keys = len(bucket)

    print("=== Artist Key Audit ===")
    print(f"Artifact: {artifact_path}")
    print(f"Total tracks: {n}")
    print(f"Missing artist_keys: {n_missing} ({missing_frac:.2%})")
    print(f"Unique artist_keys: {unique_keys}")
    print(f"Largest bucket: '{largest_key}' size={largest_size} example_display='{_most_common_display(bucket[largest_key], track_artists)}'")
    print()

    ctr = Counter({k: len(v) for k, v in bucket.items()})
    print(f"Top {top_n} artist_keys by count:")
    for key, count in ctr.most_common(top_n):
        display = _most_common_display(bucket[key], track_artists)
        print(f"  {key!r:<30} count={count:<6} example_artist={display}")
    print()

    if missing_indices:
        print("Example rows with missing artist_keys (first 20):")
        for idx in missing_indices[:20]:
            tid = _as_str(track_ids[idx])
            artist = _as_str(track_artists[idx])
            title = _as_str(track_titles[idx])
            key = _as_str(artist_keys[idx])
            print(f"  {tid} | {artist} | {title} | key={key!r}")
        print()

    if seed_track_id:
        matches = np.where(track_ids == seed_track_id)[0]
        if len(matches):
            seed_idx = int(matches[0])
            tid = _as_str(track_ids[seed_idx])
            artist = _as_str(track_artists[seed_idx])
            title = _as_str(track_titles[seed_idx])
            key = _as_str(artist_keys[seed_idx])
            print(f"Seed track row: idx={seed_idx} | {tid} | {artist} | {title} | key={key!r}")
            lo = max(0, seed_idx - 3)
            hi = min(n, seed_idx + 4)
            print("Nearby rows:")
            for i in range(lo, hi):
                tid_i = _as_str(track_ids[i])
                artist_i = _as_str(track_artists[i])
                title_i = _as_str(track_titles[i])
                key_i = _as_str(artist_keys[i])
                print(f"  idx={i:<5} {tid_i} | {artist_i} | {title_i} | key={key_i!r}")
            print()
        else:
            print(f"Seed track_id {seed_track_id} not found in artifact.")

    if missing_frac > fail_if_missing_frac:
        print(
            f"Missing fraction {missing_frac:.2%} exceeds threshold {fail_if_missing_frac:.2%}; exiting non-zero.",
            file=sys.stderr,
        )
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description="Audit artist_keys coverage in data_matrices_step1.npz.")
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT,
        help=f"Path to artifact (default: {DEFAULT_ARTIFACT})",
    )
    parser.add_argument("--top-n", type=int, default=30, help="How many top artist_keys to display (default: 30).")
    parser.add_argument("--seed-track-id", help="Optional seed track_id to inspect nearby rows.")
    parser.add_argument(
        "--fail-if-missing-frac",
        type=float,
        default=0.10,
        help="Exit non-zero if missing fraction exceeds this value (default: 0.10).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(audit(args.artifact_path, args.top_n, args.seed_track_id, args.fail_if_missing_frac))


if __name__ == "__main__":
    main()
