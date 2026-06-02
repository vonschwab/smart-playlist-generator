#!/usr/bin/env python3
"""
Measure Genre Baseline
======================

Snapshot genre cosine distribution stats for four reference cases before any
genre subsystem redesign changes.  Idempotent and read-only.

Usage:
    python scripts/measure_genre_baseline.py \
        data/artifacts/beat3tower_32k/data_matrices_step1.npz \
        [--output tests/fixtures/genre_redesign/baseline_metrics.json]

Running this script against the same artifact always produces the same JSON.
Check the output into tests/fixtures/genre_redesign/baseline_metrics.json so
the post-redesign calibration scripts can compare against this snapshot.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Four reference cases: (case_name, track_id, artist, title)
REFERENCE_CASES: List[Dict[str, str]] = [
    {
        "name": "acetone",
        "track_id": "6d5696a29b57a765f945ed1be2d6dfee",
        "artist": "Acetone",
        "title": "Every Kiss",
        "description": "narrow-niche raw — baseline OK case",
    },
    {
        "name": "charli_xcx",
        "track_id": "065933d8e2e0db664ec57af1511b662b",
        "artist": "Charli XCX",
        "title": "The girl, so confusing version with lorde",
        "description": "dense-niche enriched — known failure case post-enrichment",
    },
    {
        "name": "pharoah_sanders",
        "track_id": "a3bb6db554f1d2bae1a8b998ebf53925",
        "artist": "Pharoah Sanders",
        "title": "The Creator Has a Master Plan",
        "description": "rich-mixed raw — healthy baseline case",
    },
    {
        "name": "beach_boys",
        "track_id": "fc302980e8359e5bab53e7a2f45fc61b",
        "artist": "The Beach Boys",
        "title": "Little Deuce Coupe (Mono)",
        "description": "sparse raw — healthy baseline case",
    },
]

FLOOR_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
HASH_CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 fingerprint without loading the full artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine_genre_sim(seed_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between seed and every row in matrix. Matches candidate_pool logic."""
    seed_norm = seed_vec / (np.linalg.norm(seed_vec) + 1e-12)
    cand_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(cand_norm, seed_norm)
    return np.clip(sims, 0.0, 1.0)


def _ensemble_genre_sim(seed_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """0.6*cosine + 0.4*jaccard — matches candidate_pool 'ensemble' method."""
    seed_binary = (seed_vec > 0).astype(float)
    cand_binary = (matrix > 0).astype(float)
    intersection = np.dot(cand_binary, seed_binary)
    union = cand_binary.sum(axis=1) + seed_binary.sum() - intersection
    union = np.maximum(union, 1e-12)
    jaccard = intersection / union

    seed_norm = seed_vec / (np.linalg.norm(seed_vec) + 1e-12)
    cand_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    cosine = np.clip(np.dot(cand_norm, seed_norm), 0.0, 1.0)

    return 0.6 * cosine + 0.4 * jaccard


def _distribution_stats(sims: np.ndarray, exclude_idx: int) -> Dict[str, float]:
    mask = np.ones(len(sims), dtype=bool)
    mask[exclude_idx] = False
    s = sims[mask]
    return {
        "min": float(np.min(s)),
        "p50": float(np.percentile(s, 50)),
        "p90": float(np.percentile(s, 90)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "max": float(np.max(s)),
        "mean": float(np.mean(s)),
    }


def _candidates_at_floors(sims: np.ndarray, exclude_idx: int) -> Dict[str, int]:
    mask = np.ones(len(sims), dtype=bool)
    mask[exclude_idx] = False
    s = sims[mask]
    return {str(floor): int(np.sum(s >= floor)) for floor in FLOOR_THRESHOLDS}


def measure_case(
    case: Dict[str, str],
    track_ids: np.ndarray,
    X_genre_raw: np.ndarray,
    genre_vocab: np.ndarray,
) -> Dict[str, Any]:
    tid = case["track_id"]
    id_to_idx = {str(t): i for i, t in enumerate(track_ids)}
    idx = id_to_idx.get(tid)
    if idx is None:
        return {
            "name": case["name"],
            "error": f"track_id {tid} not found in artifact",
        }

    seed_vec = X_genre_raw[idx]
    active_dims = int(np.sum(seed_vec > 0))
    seed_l2_norm = float(np.linalg.norm(seed_vec))
    active_genres = [str(genre_vocab[i]) for i in np.where(seed_vec > 0)[0]]

    cosine_sims = _cosine_genre_sim(seed_vec, X_genre_raw)
    ensemble_sims = _ensemble_genre_sim(seed_vec, X_genre_raw)

    return {
        "name": case["name"],
        "artist": case["artist"],
        "title": case["title"],
        "description": case["description"],
        "track_id": tid,
        "artifact_idx": idx,
        "seed_genre_count": active_dims,
        "seed_l2_norm": seed_l2_norm,
        "active_genres": active_genres,
        "cosine": {
            "distribution": _distribution_stats(cosine_sims, idx),
            "candidates_at_floor": _candidates_at_floors(cosine_sims, idx),
        },
        "ensemble": {
            "distribution": _distribution_stats(ensemble_sims, idx),
            "candidates_at_floor": _candidates_at_floors(ensemble_sims, idx),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot genre baseline metrics")
    parser.add_argument(
        "artifact",
        help="Path to data_matrices_step1.npz",
    )
    parser.add_argument(
        "--output",
        default="tests/fixtures/genre_redesign/baseline_metrics.json",
        help="Where to write the JSON snapshot",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against existing snapshot and report diffs instead of writing",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"ERROR: artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading artifact: {artifact_path}")
    data = np.load(artifact_path, allow_pickle=True)
    track_ids = data["track_ids"]
    X_genre_raw = data["X_genre_raw"]
    genre_vocab = data["genre_vocab"]

    print(f"  {len(track_ids)} tracks, {X_genre_raw.shape[1]}-dim genre vocab")

    results = []
    for case in REFERENCE_CASES:
        print(f"  Measuring: {case['name']} ({case['artist']} — {case['title']})")
        result = measure_case(case, track_ids, X_genre_raw, genre_vocab)
        results.append(result)

        if "error" in result:
            print(f"    WARNING: {result['error']}")
        else:
            c = result["cosine"]["candidates_at_floor"]
            print(f"    active_genres={result['seed_genre_count']}, "
                  f"cosine@0.1={c.get('0.1', 0)}, cosine@0.3={c.get('0.3', 0)}, "
                  f"cosine@0.4={c.get('0.4', 0)}")

    # SHA-256 fingerprint so we can verify which artifact the baseline was run against.
    # The artifact is ~400 MB — too large to commit; we store the hash instead.
    sha256 = _sha256_file(artifact_path)

    output = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256,
        "artifact_shape": {
            "n_tracks": int(len(track_ids)),
            "genre_vocab_size": int(X_genre_raw.shape[1]),
        },
        "floor_thresholds": FLOOR_THRESHOLDS,
        "cases": results,
    }

    output_path = Path(args.output)

    if args.check:
        if not output_path.exists():
            print(f"No baseline at {output_path} — nothing to compare against.", file=sys.stderr)
            sys.exit(1)
        with open(output_path) as f:
            baseline = json.load(f)
        _compare(baseline, output)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBaseline written to: {output_path}")


def _compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> None:
    baseline_cases = {c["name"]: c for c in baseline.get("cases", [])}
    current_cases = {c["name"]: c for c in current.get("cases", [])}
    diffs: List[str] = []

    _append_value_diffs("artifact_sha256", baseline.get("artifact_sha256"), current.get("artifact_sha256"), diffs)
    _append_value_diffs("artifact_shape", baseline.get("artifact_shape"), current.get("artifact_shape"), diffs)
    _append_value_diffs("floor_thresholds", baseline.get("floor_thresholds"), current.get("floor_thresholds"), diffs)

    for name in sorted(baseline_cases.keys() - current_cases.keys()):
        diffs.append(f"  REMOVED CASE: {name}")
    for name in sorted(current_cases.keys() - baseline_cases.keys()):
        diffs.append(f"  NEW CASE: {name}")
    for name in sorted(baseline_cases.keys() & current_cases.keys()):
        base = {key: value for key, value in baseline_cases[name].items() if key != "name"}
        cur = {key: value for key, value in current_cases[name].items() if key != "name"}
        _append_value_diffs(name, base, cur, diffs)

    if diffs:
        print("DIFF vs baseline:")
        for d in diffs:
            print(d)
        sys.exit(1)
    else:
        print("OK — current metrics match baseline exactly.")


def _append_value_diffs(path: str, baseline: Any, current: Any, diffs: List[str]) -> None:
    """Append actionable differences for nested baseline values."""
    if isinstance(baseline, dict) and isinstance(current, dict):
        for key in sorted(baseline.keys() | current.keys()):
            child_path = f"{path}.{key}"
            if key not in baseline:
                diffs.append(f"  {child_path}: <missing> -> {current[key]!r}")
            elif key not in current:
                diffs.append(f"  {child_path}: {baseline[key]!r} -> <missing>")
            else:
                _append_value_diffs(child_path, baseline[key], current[key], diffs)
        return
    if baseline != current:
        diffs.append(f"  {path}: {baseline!r} -> {current!r}")


if __name__ == "__main__":
    main()
