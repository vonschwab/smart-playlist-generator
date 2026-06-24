"""A/B the energy-aware spread: pier arousal-span, energy-off vs energy-on.

Measures the spec's primary "spread" metric (z-arousal span across the pier set)
for a panel of artists. Guardrails (transition T, diversity, wall-clock) require a
full multi-pier generation run via the gui_fidelity harness — see the
playlist-testing skill; this script only measures the selection-level spread.

Usage:
    python -m scripts.research.energy_spread_eval --artists "Nirvana" "Slowdive" \
        --energy-weight 5.0 --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.playlist.artist_style import ArtistStyleConfig, cluster_artist_tracks
from src.playlist.energy_loader import load_energy_matrix


def pier_arousal_span(
    medoid_track_ids: Sequence[str], track_ids: Sequence[str], arousal: np.ndarray
) -> float:
    """z-arousal span (max-min over finite) of a medoid set; 0.0 if <2 finite."""
    pos = {str(t): i for i, t in enumerate(track_ids)}
    vals = [arousal[pos[str(m)]] for m in medoid_track_ids if str(m) in pos]
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    return float(max(finite) - min(finite))


def _run(bundle, artist: str, energy_weight: float, energy: np.ndarray) -> float:
    cfg = ArtistStyleConfig(enabled=True, medoid_energy_weight=energy_weight)
    try:
        _clusters, medoids, _by_cluster, _X = cluster_artist_tracks(
            bundle=bundle, artist_name=artist, cfg=cfg, random_seed=0,
            energy_values=energy if energy_weight > 0 else None,
        )
    except ValueError as exc:
        print(f"  {artist}: skipped ({exc})")
        return float("nan")
    ids = [str(bundle.track_ids[m]) for m in medoids]
    return pier_arousal_span(ids, [str(t) for t in bundle.track_ids], energy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", nargs="+", required=True)
    ap.add_argument("--energy-weight", type=float, default=5.0)
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    args = ap.parse_args()

    bundle = load_artifact_bundle(Path(args.artifact))
    sidecar = Path(args.artifact).parent / "energy" / "energy_sidecar.npz"
    energy = load_energy_matrix(
        list(bundle.track_ids), sidecar_path=str(sidecar), features=("arousal_p50",)
    )[:, 0]

    deltas = []
    print(f"{'artist':30s} {'off':>8s} {'on':>8s} {'delta':>8s}")
    for artist in args.artists:
        off = _run(bundle, artist, 0.0, energy)
        on = _run(bundle, artist, args.energy_weight, energy)
        if np.isfinite(off) and np.isfinite(on):
            deltas.append(on - off)
            print(f"{artist:30s} {off:8.3f} {on:8.3f} {on - off:+8.3f}")
        else:
            print(f"{artist:30s} {'skip':>8s} {'skip':>8s} {'—':>8s}")

    if deltas:
        print(f"\nmean span delta (on - off): {np.mean(deltas):+.3f}  (n={len(deltas)})")
        print("ACCEPTANCE: mean delta should be > 0 (piers spread wider in energy).")


if __name__ == "__main__":
    main()
