"""Probe: how do artist-mode seed anchors distribute across intensity
neighborhoods (soft / mid / aggressive), feature off vs on?

Bands = library-wide arousal terciles (the bottom / middle / top third of ALL
music by arousal). Energy is library z-scored. We print the artist's catalog
distribution and the chosen medoids' distribution OFF (sound only) vs ON
(energy weight), to see whether the energy staircase distorts the band's
natural lean (e.g. Nirvana ~aggressive-heavy → does ON flatten it?).

Untracked research probe; run from the worktree:
    python -m scripts.research.energy_neighborhood_probe --artist Nirvana
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features.artifacts import load_artifact_bundle, set_sonic_variant_override
from src.playlist.artist_style import (
    ArtistStyleConfig,
    _artist_indices_in_bundle,
    cluster_artist_tracks,
)
from src.playlist.energy_loader import load_energy_matrix


def _bander(lo: float, hi: float):
    def band(z: float) -> str:
        if not np.isfinite(z):
            return "n/a"
        if z < lo:
            return "soft"
        if z > hi:
            return "aggressive"
        return "mid"
    return band


def _counts(zs, band) -> dict:
    out = {"aggressive": 0, "mid": 0, "soft": 0, "n/a": 0}
    for z in zs:
        out[band(z)] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="Nirvana")
    ap.add_argument("--energy-weight", type=float, default=5.0)
    ap.add_argument(
        "--num-seeds", type=int, default=12,
        help="Force this many sound-groups (=anchors) for a more granular view.",
    )
    ap.add_argument(
        "--artifact",
        default="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    args = ap.parse_args()

    set_sonic_variant_override("mert")
    bundle = load_artifact_bundle(Path(args.artifact))
    print(f"sonic space: variant=mert X_sonic.shape={bundle.X_sonic.shape}")
    sidecar = Path(args.artifact).parent / "energy" / "energy_sidecar.npz"
    energy = load_energy_matrix(
        list(bundle.track_ids), sidecar_path=str(sidecar), features=("arousal_p50",)
    )[:, 0]

    finite_all = energy[np.isfinite(energy)]
    lo = float(np.percentile(finite_all, 100.0 / 3))
    hi = float(np.percentile(finite_all, 200.0 / 3))
    band = _bander(lo, hi)
    print(f"library tercile cuts (z-arousal): soft < {lo:+.2f} < mid < {hi:+.2f} < aggressive")

    idx = _artist_indices_in_bundle(bundle, args.artist)
    n = len(idx)
    cat = _counts(energy[idx], band)
    print(f"\n{args.artist}: {n} tracks")
    print(
        f"  CATALOG lean: aggressive {cat['aggressive']} ({100*cat['aggressive']/n:.0f}%)  "
        f"mid {cat['mid']} ({100*cat['mid']/n:.0f}%)  "
        f"soft {cat['soft']} ({100*cat['soft']/n:.0f}%)  n/a {cat['n/a']}"
    )

    sweep = [(0.0, "OFF (sound only)"), (2.0, "ON (light, wt 2)"), (args.energy_weight, f"ON (strong, wt {args.energy_weight})")]
    for wt, label in sweep:
        cfg = ArtistStyleConfig(
            enabled=True, medoid_energy_weight=wt,
            cluster_k_min=args.num_seeds, cluster_k_max=args.num_seeds,
        )
        _clusters, medoids, _by, _X = cluster_artist_tracks(
            bundle=bundle, artist_name=args.artist, cfg=cfg, random_seed=0,
            energy_values=energy if wt > 0 else None,
        )
        mz = energy[medoids]
        c = _counts(mz, band)
        spread = ", ".join(f"{z:+.2f}({band(z)[0]})" for z in sorted(mz))
        print(f"\nseeds {label}: {len(medoids)} anchors")
        print(f"  by band: aggressive {c['aggressive']}  mid {c['mid']}  soft {c['soft']}  n/a {c['n/a']}")
        print(f"  anchor intensities: {spread}")


if __name__ == "__main__":
    main()
