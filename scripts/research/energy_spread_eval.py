"""A/B the energy-aware spread: is the seed mix REPRESENTATIVE of the band?

Primary gate (representativeness): total-variation distance between the seed
energy-band distribution (soft / mid / aggressive, defined as library-wide
arousal terciles) and the artist's own catalog distribution. Lower = the seeds
mirror the band. Energy-ON should be at least as representative as sound-only
(ON tv <= OFF tv on average) and meaningfully better for bands whose
sound-groups misrepresent their energy (e.g. Yo La Tengo, where the loud
timbres grab groups despite being a minority).

Secondary stat: raw arousal reach (max-min z) of the seed set, for context.

Sonic space is forced to MERT (the production authority); the 'beat3tower_32k'
dir name is historical. This measures selection-level representativeness only;
transition / diversity / wall-clock guardrails still require a full multi-pier
generation run via the gui_fidelity harness (see the playlist-testing skill).

Usage:
    python -m scripts.research.energy_spread_eval --artists "Nirvana" "Yo La Tengo" \
        --energy-weight 5.0 --num-seeds 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.features.artifacts import load_artifact_bundle, set_sonic_variant_override
from src.playlist.artist_style import (
    ArtistStyleConfig,
    _artist_indices_in_bundle,
    cluster_artist_tracks,
)
from src.playlist.energy_loader import load_energy_matrix


def pier_arousal_span(
    medoid_track_ids: Sequence[str], track_ids: Sequence[str], arousal: np.ndarray
) -> float:
    """z-arousal reach (max-min over finite) of a medoid set; 0.0 if <2 finite.

    Secondary context stat only — reach is NOT the gate (a band that is
    mostly-aggressive *should* have a narrow, aggressive seed set).
    """
    pos = {str(t): i for i, t in enumerate(track_ids)}
    vals = [arousal[pos[str(m)]] for m in medoid_track_ids if str(m) in pos]
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    return float(max(finite) - min(finite))


def band_fractions(energies: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fraction of finite energies in [aggressive, mid, soft] (z > hi / between / z < lo)."""
    e = np.asarray(energies, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    agg = float((e > hi).sum())
    soft = float((e < lo).sum())
    mid = float(e.size) - agg - soft
    return np.array([agg, mid, soft]) / e.size


def representativeness_tv(
    seed_energies: np.ndarray, catalog_energies: np.ndarray, lo: float, hi: float
) -> float:
    """Total-variation distance in [0,1] between the seed and catalog band mixes.

    0.0 = the seed mix exactly mirrors the band's aggressive/mid/soft proportions.
    """
    sf = band_fractions(seed_energies, lo, hi)
    cf = band_fractions(catalog_energies, lo, hi)
    if not np.all(np.isfinite(sf)) or not np.all(np.isfinite(cf)):
        return float("nan")
    return 0.5 * float(np.abs(sf - cf).sum())


def _seed_energies(
    bundle, artist: str, energy_weight: float, energy: np.ndarray, num_seeds: int
) -> Optional[np.ndarray]:
    kw = {}
    if num_seeds > 0:
        kw = {"cluster_k_min": num_seeds, "cluster_k_max": num_seeds}
    cfg = ArtistStyleConfig(enabled=True, medoid_energy_weight=energy_weight, **kw)
    try:
        _clusters, medoids, _by, _X = cluster_artist_tracks(
            bundle=bundle, artist_name=artist, cfg=cfg, random_seed=0,
            energy_values=energy if energy_weight > 0 else None,
        )
    except ValueError as exc:
        print(f"  {artist}: skipped ({exc})")
        return None
    return energy[medoids]


def _bandstr(energies: np.ndarray, lo: float, hi: float) -> str:
    f = band_fractions(energies, lo, hi)
    n = int(np.isfinite(np.asarray(energies, dtype=float)).sum())
    return f"{int(round(f[0]*n))}/{int(round(f[1]*n))}/{int(round(f[2]*n))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", nargs="+", required=True)
    ap.add_argument("--energy-weight", type=float, default=5.0)
    ap.add_argument(
        "--num-seeds", type=int, default=0,
        help="Force this many sound-groups (=anchors). 0 = natural production k.",
    )
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    ap.add_argument("--sonic-variant", default="mert")
    args = ap.parse_args()

    set_sonic_variant_override(args.sonic_variant)
    bundle = load_artifact_bundle(Path(args.artifact))
    print(f"sonic space: variant={args.sonic_variant} X_sonic.shape={bundle.X_sonic.shape}")
    sidecar = Path(args.artifact).parent / "energy" / "energy_sidecar.npz"
    energy = load_energy_matrix(
        list(bundle.track_ids), sidecar_path=str(sidecar), features=("arousal_p50",)
    )[:, 0]

    finite_all = energy[np.isfinite(energy)]
    lo = float(np.percentile(finite_all, 100.0 / 3))
    hi = float(np.percentile(finite_all, 200.0 / 3))
    print(f"bands (z-arousal): soft < {lo:+.2f} < mid < {hi:+.2f} < aggressive  "
          f"(agg/mid/soft counts shown)\n")

    print(f"{'artist':22s} {'catalog':>11s} {'off':>9s} {'on':>9s} "
          f"{'tvOff':>6s} {'tvOn':>6s} {'repr?':>6s}")
    tv_offs, tv_ons = [], []
    better = same = worse = 0
    for artist in args.artists:
        idx = _artist_indices_in_bundle(bundle, artist)
        if not idx:
            print(f"{artist:22s}  (not in library)")
            continue
        cat = energy[idx]
        off = _seed_energies(bundle, artist, 0.0, energy, args.num_seeds)
        on = _seed_energies(bundle, artist, args.energy_weight, energy, args.num_seeds)
        if off is None or on is None:
            continue
        tv_off = representativeness_tv(off, cat, lo, hi)
        tv_on = representativeness_tv(on, cat, lo, hi)
        tv_offs.append(tv_off)
        tv_ons.append(tv_on)
        verdict = "better" if tv_on < tv_off - 1e-9 else ("worse" if tv_on > tv_off + 1e-9 else "same")
        better += verdict == "better"
        same += verdict == "same"
        worse += verdict == "worse"
        print(f"{artist:22s} {_bandstr(cat, lo, hi):>11s} {_bandstr(off, lo, hi):>9s} "
              f"{_bandstr(on, lo, hi):>9s} {tv_off:6.3f} {tv_on:6.3f} {verdict:>6s}")

    if tv_offs:
        print(f"\nmean representativeness distance (lower=better):  "
              f"OFF {np.mean(tv_offs):.3f}   ON {np.mean(tv_ons):.3f}")
        print(f"per-artist: ON more representative {better}, same {same}, worse {worse} "
              f"(of {len(tv_offs)})")
        print("ACCEPTANCE: mean ON <= mean OFF, and no artist materially worse.")


if __name__ == "__main__":
    main()
