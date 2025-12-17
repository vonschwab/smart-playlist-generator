"""
Batch playlist evaluation lab.

For a set of seed tracks and modes, builds candidate pools, constructs
playlists (segment-aware if available), and writes per-run metrics to CSV.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.genre_similarity_lab import playlist_construction_lab as pcl

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"
DEFAULT_OUT_CSV = "experiments/genre_similarity_lab/artifacts/batch_eval.csv"

# Candidate pool presets (copied from playlist_candidate_pool_lab)
POOL_PRESETS = {
    "narrow": {
        "max_artist_fraction_final": 0.20,
        "similarity_floor": 0.35,
        "max_pool_size": 800,
        "target_artists": lambda L: max(math.ceil(L / 2), 12),
        "candidate_per_artist": lambda max_per: max(3, min(2 * max_per, 8)),
    },
    "dynamic": {
        "max_artist_fraction_final": 0.125,
        "similarity_floor": 0.30,
        "max_pool_size": 1200,
        "target_artists": lambda L: max(math.ceil(0.75 * L), 16),
        "candidate_per_artist": lambda max_per: max(3, min(2 * max_per, 6)),
    },
    "discover": {
        "max_artist_fraction_final": 0.05,
        "similarity_floor": 0.25,
        "max_pool_size": 2000,
        "target_artists": lambda L: min(L, 24),
        "candidate_per_artist": lambda max_per: max(2, min(2 * max_per, 4)),
    },
}


def _safe_str(val) -> str:
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return val.decode("cp1252", errors="ignore")
    return str(val)


def fit_hybrid(X_sonic, X_genre, n_comp_sonic, n_comp_genre, w_sonic, w_genre):
    hybrid, hybrid_norm, sonic_scaler, sonic_pca = pcl._build_hybrid(
        X_sonic, X_genre, n_comp_sonic, n_comp_genre, w_sonic, w_genre
    )
    return hybrid, hybrid_norm, sonic_scaler, sonic_pca


def segment_embeddings(art, sonic_scaler, sonic_pca, key_start, key_end):
    try:
        X_start = art[key_start]
        X_end = art[key_end]
    except KeyError:
        return None, None
    if X_start.shape[0] != art["X_sonic"].shape[0] or X_end.shape[0] != art["X_sonic"].shape[0]:
        return None, None
    E_start_raw = sonic_pca.transform(sonic_scaler.transform(X_start))
    E_end_raw = sonic_pca.transform(sonic_scaler.transform(X_end))
    E_start = E_start_raw / (np.linalg.norm(E_start_raw, axis=1, keepdims=True) + 1e-12)
    E_end = E_end_raw / (np.linalg.norm(E_end_raw, axis=1, keepdims=True) + 1e-12)
    return E_start, E_end


def build_candidate_pool(seed_idx, hybrid, seed_sims, artist_keys, mode: str, playlist_len: int):
    preset = POOL_PRESETS[mode]
    max_per_artist_final = math.ceil(playlist_len * preset["max_artist_fraction_final"])
    candidate_per_artist = preset["candidate_per_artist"](max_per_artist_final)
    seed_candidate_cap = candidate_per_artist + 2
    target_artists = preset["target_artists"](playlist_len)
    max_pool_size = preset["max_pool_size"]
    similarity_floor = preset["similarity_floor"]

    eligible = [i for i, sim in enumerate(seed_sims) if i != seed_idx and sim >= similarity_floor]
    groups: Dict[str, List[int]] = {}
    for idx in eligible:
        groups.setdefault(artist_keys[idx], []).append(idx)
    artist_rank = []
    for a, idxs in groups.items():
        best_sim = max(seed_sims[i] for i in idxs)
        artist_rank.append((a, best_sim, idxs))
    artist_rank.sort(key=lambda t: -t[1])

    pool_indices: List[int] = []
    pool_artists = set()
    for a, _, idxs in artist_rank:
        per_cap = seed_candidate_cap if a == artist_keys[seed_idx] else candidate_per_artist
        take = sorted(idxs, key=lambda i: -seed_sims[i])[:per_cap]
        for idx in take:
            if len(pool_indices) >= max_pool_size and len(pool_artists) >= target_artists:
                break
            pool_indices.append(idx)
        pool_artists.add(a)
        if len(pool_indices) >= max_pool_size and len(pool_artists) >= target_artists:
            break
    return np.array(pool_indices, dtype=int), groups


def playlist_metrics(seq: Sequence[int], artist_keys: Sequence[str], local_seg_list, seed_sim_list, hybrid_norm, seed_idx, min_gap, max_per_artist):
    artist_counter = {a: list(artist_keys[seq]).count(a) for a in set(artist_keys[seq])}
    max_artist_frac = max(artist_counter.values()) / len(seq)
    # Gini/effective artists
    counts = np.array(list(artist_counter.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    effective = float(np.exp(entropy))
    sorted_counts = np.sort(counts)
    n_art = len(sorted_counts)
    gini = ((np.sum((2 * np.arange(1, n_art + 1) - n_art - 1) * sorted_counts)) / (n_art * np.sum(sorted_counts) + 1e-12))

    # Violations
    adjacency = 0
    gap_viol = 0
    recent = []
    cap_viol = 0
    for idx in seq:
        a = artist_keys[idx]
        if recent and a == recent[-1]:
            adjacency += 1
        if min_gap > 0 and a in recent[-min_gap:]:
            gap_viol += 1
        recent.append(a)
    if max(counts) > max_per_artist:
        cap_viol = int(max(counts) - max_per_artist)

    # Seed sims stats
    seed_vals = np.array(seed_sim_list[1:], dtype=float)
    seed_mean = float(np.nanmean(seed_vals))
    seed_min = float(np.nanmin(seed_vals))
    seed_p10 = float(np.nanpercentile(seed_vals, 10))

    # Local segment stats
    loc = np.array(local_seg_list[1:], dtype=float)
    loc_min = float(np.nanmin(loc))
    loc_p10 = float(np.nanpercentile(loc, 10))
    loc_mean = float(np.nanmean(loc))
    loc_p90 = float(np.nanpercentile(loc, 90))

    # Drift
    def _drift(k):
        m = np.mean(hybrid_norm[seq[:k]], axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        return float(np.dot(m, hybrid_norm[seed_idx]))

    drift_k10 = _drift(min(10, len(seq)))
    drift_kL = _drift(len(seq))

    # Worst transition
    local_used_vals = loc
    worst_idx = int(np.nanargmin(local_used_vals)) + 1  # offset because loc starts at 1
    worst_pair = (seq[worst_idx - 1], seq[worst_idx])
    worst_val = float(local_used_vals[worst_idx - 1])

    return {
        "distinct_artists": len(artist_counter),
        "effective_artists": effective,
        "gini": gini,
        "max_artist_frac": max_artist_frac,
        "adjacency_violations": adjacency,
        "min_gap_violations": gap_viol,
        "cap_violations": cap_viol,
        "seed_sim_mean": seed_mean,
        "seed_sim_min": seed_min,
        "seed_sim_p10": seed_p10,
        "local_seg_min": loc_min,
        "local_seg_p10": loc_p10,
        "local_seg_mean": loc_mean,
        "local_seg_p90": loc_p90,
        "drift_k10": drift_k10,
        "drift_kL": drift_kL,
        "worst_transition_value": worst_val,
        "worst_transition_pair": worst_pair,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch playlist evaluation across seeds/modes.")
    parser.add_argument("--artifact-path", required=True, help="Path to data_matrices_step1.npz")
    parser.add_argument("--seed-track-ids", nargs="+", help="Explicit seed track_ids")
    parser.add_argument("--n-seeds", type=int, default=25, help="Number of seeds to sample if not provided explicitly.")
    parser.add_argument("--seed-sampling", choices=["random", "top_genre_dense"], default="random")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["narrow", "dynamic", "discover"])
    parser.add_argument("--playlist-len", type=int, help="Fallback playlist length for all modes.")
    parser.add_argument("--playlist-len-narrow", type=int, default=25)
    parser.add_argument("--playlist-len-dynamic", type=int, default=30)
    parser.add_argument("--playlist-len-discover", type=int, default=40)
    parser.add_argument("--transition-mode", choices=["hybrid", "segment_sonic"], default="segment_sonic")
    parser.add_argument("--transition-gamma", type=float, default=1.0)
    parser.add_argument("--transition-floor-mode", choices=["soft", "hard", "off"], default=None)
    parser.add_argument("--transition-floor", type=float, help="Override transition floor.")
    parser.add_argument("--transition-penalty-lambda", type=float, default=None)
    parser.add_argument("--alpha-schedule", choices=["constant", "arc"], help="Override alpha schedule (default mode).")
    parser.add_argument("--alpha-start", type=float, help="Override alpha start (default mode).")
    parser.add_argument("--alpha-mid", type=float, help="Override alpha mid (default mode).")
    parser.add_argument("--alpha-end", type=float, help="Override alpha end (default mode).")
    parser.add_argument("--arc-midpoint", type=float, help="Override arc midpoint fraction (default 0.55).")
    parser.add_argument("--out-csv", help="Output CSV (default: artifacts/batch_eval_<timestamp>.csv)")
    parser.add_argument("--out-dir", default="experiments/genre_similarity_lab/artifacts", help="Default output directory when out-csv not provided.")
    parser.add_argument("--out-summary-json", help="Optional summary JSON path.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip seed/mode rows already in out_csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.random_state)

    art = np.load(args.artifact_path, allow_pickle=True)
    X_sonic = art["X_sonic"]
    X_genre = art["X_genre_smoothed"] if "X_genre_smoothed" in art else art["X_genre"]
    track_ids = art["track_ids"]
    artist_names = art["artist_names"] if "artist_names" in art else art["track_artists"]
    track_titles = art["track_titles"]
    if "artist_keys_effective" in art:
        artist_keys = art["artist_keys_effective"]
    elif "artist_keys" in art:
        artist_keys = art["artist_keys"]
    else:
        artist_keys = np.array([""] * len(track_ids))

    # Seed selection
    if args.seed_track_ids:
        seeds = [s for s in args.seed_track_ids if s in set(track_ids)]
    else:
        all_ids = np.array(track_ids)
        idxs = np.random.choice(len(all_ids), size=min(args.n_seeds, len(all_ids)), replace=False)
        seeds = all_ids[idxs].tolist()

    # Output handling
    if args.out_csv:
        candidate = Path(args.out_csv)
        if candidate.suffix.lower() != ".csv":
            # treat as directory or filename without suffix
            base_dir = candidate if candidate.is_dir() else candidate.parent
            fname = candidate.name if candidate.is_dir() else candidate.name + ".csv"
            out_path = base_dir / fname
        else:
            out_path = candidate
    else:
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        modes_joined = "-".join(args.modes)
        out_path = Path(args.out_dir) / f"batch_eval_{ts}_{args.n_seeds}seeds_{modes_joined}.csv"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if args.skip_existing and out_path.exists():
        existing_df = pd.read_csv(out_path)

    rows = []
    for seed in seeds:
        if seed not in track_ids:
            logger.warning("Seed %s not found; skipping", seed)
            continue
        seed_idx = int(np.where(track_ids == seed)[0][0])
        for mode in args.modes:
            if existing_df is not None:
                if ((existing_df["seed_track_id"] == seed) & (existing_df["mode"] == mode)).any():
                    continue
            L = args.playlist_len or {
                "narrow": args.playlist_len_narrow,
                "dynamic": args.playlist_len_dynamic,
                "discover": args.playlist_len_discover,
            }[mode]

            # Hybrid embedding
            hybrid, hybrid_norm, sonic_scaler, sonic_pca = fit_hybrid(
                X_sonic, X_genre, n_comp_sonic=32, n_comp_genre=32, w_sonic=0.6, w_genre=0.4
            )
            E_start, E_end = (None, None)
            transition_mode_run = args.transition_mode
            if transition_mode_run == "segment_sonic":
                E_start, E_end = segment_embeddings(art, sonic_scaler, sonic_pca, "X_sonic_start", "X_sonic_end")
                if E_start is None or E_end is None:
                    logger.warning("Segment matrices missing; falling back to hybrid for transitions.")
                    transition_mode_run = "hybrid"

            seed_sims = hybrid_norm @ hybrid_norm[seed_idx]
            # Candidate pool
            pool_indices, groups = build_candidate_pool(seed_idx, hybrid, seed_sims, artist_keys, mode, L)

            mode_cfg = pcl.MODE[mode].copy()
            mode_cfg["beam"] = mode_cfg["beam_segment_override"] if transition_mode_run == "segment_sonic" else mode_cfg["beam"]
            beam_source = "segment_override" if transition_mode_run == "segment_sonic" else "mode_default"
            transition_floor_override = args.transition_floor if args.transition_floor is not None else None
            floor_mode_override = args.transition_floor_mode if args.transition_floor_mode is not None else None
            penalty_lambda_override = args.transition_penalty_lambda if args.transition_penalty_lambda is not None else None
            alpha_schedule_override = args.alpha_schedule if args.alpha_schedule is not None else None
            alpha_start_override = args.alpha_start if args.alpha_start is not None else None
            alpha_mid_override = args.alpha_mid if args.alpha_mid is not None else None
            alpha_end_override = args.alpha_end if args.alpha_end is not None else None
            arc_midpoint_override = args.arc_midpoint if args.arc_midpoint is not None else None

            # Playlist
            result = pcl.construct_playlist(
                seed_idx,
                pool_indices,
                seed_sims[pool_indices],
                artist_keys,
                hybrid_norm,
                E_start if transition_mode_run == "segment_sonic" else None,
                E_end if transition_mode_run == "segment_sonic" else None,
                args.transition_gamma,
                transition_mode_run,
                transition_floor_override if transition_mode_run == "segment_sonic" else None,
                floor_mode_override,
                penalty_lambda_override,
                L,
                mode_cfg,
                repeat_bonus=mode_cfg["repeat_bonus"],
                artist_target=min(L, math.ceil(L / mode_cfg.get("avg_tracks_per_artist_target", 1.0))),
                new_artist_penalty=mode_cfg.get("new_artist_penalty", 0.0),
                novelty_decay="linear",
                repeat_ramp_bonus=mode_cfg.get("repeat_ramp_bonus", 0.0),
                beam_source=beam_source,
                alpha_schedule=alpha_schedule_override,
                alpha_start=alpha_start_override,
                alpha_mid=alpha_mid_override,
                alpha_end=alpha_end_override,
                arc_midpoint=arc_midpoint_override,
            )
            (
                seq,
                used_counts,
                step_stats,
                seed_sim_list,
                local_sim_list,
                local_seg_list,
                local_hyb_list,
                chosen_below_floor,
                repeat_ramp_applied,
                floor_mode_effective,
                floor_value_effective,
                penalty_lambda_effective,
                transition_mode_effective,
                beam_used,
                beam_source,
                alpha_t_list,
                alpha_schedule_effective,
                alpha_start_effective,
                alpha_mid_effective,
                alpha_end_effective,
                arc_midpoint_effective,
            ) = result

            # Metrics
            metrics = playlist_metrics(
                seq,
                artist_keys,
                local_seg_list if transition_mode_effective == "segment_sonic" else local_sim_list,
                seed_sim_list,
                hybrid_norm,
                seed_idx,
                min_gap=mode_cfg["min_gap"],
                max_per_artist=math.ceil(L * mode_cfg["max_artist_frac"]),
            )
            fallback_total = sum(s.get("fallback_used", 0) for s in step_stats)
            floor_relaxed_steps = sum(s.get("floor_relaxed", 0) for s in step_stats)

            rows.append(
                {
                    "seed_track_id": seed,
                    "mode": mode,
                    "playlist_len": L,
                    "transition_mode": transition_mode_effective,
                    "gamma": args.transition_gamma,
                    "floor_mode_requested": args.transition_floor_mode,
                    "floor_value_requested": args.transition_floor,
                    "penalty_lambda_requested": args.transition_penalty_lambda,
                    "beam_requested": args.beam if hasattr(args, "beam") else None,
                    "alpha_schedule_requested": args.alpha_schedule,
                    "alpha_start_requested": args.alpha_start,
                    "alpha_mid_requested": args.alpha_mid,
                    "alpha_end_requested": args.alpha_end,
                    "arc_midpoint_requested": args.arc_midpoint,
                    "floor_mode_effective": floor_mode_effective,
                    "floor_value_effective": floor_value_effective,
                    "penalty_lambda_effective": penalty_lambda_effective,
                    "beam_used": beam_used,
                    "beam_source": beam_source,
                    "alpha_schedule_effective": alpha_schedule_effective,
                    "alpha_start_effective": alpha_start_effective,
                    "alpha_mid_effective": alpha_mid_effective,
                    "alpha_end_effective": alpha_end_effective,
                    "arc_midpoint_effective": arc_midpoint_effective,
                    "pool_size": int(len(pool_indices)),
                    "eligible_artists": len(groups),
                    "distinct_artists_in_pool": len(set(artist_keys[idx] for idx in pool_indices)),
                    "distinct_artists": metrics["distinct_artists"],
                    "effective_artists": metrics["effective_artists"],
                    "gini": metrics["gini"],
                    "max_artist_frac": metrics["max_artist_frac"],
                    "adjacency_violations": metrics["adjacency_violations"],
                    "min_gap_violations": metrics["min_gap_violations"],
                    "cap_violations": metrics["cap_violations"],
                    "fallback_count_total": fallback_total,
                    "floor_relaxed_steps": floor_relaxed_steps,
                    "chosen_below_floor": chosen_below_floor,
                    "seed_sim_mean": metrics["seed_sim_mean"],
                    "seed_sim_min": metrics["seed_sim_min"],
                    "seed_sim_p10": metrics["seed_sim_p10"],
                    "local_seg_min": metrics["local_seg_min"],
                    "local_seg_p10": metrics["local_seg_p10"],
                    "local_seg_mean": metrics["local_seg_mean"],
                    "local_seg_p90": metrics["local_seg_p90"],
                    "drift_k10": metrics["drift_k10"],
                    "drift_kL": metrics["drift_kL"],
                    "worst_transition_value": metrics["worst_transition_value"],
                    "worst_transition_pair": metrics["worst_transition_pair"],
                }
            )

    df = pd.DataFrame(rows)
    print(f"Writing batch CSV to: {out_path}")
    df.to_csv(out_path, index=False)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"CSV write failed or empty at {out_path}")
    print(f"Wrote {len(df)} rows to: {out_path} (size={out_path.stat().st_size} bytes)")

    # Per-mode aggregates
    def summary_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(series.mean()),
            "p50": float(series.quantile(0.50)),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
        }

    print(f"Rows in df: {len(df)}; modes: {df['mode'].value_counts().to_dict()}")
    for mode in args.modes:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        print(f"\n=== Mode {mode} summary ===")
        for col in ["distinct_artists", "effective_artists", "seed_sim_mean", "drift_kL", "local_seg_p10", "local_seg_min", "chosen_below_floor", "floor_relaxed_steps"]:
            stats = summary_stats(sub[col])
            print(f"{col}: {stats}")

    # Worst seeds by local_seg_min
    worst = df.sort_values("local_seg_min").head(10)
    print("\nWorst seeds by local_seg_min:")
    for _, row in worst.iterrows():
        print(
            f"{row['seed_track_id']} | mode={row['mode']} | local_seg_min={row['local_seg_min']:.3f} | pair={row['worst_transition_pair']}"
        )

    if args.out_summary_json:
        Path(args.out_summary_json).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.out_summary_json, orient="records", lines=False)


if __name__ == "__main__":
    main()
