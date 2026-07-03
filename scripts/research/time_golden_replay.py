"""Time (and optionally profile) a golden-fixture replay.

The timing gate and baseline hotspot ranking for the lossless-speedup work
(docs/superpowers/plans/2026-07-03-lossless-generation-speedup.md). Replays a
frozen golden fixture through generate_playlist_ds and asserts the output is
still identical, so timing is only ever reported for a bit-identical run.

Usage:
    python scripts/research/time_golden_replay.py --fixture herbie
    python scripts/research/time_golden_replay.py --fixture herbie --profile
"""
import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from tests.support.lossless_golden import load_golden, replay_golden  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="label, e.g. herbie")
    ap.add_argument("--profile", action="store_true", help="cProfile the replay")
    ap.add_argument("--sort", default="cumulative", help="pstats sort key")
    ap.add_argument("--rows", type=int, default=40, help="profile rows to print")
    args = ap.parse_args()

    path = os.path.join("tests", "fixtures", "lossless_speedup", f"{args.fixture}.json")
    golden = load_golden(path)

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        result = replay_golden(golden)
        prof.disable()
        pstats.Stats(prof).sort_stats(args.sort).print_stats(args.rows)
    else:
        t0 = time.perf_counter()
        result = replay_golden(golden)
        dt = time.perf_counter() - t0
        print(f"[{args.fixture}] replay wall-clock: {dt:.1f}s  tracks={len(result.track_ids)}")

    assert list(result.track_ids) == list(golden["track_ids"]), (
        "replay diverged from golden — this run is NOT bit-identical"
    )


if __name__ == "__main__":
    main()
