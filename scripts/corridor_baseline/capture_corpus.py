"""12-cell corpus baseline -- topology fingerprints + Category F reporting checklist.

Corridor Phase 0a, Task 4. Runs the full 6-artist x {home, open} corpus (12 real,
faithful artist-mode generations via runner.run_cell) and captures a per-cell
topology fingerprint (Category D: segments, interior lengths, mini-pier/tail-DP/
edge-repair activity, solo/collab split, transition stats) plus a reporting-presence
checklist (Category F: which diagnostic log lines actually fired) into
docs/corridor_baseline/corpus_baseline.json.

Any cell whose generation raises is recorded with its error and the run continues;
the process exits nonzero at the end if any cell failed. Raw DEBUG logs land under
logs/corridor_baseline/ (gitignored) -- kept so later tasks / regex fixes can
re-extract without re-generating.

Corridor-scoped tooling: delete this module when the corridor contract closes
(see docs/corridor_baseline/README.md).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.corridor_baseline.patterns import F_PATTERNS, extract_fingerprint  # noqa: E402
from scripts.corridor_baseline.runner import CORPUS, DETENTS, OUT_DIR, run_cell  # noqa: E402

logger = logging.getLogger(__name__)


def parse_set_args(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set path=value`` CLI args into a ``deep_set``-ready dict.

    ``value`` is parsed as a YAML scalar (via ``yaml.safe_load``) so bools/floats/
    ints/strings all come through as the right Python type without a bespoke
    mini-parser -- ``--set foo.bar=0.90`` yields a float, ``--set foo.bar=true``
    yields a bool, ``--set foo.bar=corridor`` yields a str. Forwarded verbatim into
    ``runner.run_cell(set_paths=...)``, which applies each entry via ``deep_set``
    AFTER the policy layer runs (so mode presets cannot clobber it -- see
    runner.py's module docstring).
    """
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects path=value, got: {pair!r}")
        path, raw_value = pair.split("=", 1)
        out[path.strip()] = yaml.safe_load(raw_value)
    return out


def capture_cell(artist: str, detent: str, set_paths: dict[str, Any] | None = None) -> dict:
    run = run_cell(
        artist, detent, set_paths=set_paths,
        log_tag=f"corpus_{artist}_{detent}".replace(" ", "_"),
    )
    log_path = Path(run["log_path"])
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    run_stamped = dict(run)
    run_stamped["artist"] = artist
    run_stamped["detent"] = detent
    fp = extract_fingerprint(log_text, run_stamped)
    return fp


def print_summary_table(fingerprints: list[dict]) -> None:
    header = f"{'artist':<16} {'detent':<6} {'min_T':>7} {'below_floor':>11} {'distinct':>8} {'admitted':>8} {'wall_s':>7}"
    print(header)
    print("-" * len(header))
    for fp in fingerprints:
        min_t = fp.get("min_transition")
        min_t_str = f"{min_t:.3f}" if isinstance(min_t, (int, float)) else "n/a"
        print(
            f"{fp.get('artist', '?'):<16} {fp.get('detent', '?'):<6} "
            f"{min_t_str:>7} {str(fp.get('below_floor')):>11} "
            f"{str(fp.get('distinct_artists')):>8} {str(fp.get('admitted')):>8} "
            f"{str(fp.get('wall_s')):>7}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "corpus_baseline.json",
        help="Output path for the captured fingerprint JSON "
             "(default: docs/corridor_baseline/corpus_baseline.json, unchanged behavior). "
             "Use a different path to avoid overwriting the committed baseline.",
    )
    parser.add_argument(
        "--set",
        dest="set_paths",
        action="append",
        default=None,
        metavar="path=value",
        help="Repeatable post-policy config override, e.g. "
             "--set playlists.ds_pipeline.pier_bridge.pooling=corridor. Value is parsed "
             "as a YAML scalar (so 0.90/true/corridor come through as float/bool/str). "
             "Forwarded into runner.run_cell(set_paths=...) -- applied AFTER the policy "
             "layer, so mode presets cannot clobber it. Applied to every cell in this run.",
    )
    args = parser.parse_args()
    set_paths = parse_set_args(args.set_paths)

    # No basicConfig in scripts/ (test_no_basicconfig_in_src_scripts) -- attach one
    # console handler explicitly instead. run_cell attaches its own per-cell
    # FileHandler on top of this.
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(logging.INFO)

    t0 = time.time()
    fingerprints: list[dict] = []
    errors: dict[str, str] = {}

    for artist in CORPUS:
        for detent in DETENTS:
            cell_key = f"{artist}::{detent}"
            logger.info("=== capturing cell %s ===", cell_key)
            try:
                fp = capture_cell(artist, detent, set_paths=set_paths)
            except Exception as e:  # capture_cell itself should not normally raise
                # (run_cell already catches generation errors into run["err"]), but
                # guard the loop so one broken cell doesn't abort the other 11.
                logger.error("cell %s: harness error: %s: %s", cell_key, type(e).__name__, e)
                errors[cell_key] = f"{type(e).__name__}: {e}"
                fingerprints.append({
                    "artist": artist, "detent": detent, "err": f"{type(e).__name__}: {e}",
                })
                continue
            fingerprints.append(fp)
            if fp.get("err"):
                errors[cell_key] = fp["err"]
                logger.warning("cell %s: generation error: %s", cell_key, fp["err"])

    print_summary_table(fingerprints)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(CODE_ROOT), capture_output=True, text=True, check=True,
    ).stdout.strip()

    always_on = [name for name, spec in F_PATTERNS.items() if spec["conditional_on"] is None]
    always_on_never_seen = [
        name for name in always_on
        if not any(fp.get("reporting_presence", {}).get(name) for fp in fingerprints)
    ]

    summary = {
        "cells": fingerprints,
        "errors": errors,
        "generated_on": git_sha,
        "wall_s": round(time.time() - t0, 1),
        "always_on_patterns_never_seen": always_on_never_seen,
        "set_paths": set_paths,
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("done in %.1fs; %d/%d cells clean; wrote %s", time.time() - t0,
                len(fingerprints) - len(errors), len(fingerprints), out_path)
    if always_on_never_seen:
        logger.error("always-on F_PATTERNS entries never seen in ANY cell (harness bug -- "
                      "fix the regex, see task-4 brief Step 6): %s", always_on_never_seen)

    return 1 if (errors or always_on_never_seen) else 0


if __name__ == "__main__":
    sys.exit(main())
