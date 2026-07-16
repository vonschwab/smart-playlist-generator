"""No-knob-goes-inert sweep (contract "Automated completeness net").

Resumable: one checkpoint JSON per (cell, field); re-running skips existing.
    python scripts/corridor_baseline/capture_knob_sweep.py                 # full sweep, both cells
    python scripts/corridor_baseline/capture_knob_sweep.py --only-field candidate_pool.similarity_floor
    python scripts/corridor_baseline/capture_knob_sweep.py --limit 5       # smoke
    python scripts/corridor_baseline/capture_knob_sweep.py --merge         # merge-only (no new runs)

For each field in the reference cell's flattened effective config, this
perturbs it (perturb.perturb_value), re-runs the SAME faithful generation
cell with that one field overridden (runner.run_cell(set_paths=...)), and
records whether the perturbation actually changed the resulting playlist
(status="changed"/"inert") or never made it through the config/mapping
plumbing at all (status="unmapped"/"skipped_type"/"override_failed"/
"did_not_resolve"/"error"). See docs/corridor_baseline (Task 5 brief) for
the full status taxonomy -- these distinctions are the deliverable, not an
implementation detail.

Checkpoints (NOT committed, logs/ is blanket-gitignored):
    logs/corridor_baseline/sweep/<cell_tag>/_reference.json
    logs/corridor_baseline/sweep/<cell_tag>/<sanitized_field>.json

Merged output (committed): docs/corridor_baseline/knob_sweep.json

Corridor-scoped tooling: delete this module when the corridor contract closes
(see docs/corridor_baseline/README.md).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.corridor_baseline.perturb import (  # noqa: E402
    SKIP,
    config_path_for,
    flatten_leaves,
    perturb_value,
)
from scripts.corridor_baseline.runner import (  # noqa: E402
    DETENTS,
    LOG_DIR,
    OUT_DIR,
    SWEEP_CELLS,
    build_cell_config,
    deep_get,
    run_cell,
)

logger = logging.getLogger(__name__)

SWEEP_DIR = LOG_DIR / "sweep"
# Reference blob leaves that are never mapped to a config path (documented in
# perturb.py: 11 unmapped = sonic_variant.*/embedding.*) -- drop before the
# per-field loop rather than churning out "unmapped" checkpoints for them.
_DROP_PREFIXES = ("sonic_variant.", "embedding.")


def sanitize(name: str) -> str:
    """Filesystem-safe token for a dotted blob path (used as checkpoint filename
    and log_tag suffix). Field names contain dots; keep the raw path INSIDE the
    JSON, never rely on the sanitized form for anything but a filename."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name)


def assert_no_checkpoint_collisions(fields: list[str]) -> None:
    """Fail fast if two distinct field paths sanitize to the same checkpoint
    filename -- silently overwriting each other's checkpoint would misreport
    one field's status as the other's and is invisible unless checked up
    front (Task 5's self-review flagged this as unverified; the sweep runs
    unattended for hours, so it must fail loud, before any generation runs)."""
    seen: dict[str, str] = {}
    for field in fields:
        token = sanitize(field)
        if token in seen and seen[token] != field:
            raise AssertionError(
                f"Checkpoint filename collision: {seen[token]!r} and {field!r} "
                f"both sanitize to {token!r}.json"
            )
        seen[token] = field


def cell_tag(artist: str, detent: str) -> str:
    return sanitize(f"{artist}_{detent}")


def _eq(a: Any, b: Any) -> bool:
    """Tolerance-free equality, NaN-safe (NaN != NaN is True in Python; for
    this sweep's "did it change at all" comparisons, NaN==NaN should read as
    unchanged, not as a spurious diff)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _n_position_diffs(a: list[str], b: list[str]) -> int:
    n = max(len(a), len(b))
    diffs = 0
    for i in range(n):
        ai = a[i] if i < len(a) else None
        bi = b[i] if i < len(b) else None
        if ai != bi:
            diffs += 1
    return diffs


def _record(
    field: str,
    config_path: str | None,
    baseline_value: Any,
    perturbed_value: Any,
    status: str,
    wall_s: float,
    *,
    jaccard: float | None = None,
    n_position_diffs: int | None = None,
    delta_min_T: float | None = None,
    delta_mean_T: float | None = None,
    note: str | None = None,
) -> dict:
    rec: dict[str, Any] = {
        "field": field,
        "config_path": config_path,
        "baseline_value": baseline_value,
        "perturbed_value": perturbed_value,
        "status": status,
        "jaccard": jaccard,
        "n_position_diffs": n_position_diffs,
        "delta_min_T": delta_min_T,
        "delta_mean_T": delta_mean_T,
        "wall_s": wall_s,
    }
    if note is not None:
        rec["note"] = note
    return rec


def process_field(
    artist: str, detent: str, tag: str, field: str, baseline_value: Any, ref: dict,
    log_level: int = logging.INFO,
) -> dict:
    """Perturb one field for one cell and fingerprint the result against the
    cell's reference run. Mirrors the task-5 brief's per-field algorithm
    exactly -- see module docstring / brief for the status taxonomy."""
    t0 = time.time()

    config_path = config_path_for(field)
    if config_path is None:
        return _record(field, None, baseline_value, None, "unmapped", round(time.time() - t0, 3))

    pv = perturb_value(field, baseline_value)
    if pv is SKIP:
        return _record(field, config_path, baseline_value, None, "skipped_type", round(time.time() - t0, 3))
    if _eq(pv, baseline_value):
        return _record(field, config_path, baseline_value, pv, "skipped_type", round(time.time() - t0, 3),
                        note="perturbed_equals_baseline")

    cfg = build_cell_config(DETENTS[detent], {config_path: pv})
    actual = deep_get(cfg, config_path)
    if not _eq(actual, pv):
        return _record(field, config_path, baseline_value, pv, "override_failed", round(time.time() - t0, 3),
                        note=f"deep_get(cfg, config_path) == {actual!r}, expected {pv!r}")

    run = run_cell(
        artist, detent, set_paths={config_path: pv}, log_level=log_level,
        log_tag=f"sweep_{tag}_{sanitize(field)}",
    )
    wall = run["wall"]

    if run["err"]:
        return _record(field, config_path, baseline_value, pv, "error", wall, note=run["err"])
    if run["effective"] is None:
        return _record(field, config_path, baseline_value, pv, "error", wall,
                        note="no DS-success effective blob captured (log parse miss)")

    perturbed_flat = flatten_leaves(run["effective"])
    perturbed_leaf = perturbed_flat.get(field)
    if _eq(perturbed_leaf, baseline_value):
        return _record(field, config_path, baseline_value, pv, "did_not_resolve", wall)

    ref_ids = ref["track_ids"]
    cur_ids = run["track_ids"]
    status = "inert" if cur_ids == ref_ids else "changed"
    ref_min_t, cur_min_t = ref.get("min_transition"), run.get("min_transition")
    ref_mean_t, cur_mean_t = ref.get("mean_transition"), run.get("mean_transition")
    delta_min_T = (cur_min_t - ref_min_t) if (cur_min_t is not None and ref_min_t is not None) else None
    delta_mean_T = (cur_mean_t - ref_mean_t) if (cur_mean_t is not None and ref_mean_t is not None) else None

    return _record(
        field, config_path, baseline_value, pv, status, wall,
        jaccard=_jaccard(ref_ids, cur_ids), n_position_diffs=_n_position_diffs(ref_ids, cur_ids),
        delta_min_T=delta_min_T, delta_mean_T=delta_mean_T,
    )


def sweep_cell(
    artist: str, detent: str, *, remaining: list[int] | None = None, only_field: str | None = None,
    log_level: int = logging.INFO,
) -> None:
    """Sweep one cell's fields. `remaining` is a mutable 1-element box holding
    the global (cross-cell) --limit budget: every field CONSIDERED (whether
    newly run or skipped because its checkpoint already exists) decrements it
    by one, and the cell is entered at all only if budget remains -- so
    `--limit 5` with no `--cell` filter touches only the first cell (matching
    "up to 5 perturbed generations on the first cell"), and a resumed run with
    the same limit and the same checkpoints on disk performs zero new
    generations (it re-consumes the identical budget against existing files)."""
    if remaining is not None and remaining[0] <= 0:
        logger.info("cell %s: skipped entirely (--limit budget exhausted)", cell_tag(artist, detent))
        return

    tag = cell_tag(artist, detent)
    cell_dir = SWEEP_DIR / tag
    cell_dir.mkdir(parents=True, exist_ok=True)

    ref_path = cell_dir / "_reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        logger.info("cell %s: cached reference (tracks=%d)", tag, len(ref["track_ids"]))
    else:
        logger.info("cell %s: running reference generation ...", tag)
        run = run_cell(artist, detent, log_level=log_level, log_tag=f"sweep_ref_{tag}")
        if run["err"]:
            logger.error("cell %s: reference run FAILED (%s) -- aborting cell", tag, run["err"])
            return
        if run["effective"] is None:
            logger.error("cell %s: reference run had no DS-success effective blob -- aborting cell", tag)
            return
        ref = {
            "track_ids": run["track_ids"],
            "effective_flat": flatten_leaves(run["effective"]),
            "min_transition": run["min_transition"],
            "mean_transition": run["mean_transition"],
            "wall": run["wall"],
        }
        ref_path.write_text(json.dumps(ref, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("cell %s: reference done wall=%.1fs tracks=%d", tag, run["wall"], len(run["track_ids"]))

    flat = ref["effective_flat"]
    fields = sorted(k for k in flat if not k.startswith(_DROP_PREFIXES))
    # Fail fast, before any generation runs, if two field paths would clobber
    # each other's checkpoint file.
    assert_no_checkpoint_collisions(fields)
    if only_field is not None:
        fields = [f for f in fields if f == only_field]
        if not fields:
            logger.warning("cell %s: --only-field %r not present in reference blob", tag, only_field)

    for field in fields:
        if remaining is not None and remaining[0] <= 0:
            logger.info("cell %s: --limit budget exhausted, stopping mid-cell", tag)
            break
        ckpt_path = cell_dir / f"{sanitize(field)}.json"
        if ckpt_path.exists():
            logger.info("cell %s field=%s: checkpoint exists -- skip", tag, field)
            if remaining is not None:
                remaining[0] -= 1
            continue
        logger.info("cell %s field=%s: perturbing ...", tag, field)
        t_field = time.time()
        try:
            record = process_field(artist, detent, tag, field, flat[field], ref, log_level)
        except Exception as exc:
            # A single pathological field must not kill the unattended,
            # multi-hour sweep. Record it as its own status and move on --
            # the raw field path stays inside the JSON (not just the
            # sanitized filename) so it's identifiable in the merged output.
            logger.exception("cell %s field=%s: process_field raised -- recording error checkpoint", tag, field)
            record = _record(
                field, None, flat[field], None, "error", round(time.time() - t_field, 3),
                note=f"{type(exc).__name__}: {exc}",
            )
        ckpt_path.write_text(json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("cell %s field=%s status=%s wall=%.1fs", tag, field, record["status"], record["wall_s"])
        if remaining is not None:
            remaining[0] -= 1


# ---- dead-outlet annotation (pure, extracted for a light unit test) ---------

def find_enabling_parent(leaf: str, reference_flat: dict[str, Any]) -> str | None:
    """Return the blob path of the `*_enabled` sibling flag that gates `leaf`,
    if that flag resolves to False in the reference blob -- else None.

    Family match: among all leaves ending in "_enabled" that are False in
    `reference_flat`, pick the one whose name shares the longest common
    prefix with `leaf` (e.g. "dj_genre_coverage_weight" -> "dj_bridging_enabled",
    not "genre_steering_enabled" -- shares "dj_" not "genre_"). Restricted to
    candidates sharing at least the leaf's first "_"-delimited token so
    unrelated flags never match.
    """
    disabled: dict[str, str] = {}
    for path, val in reference_flat.items():
        if val is not False:
            continue
        other_leaf = path.rsplit(".", 1)[-1]
        if other_leaf.endswith("_enabled"):
            disabled[other_leaf] = path

    if leaf in disabled:
        return None  # the field IS the flag itself, not one of its dependents

    prefix = leaf.split("_", 1)[0] + "_"
    candidates = [ol for ol in disabled if ol.startswith(prefix)]
    if not candidates:
        return None
    best = max(candidates, key=lambda ol: len(os.path.commonprefix([ol, leaf])))
    return disabled[best]


def merge_checkpoints() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    reference_by_cell: dict[str, dict] = {}

    if SWEEP_DIR.exists():
        for cell_dir in sorted(p for p in SWEEP_DIR.iterdir() if p.is_dir()):
            ref_path = cell_dir / "_reference.json"
            if ref_path.exists():
                reference_by_cell[cell_dir.name] = json.loads(ref_path.read_text(encoding="utf-8"))
            for ckpt in sorted(cell_dir.glob("*.json")):
                if ckpt.name == "_reference.json":
                    continue
                rec = dict(json.loads(ckpt.read_text(encoding="utf-8")))
                rec["cell"] = cell_dir.name
                records.append(rec)

    status_counts: dict[str, int] = {}
    for rec in records:
        status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1

    dead_outlets = []
    for rec in records:
        if rec["status"] != "inert":
            continue
        ref = reference_by_cell.get(rec["cell"], {})
        flat_ref = ref.get("effective_flat", {})
        leaf = rec["field"].rsplit(".", 1)[-1]
        dead_outlets.append({
            "cell": rec["cell"],
            "field": rec["field"],
            "enabling_parent_flag": find_enabling_parent(leaf, flat_ref),
        })

    summary = {
        "records": sorted(records, key=lambda r: (r["cell"], r["field"])),
        "status_counts": status_counts,
        "dead_outlets": sorted(dead_outlets, key=lambda d: (d["cell"], d["field"])),
        "cells_swept": sorted(reference_by_cell.keys()),
        "n_checkpoints": len(records),
    }
    out_path = OUT_DIR / "knob_sweep.json"
    out_path.write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("merged %d records %s -> %s", len(records), status_counts, out_path)
    return 0


def main() -> int:
    # No basicConfig in scripts/ (test_no_basicconfig_in_src_scripts) -- attach one
    # console handler explicitly instead. run_cell attaches its own per-cell
    # FileHandler on top of this.
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(logging.INFO)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                     help="Process only the first N fields total, across cells in SWEEP_CELLS order (smoke)")
    ap.add_argument("--only-field", default=None, help="Process only this single blob field path")
    ap.add_argument("--cell", default=None, help='Filter to one SWEEP_CELLS entry, e.g. "Bill Evans Trio:open"')
    ap.add_argument("--merge", action="store_true", help="Merge-only mode: skip sweeping, just merge checkpoints")
    args = ap.parse_args()

    if args.merge:
        return merge_checkpoints()

    cells = SWEEP_CELLS
    if args.cell:
        artist, _, detent = args.cell.partition(":")
        cells = [c for c in SWEEP_CELLS if c == (artist, detent)]
        if not cells:
            logger.error("--cell %r does not match any SWEEP_CELLS entry %s", args.cell, SWEEP_CELLS)
            return 2

    remaining = [args.limit] if args.limit is not None else None
    t0 = time.time()
    for artist, detent in cells:
        logger.info("=== sweeping cell %s / %s ===", artist, detent)
        sweep_cell(artist, detent, remaining=remaining, only_field=args.only_field)
    logger.info("sweep loop done in %.1fs", time.time() - t0)

    return merge_checkpoints()


if __name__ == "__main__":
    sys.exit(main())
