"""Assemble docs/corridor_baseline/feature_baseline.json from the three
capture outputs (transforms_summary.json, corpus_baseline.json,
knob_sweep.json) plus provenance metadata (git commits, artifact hash,
DB track count, corpus/detents/sweep_cells constants, perturbation rules).

Usage:
    python scripts/corridor_baseline/assemble_baseline.py

Every input path and the output path can be overridden (used by tests and by
anyone re-running against a non-default capture location):
    python scripts/corridor_baseline/assemble_baseline.py \\
        --transforms path/to/transforms_summary.json \\
        --corpus path/to/corpus_baseline.json \\
        --knob-sweep path/to/knob_sweep.json \\
        --out path/to/feature_baseline.json \\
        --config path/to/config.yaml

Fails loudly (FileNotFoundError naming the missing file) if any input JSON
is absent -- no partial baseline is ever written.

Corridor-scoped tooling: delete this module when the corridor contract
closes (see docs/corridor_baseline/README.md).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# runner.py only does its (heavier) production imports lazily inside
# functions -- these module-level constants are cheap stdlib-only reads, so
# importing them here does not pull in the engine (reuse-first: don't
# hand-copy CORPUS/DETENTS/SWEEP_CELLS, they must stay one source of truth).
from scripts.corridor_baseline.runner import CONFIG, CORPUS, DETENTS, OUT_DIR, SWEEP_CELLS  # noqa: E402

DEFAULT_TRANSFORMS = OUT_DIR / "transforms_summary.json"
DEFAULT_CORPUS = OUT_DIR / "corpus_baseline.json"
DEFAULT_KNOB_SWEEP = OUT_DIR / "knob_sweep.json"
DEFAULT_OUT = OUT_DIR / "feature_baseline.json"
PERTURB_PY = Path(__file__).resolve().parent / "perturb.py"

_HASH_CHUNK = 1 << 20  # 1 MiB streaming chunks -- the artifact is ~507MB; never load whole file


def to_posix(path: str | Path) -> str:
    """POSIX-separator string for any path, regardless of host OS. A Task 3
    review flagged Windows backslashes baked into transforms_summary.json as
    a portability wart -- every path THIS module newly emits uses forward
    slashes (the embedded transforms/corpus JSON is passed through verbatim,
    not rewritten)."""
    return Path(path).as_posix()


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK) -> str:
    """Stream-hash a file without loading it into memory (the sonic artifact
    is ~507MB; this takes seconds, not minutes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    """Load a JSON capture input, failing loudly (naming the file) if
    missing -- contract requirement: no partial baseline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing corridor-baseline input: {path} -- run its capture script first "
            f"(see docs/corridor_baseline/README.md)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_perturbation_docstring(perturb_py: Path = PERTURB_PY) -> str:
    """The perturbation-rules docstring from perturb.py, read via `ast`
    (parse, don't import) so this module never gains a dependency on the
    sweep engine -- it only needs the text."""
    if not perturb_py.exists():
        raise FileNotFoundError(f"Missing {perturb_py} -- cannot read perturbation_rules docstring.")
    tree = ast.parse(perturb_py.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree)
    if doc is None:
        raise ValueError(f"{perturb_py} has no module docstring -- perturbation_rules would be empty.")
    return doc


def compute_artifact_info(artifact_path: str | Path) -> dict:
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing sonic artifact: {p} -- check playlists.ds_pipeline.artifact_path in config.yaml."
        )
    return {"path": to_posix(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}


def compute_db_info(db_path: str | Path) -> dict:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing library DB: {p} -- check library.database_path in config.yaml.")
    conn = sqlite3.connect(f"file:{to_posix(p)}?mode=ro", uri=True)
    try:
        track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    finally:
        conn.close()
    return {"path": to_posix(p), "track_count": track_count}


def git_output(*args: str, cwd: Path = CODE_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True,
    ).stdout.strip()


def assemble_meta(
    *,
    captured_on_commit: str,
    branch_tip: str,
    captured_date: str,
    artifact_info: dict,
    db_info: dict,
    corpus: list,
    detents: list,
    sweep_cells: list,
    perturbation_rules: str,
) -> dict:
    """Pure assembly of the meta block -- no I/O, fully unit-testable."""
    return {
        "captured_on_commit": captured_on_commit,
        "branch_tip": branch_tip,
        "captured_date": captured_date,
        "artifact": artifact_info,
        "db": db_info,
        "corpus": list(corpus),
        "detents": list(detents),
        "sweep_cells": [list(cell) for cell in sweep_cells],
        "perturbation_rules": perturbation_rules,
    }


def assemble_baseline(*, meta: dict, transforms: Any, corpus: Any, knob_sweep: Any) -> dict:
    """Pure assembly of the top-level baseline document -- no I/O."""
    return {"meta": meta, "transforms": transforms, "corpus": corpus, "knob_sweep": knob_sweep}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS, help="transforms_summary.json path")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus_baseline.json path")
    ap.add_argument("--knob-sweep", type=Path, default=DEFAULT_KNOB_SWEEP, help="knob_sweep.json path")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output feature_baseline.json path")
    ap.add_argument("--config", type=Path, default=CONFIG,
                     help="config.yaml to source playlists.ds_pipeline.artifact_path / library.database_path from")
    ap.add_argument("--captured-date", default=None,
                     help="YYYY-MM-DD; default: today (local date, at run time)")
    args = ap.parse_args()

    # Fail loud, up front, before any hashing/git work: no partial baseline.
    transforms = read_json(args.transforms)
    corpus_baseline = read_json(args.corpus)
    knob_sweep = read_json(args.knob_sweep)

    if not args.config.exists():
        raise FileNotFoundError(f"Missing config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Plain config.yaml, not the policy-merged config runner.build_cell_config
    # produces: load_config_with_overrides only deep-merges empty overrides and
    # applies mode presets (playlists.* gate/weight fields), neither of which
    # ever touches artifact_path/database_path -- so the value is identical
    # either way, and reading it directly avoids importing the engine stack
    # for two path strings (reuse-first: minimal dependency, not minimal
    # correctness).
    artifact_path = cfg["playlists"]["ds_pipeline"]["artifact_path"]
    db_path = cfg["library"]["database_path"]

    captured_on_commit = git_output("merge-base", "HEAD", "origin/master")
    branch_tip = git_output("rev-parse", "HEAD")
    captured_date = args.captured_date or time.strftime("%Y-%m-%d")

    meta = assemble_meta(
        captured_on_commit=captured_on_commit,
        branch_tip=branch_tip,
        captured_date=captured_date,
        artifact_info=compute_artifact_info(artifact_path),
        db_info=compute_db_info(db_path),
        corpus=CORPUS,
        detents=list(DETENTS),
        sweep_cells=SWEEP_CELLS,
        perturbation_rules=read_perturbation_docstring(),
    )

    baseline = assemble_baseline(meta=meta, transforms=transforms, corpus=corpus_baseline, knob_sweep=knob_sweep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
