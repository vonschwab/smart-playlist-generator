"""Corridor Phase 0a baseline runner — faithful artist-mode generation cells.

Reuses the production policy chain exactly as scripts/research/
slider_differentiation_eval.py does (never reimplement the slider->config
mapping — tests/support/gui_fidelity.py design rule). One new seam:
post-policy perturbation. _apply_mode_presets runs INSIDE
load_config_with_overrides (src/playlist_gui/worker.py:442), so a perturbation
merged as a pre-policy override would be clobbered for preset-controlled keys;
deep_set into the merged dict afterwards is the only reliable injection point.

Corridor-scoped tooling: delete this package when the corridor contract closes
(see docs/corridor_baseline/README.md).
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Tuple

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

CONFIG = CODE_ROOT / "config.yaml"
OUT_DIR = CODE_ROOT / "docs" / "corridor_baseline"
LOG_DIR = CODE_ROOT / "logs" / "corridor_baseline"

CORPUS = ["SADE", "Bill Evans Trio", "The Strokes", "Swirlies", "Aaliyah", "Alex G"]
DETENTS = {
    "home": {"sonic_mode": "strict", "genre_mode": "strict", "cohesion_mode": "dynamic", "pace_mode": "dynamic"},
    "open": {"sonic_mode": "dynamic", "genre_mode": "dynamic", "cohesion_mode": "dynamic", "pace_mode": "dynamic"},
    # Phase 2 Task 4: narrow/discover width calibration probe -- the GUI's
    # "range" dial detents (src/playlist_gui/policy.py::DIAL_TO_AXES) between
    # home/open. cohesion_mode/pace_mode stay dynamic (matching home/open):
    # the range dial only ever touches sonic_mode/genre_mode.
    "close": {"sonic_mode": "narrow", "genre_mode": "narrow", "cohesion_mode": "dynamic", "pace_mode": "dynamic"},
    "wander": {"sonic_mode": "discover", "genre_mode": "discover", "cohesion_mode": "dynamic", "pace_mode": "dynamic"},
}
SWEEP_CELLS = [("Bill Evans Trio", "open"), ("Swirlies", "home")]


def deep_set(d: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def deep_get(d: Any, path: str, default: Any = None) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def parse_ds_success(log_text: str) -> Tuple[Optional[dict], Optional[dict]]:
    """Extract (effective, metrics) from the last DS-success JSON line
    (emitted at src/playlist/ds_pipeline_runner.py:206)."""
    for line in reversed(log_text.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"pipeline": "ds"' in line:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            return payload.get("effective"), payload.get("metrics")
    return None, None


def build_cell_config(axes: dict[str, str], set_paths: dict[str, Any] | None = None) -> dict:
    from src.playlist_gui.ui_state import UIStateModel
    from src.playlist_gui.policy import derive_runtime_config, merge_overrides
    from src.playlist_gui.worker import load_config_with_overrides

    ui = replace(UIStateModel(mode="seeds"), artist_spacing="strong", **axes)
    decisions = derive_runtime_config(ui, seed_artist_keys=None)
    ov = merge_overrides({}, decisions.overrides)
    merged = load_config_with_overrides(str(CONFIG), ov)
    for p, v in (set_paths or {}).items():
        deep_set(merged, p, v)  # post-policy: mode presets cannot clobber
    return merged


# ---- faithful generation via the policy layer -------------------------------
# Lazy-import cache, mirroring scripts/research/slider_differentiation_eval.py:127-147
# (_imp()) so this module can be exec'd standalone (importlib.util loading in the
# unit test) without eagerly importing the full production stack.
_IMP: dict[str, Any] = {}


def _imp() -> dict[str, Any]:
    if not _IMP:
        from src.local_library_client import LocalLibraryClient
        from src.metadata_client import MetadataClient
        from src.playlist_generator import PlaylistGenerator
        from src.track_matcher import TrackMatcher
        _IMP.update(
            LocalLibraryClient=LocalLibraryClient, PlaylistGenerator=PlaylistGenerator,
            TrackMatcher=TrackMatcher, MetadataClient=MetadataClient,
        )
    return _IMP


def build_generator(merged_cfg: dict):
    """Copy-adapted from slider_differentiation_eval.py:174-212 (build_generator
    + MC). DB/artifact paths come from this satellite's config.yaml, already
    absolute — no MAIN_OV-style path override is applied here."""
    I = _imp()

    class MC:
        def __init__(s, c):
            s.config = c
            s.config_path = str(CONFIG)

        def get(s, section, key=None, default=None):
            if section not in s.config:
                return default
            return s.config.get(section, default) if key is None else s.config[section].get(key, default)

        @property
        def library_database_path(s):
            return s.config["library"]["database_path"]

        @property
        def library_music_directory(s):
            return s.config["library"].get("music_directory", "E:/MUSIC")

        @property
        def lastfm_api_key(s):
            import os
            return os.getenv("LASTFM_API_KEY") or s.config.get("lastfm", {}).get("api_key", "")

        @property
        def lastfm_username(s):
            import os
            return os.getenv("LASTFM_USERNAME") or s.config.get("lastfm", {}).get("username", "")

    mc = MC(merged_cfg)
    lib = I["LocalLibraryClient"](db_path=mc.library_database_path)
    matcher = I["TrackMatcher"](lib, library_id=None, db_path=mc.library_database_path)
    try:
        meta = I["MetadataClient"](mc.library_database_path)
    except Exception:
        meta = None
    return I["PlaylistGenerator"](lib, mc, lastfm_client=None, track_matcher=matcher, metadata_client=meta)


def run_cell(
    artist: str,
    detent: str,
    *,
    set_paths: dict | None = None,
    length: int = 30,
    log_level: int = logging.DEBUG,
    log_tag: str = "",
) -> dict:
    """Run one faithful artist-mode generation cell, capturing its log to a
    unique file and parsing the DS-success JSON line (log-capture pattern
    copy-adapted from _run_and_parse, slider_differentiation_eval.py:215-274)."""
    axes = DETENTS[detent]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{time.time_ns()}_{log_tag or 'cell'}_{uuid.uuid4().hex[:6]}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(fh)
    root.setLevel(log_level)

    t0 = time.time()
    err: str | None = None
    track_ids: list[str] = []
    try:
        try:
            merged_cfg = build_cell_config(axes, set_paths)
            g = build_generator(merged_cfg)
            res = g.create_playlist_for_artist(
                artist,
                track_count=length,
                dynamic=(axes["cohesion_mode"] == "dynamic"),
                cohesion_mode_override=axes["cohesion_mode"],
            )
            tracks = res.get("tracks", []) if isinstance(res, dict) else []
            track_ids = [str(t.get("rating_key") or t.get("id") or t.get("track_id") or "") for t in tracks]
            track_ids = [t for t in track_ids if t]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    finally:
        # Not try/finally-protected previously: an uncaught BaseException (e.g.
        # KeyboardInterrupt) mid-generation would skip this cleanup and leak a DEBUG
        # FileHandler on the root logger for every subsequent run_cell() call in a
        # loop (capture_corpus.py runs 12 in-process). Guarantee cleanup runs even
        # when the inner `except Exception` doesn't catch what was raised.
        root.removeHandler(fh)
        fh.close()
        root.setLevel(prev_level)
    wall = round(time.time() - t0, 1)

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    effective, run_metrics = parse_ds_success(log_text)

    def grp(pat, cast=str, default=None):
        m = re.search(pat, log_text)
        return cast(m.group(1)) if m else default

    return {
        "track_ids": track_ids,
        "err": err,
        "wall": wall,
        "effective": effective,
        "run_metrics": run_metrics,
        "log_path": str(log_path),
        "min_transition": grp(r'min_transition[="\s:]+([0-9.]+)', float),
        "mean_transition": grp(r'mean_transition[="\s:]+([0-9.]+)', float),
        "admitted": grp(r"Candidate pool:.*?admitted=(\d+)", int),
        "below_floor": (run_metrics or {}).get("below_floor"),
        "distinct_artists": (run_metrics or {}).get("distinct_artists"),
    }
