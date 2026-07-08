"""Beam-weight response sweep: does pushing the beam tag term harder move a close-genre
artist (Real Estate/jangle) toward the tag, and at what worst-edge cost? Also gives the
pier+pool-only number (beam weight 0.0). Read-only DB; temp configs + logs to out dir.

Usage: python scripts/research/tag_steering_beam_sweep.py [out_dir]
"""
import sys
import os
import copy
import logging
import re
import sqlite3
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import numpy as np
import yaml

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "/tmp"), "beam_sweep")
os.makedirs(OUT, exist_ok=True)
DB = "data/metadata.db"
ART = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"

CASES = [
    ("Brian Eno", ["ambient", "drone", "dark ambient", "space ambient"]),
    ("Real Estate", ["jangle pop"]),
]
WEIGHTS = [0.0, 0.15, 0.5, 1.0]

from src.playlist.tag_steering import (
    resolve_tag_sonic_prototype_rows, sonic_prototype_from_rows, sonic_global_mean)
from support.gui_fidelity import resolved_artifact_path
from src.features.artifacts import load_artifact_bundle
resolved_artifact_path()
_bundle = load_artifact_bundle(ART)
_xs = np.asarray(_bundle.X_sonic, dtype=np.float64)
_xsn = _xs / (np.linalg.norm(_xs, axis=1, keepdims=True) + 1e-12)
_gm = sonic_global_mean(_xs)
_t2r = {str(t): i for i, t in enumerate(_bundle.track_ids)}
_con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

def aff_vec(tags, artist):
    rows, n, _ = resolve_tag_sonic_prototype_rows(tags, metadata_db_path=DB,
        track_id_to_row=_t2r, exclude_artist=artist, min_support=25)
    proto, coh, _ = sonic_prototype_from_rows(_xs, rows, global_mean=_gm)
    return (_xsn - _gm) @ proto

def track_row(a, t):
    c = _con.cursor()
    c.execute("SELECT track_id FROM tracks WHERE artist=? AND title=?", (a, t))
    r = c.fetchone()
    if not r:
        c.execute("SELECT track_id FROM tracks WHERE artist=? AND title LIKE ?", (a, (t or "")[:12] + "%"))
        r = c.fetchone()
    return _t2r.get(str(r[0])) if r else None

with open("config.yaml") as f:
    BASE = yaml.safe_load(f)

def run(artist, tags, weight, label):
    cfg = copy.deepcopy(BASE)
    pb = cfg.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    pb["tag_steering_tags"] = list(tags)
    pb["tag_steering_prototype_min_support"] = 25
    pb["tag_steering_sonic_beam_weight"] = float(weight)
    cfgp = os.path.join(OUT, f"cfg_{label}.yaml")
    with open(cfgp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    logp = os.path.join(OUT, f"log_{label}.txt")
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(logp, "w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    from main_app import PlaylistApp
    app = PlaylistApp(config_path=cfgp)
    res = app.generator.create_playlist_for_artist(artist, 30, popular_seeds_mode="off")
    tracks = (res or {}).get("tracks") or []
    ordered = [(str(t.get("artist", "")), str(t.get("title") or t.get("name", ""))) for t in tracks if isinstance(t, dict)]
    log = open(logp, encoding="utf-8").read()
    m = re.search(r"min_transition=([0-9.]+)\s+mean_transition=([0-9.]+)", log)
    return ordered, (float(m.group(1)) if m else float("nan")), (float(m.group(2)) if m else float("nan"))

print(f"out: {OUT}\n")
for artist, tags in CASES:
    av = aff_vec(tags, artist)
    print("=" * 70)
    print(f"{artist} + {tags}")
    for w in WEIGHTS:
        label = f"{artist.replace(' ','')}_{tags[0].split()[0]}_w{str(w).replace('.','')}"
        ordered, min_t, mean_t = run(artist, tags, w, label)
        affs = [float(av[track_row(a, t)]) for a, t in ordered if track_row(a, t) is not None]
        affs = np.array(affs) if affs else np.array([0.0])
        print(f"  beam_w={w:<4}: on-tag aff mean={affs.mean():+.3f} "
              f"[p10={np.percentile(affs,10):+.3f} p50={np.percentile(affs,50):+.3f}] "
              f"min_T={min_t:.3f} mean_T={mean_t:.3f}  ({len(affs)} matched)")
print("\nDONE.")
