"""Tag-steering sonic-prototype end-to-end eval: sonic-OFF vs sonic-ON.

Runs REAL create_playlist_for_artist generations through the production config, toggling the
sonic prototype via tag_steering_prototype_min_support (huge => OFF/genre-dense-only, 25 => ON).
Scores each realized playlist by its mean affinity to the tag's (centered) sonic prototype, and
greps the tag-steering log lines. Read-only on the DB; writes only temp configs + logs to scratchpad.

Usage: python scripts/research/tag_steering_sonic_eval.py [out_dir]
"""
import sys
import os
import copy
import logging
import re
import sqlite3
sys.path.insert(0, ".")
import numpy as np
import yaml

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("TEMP", "/tmp"), "tag_steer_eval")
os.makedirs(OUT, exist_ok=True)

DB = "data/metadata.db"
ART = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"

# arms: (artist, tags, sonic_on)
ARMS = [
    ("Brian Eno", ["ambient", "drone", "dark ambient", "space ambient"], False),
    ("Brian Eno", ["ambient", "drone", "dark ambient", "space ambient"], True),
    ("Real Estate", ["jangle pop"], False),
    ("Real Estate", ["jangle pop"], True),
]

# ---- prototype (centered, for a consistent discrimination metric) ----
from src.playlist.tag_steering import (
    resolve_tag_sonic_prototype_rows, sonic_prototype_from_rows, sonic_global_mean,
)
import sys as _s
_s.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path
from src.features.artifacts import load_artifact_bundle
resolved_artifact_path()
_bundle = load_artifact_bundle(ART)
_xs = np.asarray(_bundle.X_sonic, dtype=np.float64)
_xsn = _xs / (np.linalg.norm(_xs, axis=1, keepdims=True) + 1e-12)
_gm = sonic_global_mean(_xs)
_t2r = {str(t): i for i, t in enumerate(_bundle.track_ids)}

def prototype_aff_vec(tags, exclude_artist):
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        tags, metadata_db_path=DB, track_id_to_row=_t2r,
        exclude_artist=exclude_artist, min_support=25)
    if rows is None:
        return None, 0
    proto, coh, _ = sonic_prototype_from_rows(_xs, rows, global_mean=_gm)
    return ((_xsn - _gm) @ proto), n

# ---- DB (artist,title) -> track_id ----
_con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
def track_row(artist, title):
    cur = _con.cursor()
    cur.execute("SELECT track_id FROM tracks WHERE artist=? AND title=?", (artist, title))
    r = cur.fetchone()
    if not r:
        cur.execute("SELECT track_id FROM tracks WHERE artist=? AND title LIKE ?",
                    (artist, (title or "")[:12] + "%"))
        r = cur.fetchone()
    return _t2r.get(str(r[0])) if r else None

# ---- config + run ----
with open("config.yaml", "r", encoding="utf-8") as f:
    BASE = yaml.safe_load(f)

def run(artist, tags, sonic_on, label):
    cfg = copy.deepcopy(BASE)
    pb = cfg.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    pb["tag_steering_tags"] = list(tags)
    pb["tag_steering_prototype_min_support"] = 25 if sonic_on else 10_000_000
    cfgp = os.path.join(OUT, f"cfg_{label}.yaml")
    with open(cfgp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    logp = os.path.join(OUT, f"log_{label}.txt")
    _root = logging.getLogger()
    for h in list(_root.handlers):
        _root.removeHandler(h)
    _root.setLevel(logging.INFO)
    _fh = logging.FileHandler(logp, "w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _root.addHandler(_fh)
    from main_app import PlaylistApp
    app = PlaylistApp(config_path=cfgp)
    res = app.generator.create_playlist_for_artist(artist, 30, popular_seeds_mode="off")
    tracks = (res or {}).get("tracks") or []
    ordered = [(str(t.get("artist", "")), str(t.get("title") or t.get("name", ""))) for t in tracks
               if isinstance(t, dict)]
    with open(logp, encoding="utf-8") as f:
        log = f.read().splitlines()
    steer = [l for l in log if re.search(r"sonic prototype|sonic pool|pier allocation|pool affinity|Tag steering piers|beam term", l)]
    m = re.search(r"min_transition=([0-9.]+)\s+mean_transition=([0-9.]+)", "\n".join(log))
    min_t = float(m.group(1)) if m else float("nan")
    mean_t = float(m.group(2)) if m else float("nan")
    return ordered, steer, min_t, mean_t

print(f"out dir: {OUT}\n")
summary = {}
for artist, tags, sonic_on in ARMS:
    label = f"{artist.replace(' ','')}_{'_'.join(tags[:1])}_{'ON' if sonic_on else 'OFF'}"
    aff_vec, support = prototype_aff_vec(tags, artist)
    ordered, steer, min_t, mean_t = run(artist, tags, sonic_on, label)
    affs = []
    for a, ti in ordered:
        r = track_row(a, ti)
        if r is not None and aff_vec is not None:
            affs.append(float(aff_vec[r]))
    affs = np.array(affs) if affs else np.array([0.0])
    summary[label] = (ordered, affs, steer, min_t, mean_t)
    print("=" * 74)
    print(f"{label}  (prototype support={support}, {len(ordered)} tracks, matched {len(affs)})")
    print(f"  mean sonic-tag affinity = {affs.mean():+.3f}  "
          f"[p10={np.percentile(affs,10):+.3f} p50={np.percentile(affs,50):+.3f} p90={np.percentile(affs,90):+.3f}]")
    print(f"  worst-edge min_T = {min_t:.3f}   mean_T = {mean_t:.3f}")
    for l in steer[:9]:
        print("   " + l.replace("INFO ", ""))

print("\n" + "#" * 74 + "\nOFF -> ON deltas (mean playlist sonic-tag affinity)\n" + "#" * 74)
for artist, tags in [("Brian Eno", ["ambient"]), ("Real Estate", ["jangle pop"])]:
    base = f"{artist.replace(' ','')}_{tags[0].split()[0] if artist=='Brian Eno' else tags[0].split()[0]}"
    # rebuild labels
    off_lbl = [k for k in summary if k.startswith(artist.replace(' ','')) and k.endswith("OFF")][0]
    on_lbl = [k for k in summary if k.startswith(artist.replace(' ','')) and k.endswith("ON")][0]
    off_m = summary[off_lbl][1].mean()
    on_m = summary[on_lbl][1].mean()
    off_mt = summary[off_lbl][3]
    on_mt = summary[on_lbl][3]
    print(f"  {artist:12} {tags[0]:12}: sonic-tag aff OFF={off_m:+.3f} ON={on_m:+.3f} (d={on_m-off_m:+.3f})"
          f"  |  worst-edge min_T OFF={off_mt:.3f} ON={on_mt:.3f} (d={on_mt-off_mt:+.3f})")

# dump tracklists for the user to listen to
import json
with open(os.path.join(OUT, "tracklists.json"), "w", encoding="utf-8") as f:
    json.dump({k: v[0] for k, v in summary.items()}, f, indent=2, ensure_ascii=False)
print(f"\ntracklists -> {os.path.join(OUT, 'tracklists.json')}")
print("DONE.")
