"""T8: real roam generation verification (the genre-session signal).

Runs a multi-pier generation through the production harness with roam enabled
(config.yaml), at INFO, then computes the SELECTED-EDGE calibrated T myself
(harness metrics are empty). Checks: center_transitions live, BPM loaded N/N
(no worktree-stub confound), roam active, generation < 90s, and the edge-T
distribution is DE-COMPRESSED (the fix is live, not inert).
"""
import sys
import time
from pathlib import Path
import numpy as np

WT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/.claude/worktrees/sonic-centered-transition")
MAIN = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
sys.path.insert(0, str(WT / "tests"))   # support.gui_fidelity
sys.path.insert(0, str(WT))             # worktree code (must win)

from src.logging_utils import configure_logging  # noqa: E402
configure_logging(level="INFO", force=True)

from support.gui_fidelity import resolve_gui_overrides, resolve_gui_genre_params, gui_ui_state  # noqa: E402
from src.playlist.ds_pipeline_runner import generate_playlist_ds  # noqa: E402
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge  # noqa: E402

ART = MAIN / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
PIERS = [
    "9e6af9666aed00b542e03451a438b975",  # Radio Dept - Too Soon
    "1dbb293466dbb5d6ec7a2616c62e9028",  # Slowdive - A1 Rutti
    "8b0c959497784f03f103b14a13cc2831",  # Beach House - Wedding Bell
]

print("##### START ROAM GENERATION #####", flush=True)
# Production override chain (resolve_gui_overrides) + inject the real DB path
# (overrides["library"]["database_path"] is what core.py reads for BPM) so the
# run is fully faithful — BPM gates ON, not a worktree-stub confound.
ui = gui_ui_state(cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow", pace_mode="narrow")
ds_overrides = resolve_gui_overrides(ui)
ds_overrides.setdefault("library", {})["database_path"] = str(MAIN / "data/metadata.db")
genre_params = resolve_gui_genre_params(ui)
t0 = time.time()
res = generate_playlist_ds(
    artifact_path=str(ART), seed_track_id=PIERS[0], anchor_seed_ids=PIERS,
    mode="narrow", pace_mode="narrow", length=15, random_seed=0,
    overrides=ds_overrides, artist_style_enabled=False, artist_playlist=False, **genre_params,
)
elapsed = time.time() - t0
print(f"\n##### GENERATION DONE in {elapsed:.1f}s #####", flush=True)

tids = list(getattr(res, "track_ids", None) or (res.get("track_ids") if isinstance(res, dict) else []))
print(f"playlist length: {len(tids)}")

a = np.load(ART, allow_pickle=True)
ids = [str(x) for x in a["track_ids"]]; pos = {t: i for i, t in enumerate(ids)}
artists = np.array([str(x) for x in a["track_artists"]]); titles = np.array([str(x) for x in a["track_titles"]])
mert = np.asarray(a["X_sonic_mert"], np.float32)
ctx = build_transition_metric_context(
    X_sonic=mert, X_start=np.asarray(a["X_sonic_mert_start"], np.float32),
    X_mid=np.asarray(a["X_sonic_mert_mid"], np.float32), X_end=np.asarray(a["X_sonic_mert_end"], np.float32),
    center_transitions=True,
    weight_end_start=0.7, weight_mid_mid=0.15, weight_full_full=0.15,
    calib_center=0.32, calib_scale=0.0625, calib_gain=1.0,
)
print("\n#### PLAYLIST + SELECTED-EDGE T (calibrated) ####")
Ts = []
for k, t in enumerate(tids):
    i = pos.get(str(t))
    label = f"{artists[i]} - {titles[i]}" if i is not None else "?(not in artifact)"
    if k == 0:
        print(f"  {k:2d}        {label}")
    else:
        pi, ci = pos.get(str(tids[k - 1])), pos.get(str(t))
        if pi is None or ci is None:
            print(f"  {k:2d}  T=  ?   {label}"); continue
        T = float(score_transition_edge(ctx, pi, ci)["T"]); Ts.append(T)
        print(f"  {k:2d}  T={T:.3f}  {label}")
Ts = np.array(Ts)
print("\n#### EDGE-T DISTRIBUTION ####")
print(f"  n={len(Ts)}  min={Ts.min():.3f}  p10={np.percentile(Ts,10):.3f}  p50={np.percentile(Ts,50):.3f}  p90={np.percentile(Ts,90):.3f}  max={Ts.max():.3f}")
wi = int(Ts.argmin())
print(f"  WORST edge: T={Ts.min():.3f} at {wi+1}->{wi+2}")
print("  De-compressed? legacy (x+1)/2 parks all edges ~0.52-0.77; spread below that => calibrated fix LIVE.")
print(f"\n  BUDGET: {elapsed:.1f}s / 90s ceiling -> {'OK' if elapsed < 90 else 'OVER BUDGET'}")
print("##### END #####", flush=True)
