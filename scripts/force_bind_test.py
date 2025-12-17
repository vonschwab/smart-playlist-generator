#!/usr/bin/env python3
"""
Force-binding test: Verify that min_genre_similarity, genre_method,
and transition_strictness parameters are not wired into DS pipeline generation.

This script creates two EXTREME configurations that SHOULD diverge if the
parameters were properly wired:
- Config A: lenient (low genre sim, soft strictness)
- Config B: strict (high genre sim, hard strictness)

If they produce identical playlists/metrics, the parameters are not wired.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.playlist.pipeline import generate_playlist_ds
import hashlib

# Configuration
ARTIFACT_PATH = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"
SEED_ID = "1c347ff04e65adf7923a9e3927ab667a"
PLAYLIST_LENGTH = 50  # Longer playlist to give dials more room to diverge

print("=" * 80)
print("FORCE-BINDING TEST: Do non-sonic dials affect DS pipeline generation?")
print("=" * 80)

# Check artifact exists
if not Path(ARTIFACT_PATH).exists():
    print(f"ERROR: Artifact not found at {ARTIFACT_PATH}")
    sys.exit(1)

print(f"\nArtifact: {ARTIFACT_PATH}")
print(f"Seed: {SEED_ID}")
print(f"Playlist Length: {PLAYLIST_LENGTH}")

# ============================================================================
# TEST 1: min_genre_similarity (0.15 vs 0.60)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: min_genre_similarity Effect")
print("=" * 80)
print("Config A: min_genre_similarity=0.15 (lenient)")
print("Config B: min_genre_similarity=0.60 (strict)")
print("Expected: Different playlists/metrics if parameter is wired")
print()

result_a = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    min_genre_similarity=0.15,
)

result_b = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    min_genre_similarity=0.60,
)

# Compare
playlist_match = result_a.track_ids == result_b.track_ids
hash_a = hashlib.md5(",".join(result_a.track_ids).encode()).hexdigest()
hash_b = hashlib.md5(",".join(result_b.track_ids).encode()).hexdigest()

print(f"Config A tracks hash: {hash_a[:12]}")
print(f"Config B tracks hash: {hash_b[:12]}")
print(f"Playlists identical: {playlist_match}")
print(f"Metrics identical: {result_a.stats == result_b.stats}")

if playlist_match:
    print("⚠️  RESULT: min_genre_similarity is NOT WIRED (produces identical output)")
else:
    print("✓ RESULT: min_genre_similarity IS WIRED (produces different output)")

# ============================================================================
# TEST 2: genre_method (ensemble vs weighted_jaccard)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: genre_method Effect")
print("=" * 80)
print("Config A: genre_method=ensemble")
print("Config B: genre_method=weighted_jaccard")
print("Expected: Different playlists/metrics if parameter is wired")
print()

result_c = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    genre_method="ensemble",
)

result_d = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    genre_method="weighted_jaccard",
)

# Compare
playlist_match = result_c.track_ids == result_d.track_ids
hash_c = hashlib.md5(",".join(result_c.track_ids).encode()).hexdigest()
hash_d = hashlib.md5(",".join(result_d.track_ids).encode()).hexdigest()

print(f"Config A (ensemble) hash: {hash_c[:12]}")
print(f"Config B (weighted_jaccard) hash: {hash_d[:12]}")
print(f"Playlists identical: {playlist_match}")
print(f"Metrics identical: {result_c.stats == result_d.stats}")

if playlist_match:
    print("⚠️  RESULT: genre_method is NOT WIRED (produces identical output)")
else:
    print("✓ RESULT: genre_method IS WIRED (produces different output)")

# ============================================================================
# TEST 3: transition_strictness via overrides (baseline vs very strict)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: transition_strictness Effect")
print("=" * 80)
print("Config A: baseline (transition_floor=0.3)")
print("Config B: very strict (transition_floor=0.8, hard_floor=True)")
print("Expected: Different playlists/metrics if constraint is binding")
print()

result_e = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    overrides={"construct": {}},  # baseline
)

result_f = generate_playlist_ds(
    artifact_path=ARTIFACT_PATH,
    seed_track_id=SEED_ID,
    num_tracks=PLAYLIST_LENGTH,
    mode="dynamic",
    random_seed=42,
    sonic_weight=0.60,
    genre_weight=0.40,
    overrides={"construct": {"transition_floor": 0.8, "hard_floor": True}},
)

# Compare
playlist_match = result_e.track_ids == result_f.track_ids
hash_e = hashlib.md5(",".join(result_e.track_ids).encode()).hexdigest()
hash_f = hashlib.md5(",".join(result_f.track_ids).encode()).hexdigest()

print(f"Config A (baseline) hash: {hash_e[:12]}")
print(f"Config B (strict) hash: {hash_f[:12]}")
print(f"Playlists identical: {playlist_match}")

metrics_e = result_e.stats.get("playlist", {})
metrics_f = result_f.stats.get("playlist", {})
mean_trans_e = metrics_e.get("mean_transition", 0)
mean_trans_f = metrics_f.get("mean_transition", 0)
print(f"Mean transition Config A: {mean_trans_e:.4f}")
print(f"Mean transition Config B: {mean_trans_f:.4f}")
print(f"Transition metrics changed: {mean_trans_e != mean_trans_f}")

if playlist_match:
    print("⚠️  RESULT: transition_strictness constraint is NOT BINDING")
else:
    print("✓ RESULT: transition_strictness IS BINDING (produces different output)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

findings = []
if playlist_match:
    findings.append("- min_genre_similarity: NOT WIRED")
    findings.append("- genre_method: NOT WIRED")
if not (result_e.track_ids != result_f.track_ids):
    findings.append("- transition_strictness: NOT BINDING")

if findings:
    print("\nParameters with no effect on generation:")
    for finding in findings:
        print(finding)
    print("\nConclusion: These parameters need implementation work to affect DS pipeline.")
else:
    print("\nAll tested parameters affect generation. No issues found.")
