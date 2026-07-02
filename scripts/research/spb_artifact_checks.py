"""SP-B acceptance: the rebuilt artifact carries per-variant keys only, same universe.

Usage: python scripts/research/spb_artifact_checks.py <backup.npz>
Compares the live data_matrices_step1.npz against the pre-rebuild backup:
no dead MERT/tower keys, no plain X_sonic, identical track universe, and a
byte-identical X_sonic_muq matrix (the rebuild must not touch the sonic space).
"""
import sys
import zipfile

import numpy as np

NEW_P = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"


def keys(p):
    with zipfile.ZipFile(p) as z:
        return {n[:-4] for n in z.namelist() if n.endswith(".npy")}


def main() -> None:
    bak_p = sys.argv[1]
    new_keys = keys(NEW_P)
    dead = {k for k in new_keys if k.startswith((
        "X_sonic_mert", "X_sonic_rhythm", "X_sonic_timbre", "X_sonic_harmony",
        "X_sonic_tower", "X_sonic_raw", "X_sonic_pre_scaled", "X_sonic_robust",
        "mert_", "tower_", "normalizer_params", "bpm_array", "sonic_feature_names",
    ))}
    # X_sonic_pre_scaled is a muq-fold provenance flag, not a dead tower key.
    dead.discard("X_sonic_pre_scaled")
    assert not dead, f"dead keys survived: {sorted(dead)}"
    assert "X_sonic" not in new_keys, "plain X_sonic must be gone (per-variant contract)"
    assert "X_sonic_start" not in new_keys and "X_sonic_end" not in new_keys, "legacy window keys must be gone"
    for k in ("X_sonic_muq", "X_sonic_variant", "X_genre_raw", "X_genre_smoothed", "track_ids"):
        assert k in new_keys, f"missing required key {k}"

    with np.load(NEW_P, allow_pickle=True) as znew, np.load(bak_p, allow_pickle=True) as zbak:
        assert np.array_equal(znew["track_ids"], zbak["track_ids"]), "TRACK UNIVERSE CHANGED"
        stamp = znew["X_sonic_variant"]
        stamp = str(stamp.item()) if stamp.shape == () else str(stamp)
        assert stamp == "muq", f"variant stamp is {stamp!r}, expected 'muq'"
        n = znew["track_ids"].shape[0]
        assert znew["X_sonic_muq"].shape == (n, 512), znew["X_sonic_muq"].shape
        assert np.array_equal(znew["X_sonic_muq"], zbak["X_sonic_muq"]), (
            "muq matrix changed — must be byte-identical to the backup's"
        )
    print(f"OK: {n} tracks, muq-only artifact ({len(new_keys)} keys), "
          f"universe + muq matrix identical to backup")
    print(f"remaining keys: {sorted(new_keys)}")


if __name__ == "__main__":
    main()
