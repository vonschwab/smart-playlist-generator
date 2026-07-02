"""Regression test for the "silent stale" footgun in compute_stage_fingerprint.

The muq fingerprint branch used to hash only {"stage", "track_ids"} — no
signal for the active sonic variant (artifacts.sonic_variant_override). Task 2
added a variant gate so stage_muq no-ops unless variant == 'muq', but the
orchestrator skips calling a stage at all when its fingerprint is unchanged —
BEFORE the gate runs. With the track set held constant, flipping the active
variant must still bust the cached fingerprint so the gate gets a chance to
re-evaluate and the newly-active variant's extraction actually runs.

SP-B Task 7 removed the MERT analyze path (stage, fold, scripts) entirely —
"mert" is no longer a registered stage, only a legacy override value that
compute_stage_fingerprint's generic fallback still hashes deterministically.
"""
import argparse
import sqlite3
from types import SimpleNamespace

from scripts.analyze_library import compute_config_hash, compute_stage_fingerprint


def _make_conn():
    """A tiny in-memory tracks table standing in for the real DB."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tracks (track_id TEXT, file_path TEXT)")
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("t1", "/music/a.mp3"))
    conn.commit()
    return conn


def _config_hash_for_variant(tmp_path, variant):
    """Real compute_config_hash() output for a config file whose only content
    is artifacts.sonic_variant_override=<variant> — i.e. the same cfg_hash the
    orchestrator would thread into ctx["config_hash"] for a real run."""
    import yaml

    p = tmp_path / f"config_{variant}.yaml"
    p.write_text(
        yaml.safe_dump({"artifacts": {"sonic_variant_override": variant}}),
        encoding="utf-8",
    )
    cfg_data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = SimpleNamespace(config=cfg_data)
    args = argparse.Namespace(stages=None, max_tracks=None, limit=None, out_dir=None)
    return compute_config_hash(cfg, args)


def test_muq_fingerprint_differs_when_active_variant_flips(tmp_path):
    conn = _make_conn()
    mert_hash = _config_hash_for_variant(tmp_path, "mert")
    muq_hash = _config_hash_for_variant(tmp_path, "muq")

    fp_variant_mert = compute_stage_fingerprint({"conn": conn, "config_hash": mert_hash}, "muq")
    fp_variant_muq = compute_stage_fingerprint({"conn": conn, "config_hash": muq_hash}, "muq")

    assert fp_variant_mert != fp_variant_muq, (
        "stage_muq's fingerprint must change when the active sonic variant "
        "flips (same track set) — otherwise flipping to 'muq' with an "
        "already-cached fingerprint silently never runs the extraction."
    )


def test_fingerprint_unaffected_by_variant_when_config_hash_absent(tmp_path):
    """Sanity check: with track_ids held constant and no config_hash supplied
    (ctx.get("config_hash", "") default), both branches still hash deterministically
    off the same empty string — i.e. the new "config" key doesn't break the
    no-config-hash-available path, it just can't distinguish variants there."""
    conn = _make_conn()
    fp1 = compute_stage_fingerprint({"conn": conn}, "mert")
    fp2 = compute_stage_fingerprint({"conn": conn}, "mert")
    assert fp1 == fp2


def test_mert_stage_no_longer_registered():
    from scripts.analyze_library import STAGE_FUNCS
    assert "mert" not in STAGE_FUNCS


def test_artifacts_fingerprint_stable_across_manifest_write(tmp_path):
    """The artifacts fingerprint must be independent of artifact_manifest.json.

    The manifest is written AFTER this fingerprint is computed and stores the
    fingerprint itself; folding its mtime in made the fingerprint
    self-invalidating — verify's manifest-vs-recomputed check could never match
    after a rebuild, and the artifacts stage could never fingerprint_same-skip.
    Regression for the v3.2.0 manifest_mtime bug, surfaced by SP-B Task 10's
    acceptance gate (verify reported spurious `stale_artifact` right after a
    clean rebuild).
    """
    conn = _make_conn()
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()
    (out_dir / "data_matrices_step1.npz").write_bytes(b"stub-npz")
    ctx = {"conn": conn, "out_dir": out_dir, "config_hash": "h"}

    fp_before = compute_stage_fingerprint(ctx, "artifacts")
    # The manifest is (re)written after the fingerprint is computed on every rebuild.
    (out_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    fp_after = compute_stage_fingerprint(ctx, "artifacts")

    assert fp_before == fp_after, (
        "artifacts fingerprint must not change when artifact_manifest.json is "
        "written — the manifest stores this very fingerprint, so folding its "
        "mtime in makes the fingerprint self-invalidating."
    )
