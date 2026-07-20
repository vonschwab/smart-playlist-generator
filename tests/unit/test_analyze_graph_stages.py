"""Unit tests for the analyze_library graph stages (lastfm, enrich, publish)."""
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.analyze_library as al
from src.ai_genre_enrichment.storage import SidecarStore


def _metadata_db(tmp_path: Path) -> str:
    """Minimal metadata.db: one album with one track and an artist genre."""
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
            album_id TEXT, title TEXT, file_path TEXT);
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        INSERT INTO tracks VALUES ('t1','Slowdive','Souvlaki','alb1','Alison','/x/a.flac');
        INSERT INTO albums VALUES ('alb1','Souvlaki','Slowdive');
        INSERT INTO artist_genres VALUES ('Slowdive','shoegaze','musicbrainz_artist');
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _ctx(tmp_path: Path, db_path: str, **arg_overrides):
    """Build a minimal stage ctx with a live metadata.db connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    defaults = dict(
        force=False, limit=None, dry_run=False, progress=False, verbose=False,
        progress_interval=15.0, progress_every=500, max_tracks=0, model=None,
        enrich_chunk_size=50, lastfm_api_key="FAKEKEY",
    )
    defaults.update(arg_overrides)
    args = Namespace(**defaults)
    return {
        "config_path": str(tmp_path / "config.yaml"),
        "db_path": db_path,
        "out_dir": tmp_path,
        "args": args,
        "conn": conn,
        "config_hash": "test",
        "library_root": str(tmp_path),
        "genres_dirty": False, "sonic_dirty": False,
        "artifacts_dirty": False, "force_stage": False,
    }


def test_stage_lastfm_fetches_stores_and_classifies(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))

    captured = {}

    def fake_fetch(artist, album, api_key, limit=20):
        captured["args"] = (artist, album, api_key, limit)
        return ["shoegaze", "dream pop", "ambient"]

    monkeypatch.setattr(al, "fetch_lastfm_tags", fake_fetch)

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_lastfm(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["extracted"] == 1
    # tags landed in the sidecar as a lastfm_tags source page
    store = SidecarStore(sidecar)
    assert "slowdive::souvlaki" in store.release_keys_with_source_type("lastfm_tags")


def test_stage_lastfm_missing_key_skips(tmp_path, monkeypatch):
    """Not-configured (no key) is a loud skip, not a raise — see
    tests/unit/test_analyze_stage_policy.py for the full skip-vs-fail policy."""
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    ctx = _ctx(tmp_path, db_path, lastfm_api_key=None)
    # also ensure config lookup can't supply a key
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: None)
    result = al.stage_lastfm(ctx)
    ctx["conn"].close()
    assert result["skipped"] is True
    assert "not configured" in result["reason"]


def _seed_sidecar_with_pages(sidecar: str):
    """One release with a lastfm source page carrying a known + an unknown tag."""
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="alb1",
        source_url="lastfm://artist/slowdive/album/souvlaki",
        source_type="lastfm_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="seed",
    )
    # 'shoegaze' classifies deterministically; 'zzz unknown thing' is review_only.
    store.replace_source_tags(page_id, ["shoegaze", "zzz unknown thing"])
    return store


class _RecordingAdjudicator:
    """Stand-in for adjudicate_tags: records calls, returns canned classifications."""
    def __init__(self):
        self.calls = []

    def __call__(self, tags, *, model=None, dry_run=False, client=None):
        self.calls.append([norm for _, norm in tags])
        return {
            norm: {"classification": "genre_style", "confidence": 0.8, "reason": "ok"}
            for _, norm in tags
        }


def test_stage_enrich_dedupes_adjudicates_and_materializes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    rec = _RecordingAdjudicator()
    monkeypatch.setattr(al, "adjudicate_tags", rec)

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["releases_enriched"] == 1
    # exactly one distinct unknown tag adjudicated, in a single chunk
    assert rec.calls == [["zzz unknown thing"]]
    assert result["tags_adjudicated"] == 1


# ── SP-1 zero-touch gate (task 7b, 2026-07-20) ──────────────────────────────
# stage_enrich's deterministic pre-pass always runs; only the Claude
# adjudication of LEFTOVER unknown tags is gated for provider in
# (zero_touch, skip) -- those tags are "whiffed honestly" as permanently
# unmapped instead of costing an LLM call. claude_code (the owner's default)
# must be completely unaffected -- pinned by the second test below.


def test_stage_enrich_skips_adjudicate_tags_for_zero_touch_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("PG_AI_PROVIDER", "zero_touch")
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    rec = _RecordingAdjudicator()
    monkeypatch.setattr(al, "adjudicate_tags", rec)

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert rec.calls == []
    assert result["tags_adjudicated"] == 0
    # the deterministic pre-pass still ran and still materialized the release
    assert result["releases_enriched"] == 1


def test_stage_enrich_calls_adjudicate_tags_for_default_claude_code_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("PG_AI_PROVIDER", raising=False)
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    rec = _RecordingAdjudicator()
    monkeypatch.setattr(al, "adjudicate_tags", rec)

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert rec.calls == [["zzz unknown thing"]]
    assert result["tags_adjudicated"] == 1


def test_stage_enrich_no_pending_skips(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()  # empty sidecar, no source pages
    monkeypatch.setattr(al, "adjudicate_tags", _RecordingAdjudicator())

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()
    assert result["skipped"] is True
    assert result.get("releases_enriched", 0) == 0


def test_stage_enrich_propagates_hard_config_error(tmp_path, monkeypatch):
    # A config error re-running cannot fix (CLI missing / unauthenticated) must
    # propagate loudly — it is not a resumable pause.
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    def boom(tags, *, model=None, dry_run=False, client=None):
        raise RuntimeError("claude-agent-sdk is not installed")

    monkeypatch.setattr(al, "adjudicate_tags", boom)
    ctx = _ctx(tmp_path, db_path)
    with pytest.raises(RuntimeError, match="not installed"):
        al.stage_enrich(ctx)
    ctx["conn"].close()


def test_stage_enrich_pauses_on_transient_rate_window(tmp_path, monkeypatch):
    # A transient Claude failure mid-sweep (rate/usage window) pauses cleanly
    # instead of raising — the cache persists and the run resumes on re-run.
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    def boom(tags, *, model=None, dry_run=False, client=None):
        raise RuntimeError(
            "Claude Code request failed after retries: "
            "Claude Code returned an error result: success"
        )

    monkeypatch.setattr(al, "adjudicate_tags", boom)
    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["paused"] is True
    assert result["skipped"] is False
    assert "success" in result["pause_reason"]


def test_run_pipeline_pause_halts_before_publish(tmp_path, monkeypatch):
    # enrich pausing must stop the run before publish so partial enrichment is
    # never written to metadata.db; the report records the pause and rc == 0.
    import json
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    def boom(tags, *, model=None, dry_run=False, client=None):
        raise RuntimeError("Claude Code returned an error result: success")

    monkeypatch.setattr(al, "adjudicate_tags", boom)
    config_path = _write_minimal_config(tmp_path, db_path)
    out_dir = str(tmp_path / "artifacts")
    args = Namespace(
        config=config_path, db_path=db_path, out_dir=out_dir,
        stages="enrich,publish", lastfm_api_key="FAKEKEY", model=None,
        enrich_chunk_size=50, dry_run=False, beat_sync=False, force=False,
        limit=None, max_tracks=0, workers="auto", progress=False,
        progress_interval=15.0, progress_every=500, verbose=False,
        debug=False, quiet=False, log_level="INFO",
    )

    rc = al.run_pipeline(args, console_logging=False)
    assert rc == 0
    assert not _published_table_exists(db_path), "publish must not run after a pause"

    report = json.loads((Path(out_dir) / "analyze_run_report.json").read_text(encoding="utf-8"))
    assert report.get("paused") is True
    assert report.get("paused_stage") == "enrich"
    assert report["stages"]["enrich"]["decision"] == "paused"
    assert "publish" not in report["stages"]


def _published_table_exists(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='release_effective_genres'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def test_stage_publish_first_run_backs_up_and_publishes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_publish(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["backed_up"] is True
    # a timestamped backup was created next to metadata.db
    backups = list(Path(db_path).parent.glob("metadata.db.bak.*"))
    assert backups, "expected a first-publish backup"
    # release_effective_genres now exists
    assert _published_table_exists(db_path)
    assert result["validation_ok"] is True


def test_stage_publish_second_run_no_backup(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx1 = _ctx(tmp_path, db_path)
    al.stage_publish(ctx1)
    ctx1["conn"].close()  # first publish (backs up)
    before = set(Path(db_path).parent.glob("metadata.db.bak.*"))
    ctx2 = _ctx(tmp_path, db_path)
    result = al.stage_publish(ctx2)
    ctx2["conn"].close()
    after = set(Path(db_path).parent.glob("metadata.db.bak.*"))

    assert result["backed_up"] is False
    assert before == after, "second publish must not create a new backup"


def test_stage_publish_dry_run_rolls_back(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path, dry_run=True)
    result = al.stage_publish(ctx)
    ctx["conn"].close()
    assert result["dry_run"] is True
    # dry-run rolls back the publish transaction → no published table persists
    assert not _published_table_exists(db_path)


def _write_graph_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "playlists:\n  ds_pipeline:\n    genre_source: graph\n", encoding="utf-8"
    )


def test_stage_publish_marks_genres_dirty_when_graph_source(tmp_path, monkeypatch):
    """SP4: when artifacts consume the graph, a publish must trigger a rebuild."""
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()
    _write_graph_config(tmp_path)

    ctx = _ctx(tmp_path, db_path)
    al.stage_publish(ctx)
    ctx["conn"].close()
    assert ctx["genres_dirty"] is True


def test_stage_publish_keeps_genres_clean_for_legacy_source(tmp_path, monkeypatch):
    """Legacy artifacts don't read published genres — no forced rebuild."""
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()
    # no config file → genre_source defaults to legacy

    ctx = _ctx(tmp_path, db_path)
    al.stage_publish(ctx)
    ctx["conn"].close()
    assert ctx["genres_dirty"] is False


def test_stage_publish_dry_run_does_not_mark_dirty(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()
    _write_graph_config(tmp_path)

    ctx = _ctx(tmp_path, db_path, dry_run=True)
    al.stage_publish(ctx)
    ctx["conn"].close()
    assert ctx["genres_dirty"] is False


def test_artifacts_fingerprint_tracks_published_genres_in_graph_mode(tmp_path, monkeypatch):
    """A re-publish must change the artifacts fingerprint when source=graph."""
    db_path = _metadata_db(tmp_path)
    _write_graph_config(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE release_effective_genres (
            album_id TEXT, genre_id TEXT, assignment_layer TEXT,
            confidence REAL, source TEXT,
            PRIMARY KEY (album_id, genre_id, assignment_layer)
        );
        INSERT INTO release_effective_genres VALUES ('alb1','shoegaze','observed_leaf',1.0,'graph');
        """
    )
    conn.commit()
    conn.close()

    ctx = _ctx(tmp_path, db_path)
    fp_before = al.compute_stage_fingerprint(ctx, "artifacts")
    ctx["conn"].execute(
        "INSERT INTO release_effective_genres VALUES ('alb1','dream_pop','observed_leaf',0.9,'graph')"
    )
    fp_after = al.compute_stage_fingerprint(ctx, "artifacts")
    ctx["conn"].close()
    assert fp_before != fp_after


def test_taxonomy_version_folds_into_downstream_fingerprints(tmp_path, monkeypatch):
    """A taxonomy growth must invalidate publish/genre-sim/artifacts fingerprints.

    Regression: publish's fingerprint keyed only on row counts, so a growth
    (same counts, re-resolved ids) fingerprint-same-skipped publish (2026-07-02).
    """
    db_path = _metadata_db(tmp_path)
    _write_graph_config(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, "
        "assignment_layer TEXT, confidence REAL, source TEXT)")
    conn.commit()
    conn.close()

    import src.ai_genre_enrichment.layered_taxonomy as lt

    class _StubTax:
        def __init__(self, version):
            self.version = version

    def _fp(stage, version):
        monkeypatch.setattr(lt, "load_default_layered_taxonomy", lambda: _StubTax(version))
        ctx = _ctx(tmp_path, db_path)
        try:
            return al.compute_stage_fingerprint(ctx, stage)
        finally:
            ctx["conn"].close()

    for stage in ("publish", "genre-sim", "artifacts"):
        a = _fp(stage, "0.1.0")
        b = _fp(stage, "0.2.0")
        a_again = _fp(stage, "0.1.0")
        assert a != b, f"{stage} fingerprint ignored taxonomy version change"
        assert a == a_again, f"{stage} fingerprint unstable within a version"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_default_stage_order_has_new_stages_positioned():
    order = al.STAGE_ORDER_DEFAULT
    # lastfm and publish remain in the default order; enrich is opt-in only (not in default)
    for name in ("lastfm", "publish"):
        assert name in order, f"{name} missing from STAGE_ORDER_DEFAULT"
        assert name in al.STAGE_FUNCS
    # adjudicate + apply replace enrich in the default order
    assert "adjudicate" in order
    assert "apply" in order
    assert "enrich" not in order, "enrich should be opt-in only (not in default order)"
    # enrich is still registered so --stages enrich still works
    assert "enrich" in al.STAGE_FUNCS
    # lastfm after discogs; adjudicate+apply after muq; publish after apply
    assert order.index("lastfm") > order.index("discogs")
    assert order.index("adjudicate") > order.index("muq")
    assert order.index("apply") == order.index("adjudicate") + 1
    assert order.index("publish") == order.index("apply") + 1
    # publish precedes genre-sim/artifacts
    assert order.index("publish") < order.index("genre-sim")


def _write_minimal_config(tmp_path: Path, db_path: str) -> str:
    """Write a minimal config.yaml that satisfies Config._validate_config."""
    import yaml
    cfg = {"library": {"database_path": db_path, "music_directory": str(tmp_path)}}
    config_path = str(tmp_path / "config.yaml")
    Path(config_path).write_text(yaml.dump(cfg), encoding="utf-8")
    return config_path


def test_run_pipeline_runs_new_stages_then_skips_on_rerun(tmp_path, monkeypatch):
    import json
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    monkeypatch.setattr(al, "fetch_lastfm_tags",
                        lambda artist, album, api_key, limit=20: ["shoegaze", "zzz unknown thing"])
    monkeypatch.setattr(al, "adjudicate_tags", _RecordingAdjudicator())

    config_path = _write_minimal_config(tmp_path, db_path)
    out_dir = str(tmp_path / "artifacts")

    # Build args Namespace — every attribute read by run_pipeline.
    args = Namespace(
        config=config_path,
        db_path=db_path,
        out_dir=out_dir,
        stages="lastfm,enrich,publish",
        lastfm_api_key="FAKEKEY",
        model=None,
        enrich_chunk_size=50,
        dry_run=False,
        beat_sync=False,
        force=False,
        limit=None,
        max_tracks=0,
        workers="auto",
        progress=False,
        progress_interval=15.0,
        progress_every=500,
        verbose=False,
        # resolve_log_level reads these via getattr with defaults
        debug=False,
        quiet=False,
        log_level="INFO",
    )

    # First run: all three stages should execute (no prior fingerprint).
    rc = al.run_pipeline(args, console_logging=False)
    assert rc == 0
    assert _published_table_exists(db_path)

    report_path = Path(out_dir) / "analyze_run_report.json"
    assert report_path.exists(), "run_pipeline must write analyze_run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for stage in ("lastfm", "enrich", "publish"):
        decision = report["stages"][stage]["decision"]
        assert decision == "ran", f"Expected stage {stage!r} to run on first pass, got {decision!r}"

    # Re-run: inputs unchanged → all three stages skip via fingerprint.
    rc2 = al.run_pipeline(args, console_logging=False)
    assert rc2 == 0
    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    for stage in ("lastfm", "enrich", "publish"):
        decision = report2["stages"][stage]["decision"]
        assert decision == "skipped", (
            f"Expected stage {stage!r} to be skipped on rerun (fingerprint unchanged), got {decision!r}"
        )
