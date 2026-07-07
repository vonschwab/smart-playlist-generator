# tests/unit/test_worker_escalation_review.py
"""Worker handler round-trips for the album-grain escalation review commands."""
import json

from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import (
    handle_apply_escalation_decision,
    handle_get_escalation_completed,
    handle_get_escalation_queue,
)


def _events(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]


def _seed(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    q = EscalationQueue(sidecar)
    q.enqueue(album_id="a1", release_key="slowdive::souvlaki", artist="Slowdive",
              album="Souvlaki", prior_observed_leaf=["indie rock"],
              proposed_genres=[{"term": "shoegaze", "confidence": 0.9}],
              escalate_reason="sparse", dropped_file_tags=["dream pop"],
              prompt_version="pv", model="sonnet", input_hash="h1")
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))
    return sidecar


def test_get_escalation_queue_emits_result(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_get_escalation_queue({"cmd": "get_escalation_queue", "request_id": "r1"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "escalation_queue"
    assert result["pending_albums"] == 1
    assert result["escalations"][0]["album_id"] == "a1"
    assert result["escalations"][0]["dropped_file_tags"] == ["dream pop"]
    assert result["job_id"] is None
    assert done["ok"] is True


def test_apply_escalation_decision_accept_materializes(tmp_path, monkeypatch, capsys):
    sidecar = _seed(tmp_path, monkeypatch)
    handle_apply_escalation_decision({
        "cmd": "apply_escalation_decision", "request_id": "r2",
        "album_id": "a1", "decision": "accept",
    })
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    import sqlite3
    n = sqlite3.connect(sidecar).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'").fetchone()[0]
    assert n >= 1
    assert EscalationQueue(sidecar).get("a1")["status"] == "accepted"


def test_get_escalation_completed_emits_result(tmp_path, monkeypatch, capsys):
    sidecar = _seed(tmp_path, monkeypatch)
    # Mark a1 decided so it shows in the completed view
    q = EscalationQueue(sidecar)
    q._mark("a1", status="accepted", decision_genres=["shoegaze"])
    q.close()
    handle_get_escalation_completed({"cmd": "get_escalation_completed", "request_id": "r4"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "escalation_completed"
    assert result["decided_albums"] == 1
    assert result["job_id"] is None
    assert done["ok"] is True


def test_publish_decided_backs_up_and_publishes(tmp_path, monkeypatch, capsys):
    import sqlite3
    from src.playlist_gui import worker as W
    # Minimal metadata.db: publish() reads tracks, track_genres, album_genres,
    # artist_genres, and albums for legacy genre aggregation.  All tables must
    # exist; empty rows are fine — publish() will produce zero-row output tables.
    meta = tmp_path / "metadata.db"
    c = sqlite3.connect(meta)
    c.executescript("""
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, artist TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
    """)
    c.commit()
    c.close()
    side = tmp_path / "sidecar.db"
    from src.ai_genre_enrichment.storage import SidecarStore
    SidecarStore(str(side)).initialize()
    monkeypatch.setattr(W, "resolve_database_path", lambda *a, **k: str(meta))
    monkeypatch.setattr(W, "SIDECAR_DB_PATH", str(side))

    W.handle_publish_decided({"cmd": "publish_decided", "request_id": "r3", "job_id": "j3"})
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    # a timestamped backup was created next to metadata.db
    assert any(p.name.startswith("metadata.db.bak.") for p in tmp_path.iterdir())
