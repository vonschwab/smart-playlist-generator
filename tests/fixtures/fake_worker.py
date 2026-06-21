"""Minimal NDJSON worker stub for tests. Reads commands on stdin, emits events on stdout."""
import json
import sys

FAKE_WORKER_CMD = [sys.executable, __file__]


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        rid = cmd.get("request_id")
        jid = cmd.get("job_id")
        name = cmd.get("cmd")
        if name == "ping":
            emit({"type": "result", "result_type": "pong", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": "ping", "ok": True, "request_id": rid, "job_id": jid})
        elif name == "generate_playlist":
            emit({"type": "log", "level": "INFO", "msg": "fake: starting", "request_id": rid, "job_id": jid})
            emit({"type": "progress", "stage": "beam", "current": 50, "total": 100, "detail": "searching", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "playlist", "request_id": rid, "job_id": jid, "playlist": {
                "name": "Fake Playlist", "track_count": 2,
                "tracks": [
                    {"position": 0, "rating_key": "k0", "artist": "Acetone", "title": "Sundown",
                     "album": "Cindy", "duration_ms": 200000, "file_path": "/0.flac",
                     "sonic_similarity": 0.91, "genre_similarity": 0.8, "genres": ["slowcore"],
                     "transition_score": 0.62},
                    {"position": 1, "rating_key": "k1", "artist": "Mazzy Star", "title": "Taxi",
                     "album": "So Tonight", "duration_ms": 210000, "file_path": "/1.flac",
                     "sonic_similarity": 0.87, "genre_similarity": 0.7, "genres": ["dreampop"],
                     "transition_score": None},
                ],
                "metrics": {"mean_transition": 0.89, "min_transition": 0.87,
                            "p10_transition": 0.62, "p90_transition": 0.91, "distinct_artists": 2},
            }})
            emit({"type": "progress", "stage": "complete", "current": 100, "total": 100, "detail": "Done", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": "generate_playlist", "ok": True, "detail": "Generated 2 tracks", "request_id": rid, "job_id": jid})
        elif name == "find_replacement_suggestions":
            pos = cmd.get("position")
            if pos in (0, None):
                emit({"type": "error", "message": "Cannot replace pier track", "request_id": rid, "job_id": jid})
                emit({"type": "done", "cmd": name, "ok": False, "detail": "Cannot replace pier track", "request_id": rid, "job_id": jid})
            else:
                emit({"type": "result", "result_type": "replacement_suggestions", "request_id": rid, "job_id": jid,
                      "position": pos, "mode": "best",
                      "candidates": [
                          {"index": 7, "track_id": "k9", "rating_key": "k9", "artist_key": "band",
                           "title": "Fall Like Rain", "artist": "Acetone", "album": "Cindy",
                           "genres": ["slowcore"], "mean_t": 0.74, "t_prev": 0.75, "t_next": 0.73,
                           "duration_ms": 200000, "file_path": "/9.flac"},
                      ]})
                emit({"type": "done", "cmd": name, "ok": True, "detail": "Found 1", "request_id": rid, "job_id": jid})
        elif name == "blacklist_set":
            tids = cmd.get("track_ids", []) or []
            emit({"type": "result", "result_type": "blacklist_set", "request_id": rid, "job_id": jid,
                  "track_ids": tids, "value": cmd.get("value", True), "updated": len(tids)})
            emit({"type": "done", "cmd": name, "ok": True, "detail": f"Updated {len(tids)}", "request_id": rid, "job_id": jid})
        elif name == "blacklist_fetch_scopes":
            emit({"type": "result", "result_type": "blacklist_scopes", "request_id": rid, "job_id": jid,
                  "artists": [{"artist_key": "nick drake", "artist_name": "Nick Drake"}],
                  "albums": [{"artist_key": "nick drake", "album_key": "pink moon",
                              "artist_name": "Nick Drake", "album_name": "Pink Moon"}],
                  "tracks": [{"track_id": "t1", "title": "Harvest", "artist": "Neil Young", "album": "Harvest"}]})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "3 entries", "request_id": rid, "job_id": jid})
        elif name == "blacklist_scope_set":
            emit({"type": "result", "result_type": "blacklist_scope_set", "request_id": rid, "job_id": jid,
                  "scope": cmd.get("scope"), "value": cmd.get("value"), "enabled": cmd.get("enabled", True), "track_ids": []})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "scope set", "request_id": rid, "job_id": jid})
        elif name == "edit_genres":
            genres = cmd.get("genres", []) or []
            emit({"type": "result", "result_type": "edit_genres", "request_id": rid, "job_id": jid,
                  "artist": cmd.get("artist"), "album": cmd.get("album"),
                  "genres": sorted(genres), "added": sorted(genres), "removed": []})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "ok", "request_id": rid, "job_id": jid})
        elif name == "analyze_library":
            stages = cmd.get("stages") or [
                "scan", "genres", "discogs", "lastfm", "sonic", "mert",
                "enrich", "publish", "genre-sim", "artifacts", "energy",
                "genre-embedding", "verify",
            ]
            emit({"type": "log", "level": "INFO", "msg": "fake: analyze starting",
                  "request_id": rid, "job_id": jid})
            emit({"type": "progress", "stage": "analyze_library", "current": 50, "total": 100,
                  "detail": "scanning", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "analyze_library",
                  "request_id": rid, "job_id": jid,
                  "summary": "Analyze complete (fake)",
                  "stages": [
                      {"name": s, "decision": "ran", "duration_ms": 10, "errors": 0}
                      for s in stages
                  ],
                  "out_dir": "/tmp"})
            emit({"type": "done", "cmd": "analyze_library", "ok": True,
                  "detail": f"Done ({len(stages)} stages)",
                  "request_id": rid, "job_id": jid})
        elif name == "enrich_genres":
            scope = cmd.get("scope", "all_unenriched")
            emit({"type": "log", "level": "INFO", "msg": "fake: enrich starting",
                  "request_id": rid, "job_id": jid})
            emit({"type": "progress", "stage": "enrich_genres", "current": 50, "total": 100,
                  "detail": "enriching", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "enrich_genres",
                  "request_id": rid, "job_id": jid,
                  "ok": True, "scope": scope, "releases": 5, "genres_applied": 12})
            emit({"type": "done", "cmd": "enrich_genres", "ok": True,
                  "detail": "Enriched 5 releases",
                  "request_id": rid, "job_id": jid})
        elif name == "get_escalation_queue":
            emit({"type": "result", "result_type": "escalation_queue",
                  "escalations": [{"album_id": "a1", "artist": "Slowdive", "album": "Souvlaki",
                                   "prior_observed_leaf": ["indie rock"],
                                   "proposed_genres": [{"term": "shoegaze", "confidence": 0.9}],
                                   "escalate_reason": "sparse", "dropped_file_tags": [],
                                   "status": "pending", "decision_genres": None}],
                  "pending_albums": 1, "decided_albums": 0, "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "get_escalation_queue", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "get_escalation_completed":
            emit({"type": "result", "result_type": "escalation_completed",
                  "escalations": [{"album_id": "a1", "artist": "Slowdive", "album": "Souvlaki",
                                   "prior_observed_leaf": ["indie rock"],
                                   "proposed_genres": [{"term": "shoegaze", "confidence": 0.9}],
                                   "escalate_reason": "sparse", "dropped_file_tags": [],
                                   "status": "accepted", "decision_genres": ["shoegaze"]}],
                  "pending_albums": 0, "decided_albums": 1,
                  "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "get_escalation_completed", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "apply_escalation_decision":
            emit({"type": "result", "result_type": "escalation_decision",
                  "album_id": cmd.get("album_id"), "status": "accepted",
                  "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "apply_escalation_decision", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "publish_decided":
            emit({"type": "result", "result_type": "publish_decided",
                  "graph_albums": 3325, "legacy_albums": 81, "total_albums": 3428,
                  "collisions": 31, "request_id": rid, "job_id": cmd.get("job_id")})
            emit({"type": "done", "cmd": "publish_decided", "ok": True,
                  "request_id": rid, "job_id": cmd.get("job_id")})
        else:
            emit({"type": "error", "message": f"unknown cmd {name}", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": name or "?", "ok": False, "request_id": rid, "job_id": jid})


if __name__ == "__main__":
    main()
