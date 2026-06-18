"""Minimal NDJSON worker stub for tests. Reads commands on stdin, emits events on stdout."""
import json
import sys


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
        elif name == "scan_genre_review":
            emit({"type": "progress", "stage": "scan_genre_review", "current": 1, "total": 2,
                  "detail": "acetone – cindy", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "scan_genre_review",
                  "request_id": rid, "job_id": jid,
                  "releases_scanned": 2, "new_terms": 3, "pruned_terms": 0, "pending_terms": 3})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "Scanned 2 releases",
                  "request_id": rid, "job_id": jid})
        elif name == "get_genre_review_queue":
            emit({"type": "result", "result_type": "genre_review_queue",
                  "request_id": rid, "job_id": jid,
                  "releases": [{
                      "release_key": "acetone::cindy", "artist": "acetone", "album": "cindy",
                      "pending": [
                          {"term": "slowcore", "confidence": 0.4, "basis": "hybrid_fusion",
                           "sources": ["lastfm_tags"], "reason": "uncertain", "status": "pending"},
                          {"term": "sadcore", "confidence": 0.3, "basis": "layered_taxonomy",
                           "sources": ["discogs"], "reason": "Unknown layered taxonomy term.",
                           "status": "pending"},
                      ],
                      "decided": [],
                  }],
                  "pending_releases": 1, "pending_terms": 2})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "2 pending",
                  "request_id": rid, "job_id": jid})
        elif name == "get_genre_review_completed":
            emit({"type": "result", "result_type": "genre_review_completed",
                  "request_id": rid, "job_id": jid,
                  "releases": [{
                      "release_key": "acetone::cindy", "artist": "acetone", "album": "cindy",
                      "pending": [],
                      "decided": [
                          {"term": "slowcore", "confidence": 0.4, "basis": "hybrid_fusion",
                           "sources": ["lastfm_tags"], "reason": "uncertain",
                           "status": "accepted"},
                      ],
                  }],
                  "decided_releases": 1, "decided_terms": 1})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "1 decided",
                  "request_id": rid, "job_id": jid})
        elif name == "apply_genre_review_decision":
            decision = cmd.get("decision", "accept")
            status = {"accept": "accepted", "reject": "rejected", "revert": "pending"}.get(decision)
            if status is None:
                emit({"type": "error", "message": f"invalid decision: {decision}",
                      "request_id": rid, "job_id": jid})
                emit({"type": "done", "cmd": name, "ok": False, "request_id": rid, "job_id": jid})
            else:
                emit({"type": "result", "result_type": "genre_review_decision",
                      "request_id": rid, "job_id": jid,
                      "release_key": cmd.get("release_key"), "term": cmd.get("term"),
                      "decision": decision, "status": status})
                emit({"type": "done", "cmd": name, "ok": True, "detail": status,
                      "request_id": rid, "job_id": jid})
        else:
            emit({"type": "error", "message": f"unknown cmd {name}", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": name or "?", "ok": False, "request_id": rid, "job_id": jid})


if __name__ == "__main__":
    main()
