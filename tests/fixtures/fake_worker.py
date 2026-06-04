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
                     "sonic_similarity": 0.91, "genre_similarity": 0.8, "genres": ["slowcore"]},
                    {"position": 1, "rating_key": "k1", "artist": "Mazzy Star", "title": "Taxi",
                     "album": "So Tonight", "duration_ms": 210000, "file_path": "/1.flac",
                     "sonic_similarity": 0.87, "genre_similarity": 0.7, "genres": ["dreampop"]},
                ],
                "metrics": {"mean_transition": 0.89, "min_transition": 0.87, "distinct_artists": 2},
            }})
            emit({"type": "progress", "stage": "complete", "current": 100, "total": 100, "detail": "Done", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": "generate_playlist", "ok": True, "detail": "Generated 2 tracks", "request_id": rid, "job_id": jid})
        else:
            emit({"type": "error", "message": f"unknown cmd {name}", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": name or "?", "ok": False, "request_id": rid, "job_id": jid})


if __name__ == "__main__":
    main()
