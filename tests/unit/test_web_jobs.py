# tests/unit/test_web_jobs.py
from src.playlist_web.jobs import JobRegistry


def _gen_events(job_id):
    return [
        {"type": "log", "level": "INFO", "msg": "starting", "job_id": job_id},
        {"type": "progress", "stage": "beam", "current": 50, "total": 100, "detail": "searching", "job_id": job_id},
        {"type": "result", "result_type": "playlist", "job_id": job_id, "playlist": {
            "name": "P", "track_count": 1,
            "tracks": [{"position": 0, "artist": "A", "title": "T", "genres": []}],
            "metrics": {"mean_transition": 0.9, "min_transition": 0.8, "distinct_artists": 1}}},
        {"type": "done", "cmd": "generate_playlist", "ok": True, "detail": "Generated 1 tracks", "job_id": job_id},
    ]


def test_registry_tracks_job_to_success_with_playlist():
    reg = JobRegistry()
    jid = reg.create()
    for e in _gen_events(jid):
        reg.apply_event(e)
    job = reg.get(jid)
    assert job.status == "success"
    assert job.playlist.track_count == 1
    assert job.playlist.metrics.distinct_artists == 1
    assert reg.logs(jid)[0].endswith("starting")


def test_registry_marks_failure_with_error():
    reg = JobRegistry()
    jid = reg.create()
    reg.apply_event({"type": "error", "message": "boom", "job_id": jid})
    reg.apply_event({"type": "done", "cmd": "generate_playlist", "ok": False, "job_id": jid})
    job = reg.get(jid)
    assert job.status == "failed"
    assert job.error == "boom"


def test_recent_is_capped_and_newest_first():
    reg = JobRegistry(max_jobs=2)
    ids = [reg.create() for _ in range(3)]
    recent = reg.recent()
    assert len(recent) == 2
    assert recent[0].job_id == ids[-1]
