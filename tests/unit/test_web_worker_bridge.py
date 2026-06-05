# tests/unit/test_web_worker_bridge.py
import asyncio
import sys

import pytest

from src.playlist_web.worker_bridge import WorkerBridge, BridgeBusy

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.mark.asyncio
async def test_bridge_streams_events_for_generate():
    events = []
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=lambda e: events.append(e) or asyncio.sleep(0))
    await bridge.start()
    assert bridge.running
    rid = await bridge.submit({"cmd": "generate_playlist", "job_id": "j1",
                               "base_config_path": "config.yaml", "overrides": {}, "args": {"mode": "artist", "artist": "Acetone", "tracks": 2}})
    # drain until a done event for this request arrives
    for _ in range(100):
        if any(e.get("type") == "done" and e.get("request_id") == rid for e in events):
            break
        await asyncio.sleep(0.02)
    await bridge.stop()
    types = [e.get("type") for e in events]
    assert "result" in types and "done" in types
    result = next(e for e in events if e.get("type") == "result")
    assert result["playlist"]["track_count"] == 2
    assert all(e.get("request_id") == rid for e in events)


@pytest.mark.asyncio
async def test_bridge_rejects_concurrent_submit():
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=lambda e: asyncio.sleep(0))
    await bridge.start()
    await bridge.submit({"cmd": "generate_playlist", "job_id": "j1", "args": {}})
    with pytest.raises(BridgeBusy):
        await bridge.submit({"cmd": "generate_playlist", "job_id": "j2", "args": {}})
    await bridge.stop()
