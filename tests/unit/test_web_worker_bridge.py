# tests/unit/test_web_worker_bridge.py
import asyncio
import sys

import pytest

from src.playlist_web.worker_bridge import WorkerBridge, BridgeBusy, WorkerCommandError

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]

# A worker stub that emits a single result line LARGER than asyncio's 64KB
# StreamReader.readline() default limit. The real Genre Review limit=50 page is
# ~192KB on one line; that overran the limit, killed the bridge read loop, and
# every command timed out (the multi-day 504 bug, 2026-06-12).
_BIG_LINE_WORKER = [
    sys.executable, "-c",
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    c = json.loads(line); rid = c.get('request_id')\n"
    "    print(json.dumps({'type':'result','result_type':'big','request_id':rid,'blob':'x'*200000}), flush=True)\n"
    "    print(json.dumps({'type':'done','ok':True,'request_id':rid}), flush=True)\n",
]


@pytest.mark.asyncio
async def test_command_handles_result_line_over_64kb():
    bridge = WorkerBridge(_BIG_LINE_WORKER, on_event=_noop)
    await bridge.start()
    try:
        result = await bridge.command({"cmd": "big"}, timeout=10)
        assert len(result["blob"]) == 200000
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_bridge_streams_events_for_generate():
    events = []
    async def collect(e):
        events.append(e)
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=collect)
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
    async def noop(e):
        pass
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=noop)
    await bridge.start()
    await bridge.submit({"cmd": "generate_playlist", "job_id": "j1", "args": {}})
    with pytest.raises(BridgeBusy):
        await bridge.submit({"cmd": "generate_playlist", "job_id": "j2", "args": {}})
    await bridge.stop()


async def _noop(_e):
    pass


async def test_command_returns_result_payload():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        result = await bridge.command({"cmd": "ping"})
        assert result["result_type"] == "pong"
    finally:
        await bridge.stop()


async def test_command_raises_on_worker_error():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        with pytest.raises(WorkerCommandError):
            await bridge.command({"cmd": "does_not_exist"})
    finally:
        await bridge.stop()


async def test_command_rejects_when_busy():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        bridge._active_request_id = "someone-else"  # simulate in-flight request
        with pytest.raises(BridgeBusy):
            await bridge.command({"cmd": "ping"})
    finally:
        bridge._active_request_id = None
        await bridge.stop()


async def test_command_untracked_bypasses_busy():
    """Untracked commands (review queue/decision) run while a tracked job is busy.

    The worker handles these inline (UNTRACKED_COMMAND_HANDLERS) so they must not
    be gated by the bridge, and must not disturb the in-flight request's id.
    """
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        bridge._active_request_id = "scan-in-flight"  # a long tracked job holds the bridge
        result = await bridge.command({"cmd": "ping"}, untracked=True)
        assert result["result_type"] == "pong"
        # The tracked job's active id is untouched.
        assert bridge._active_request_id == "scan-in-flight"
    finally:
        bridge._active_request_id = None
        await bridge.stop()
