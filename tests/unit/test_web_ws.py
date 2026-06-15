# tests/unit/test_web_ws.py
import asyncio

import pytest

from src.playlist_web.ws import WsHub


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class StuckWS:
    """A client whose send never completes — a browser applying backpressure."""
    def __init__(self):
        self.gate = asyncio.Event()  # never set
        self.attempts = 0

    async def send_json(self, data):
        self.attempts += 1
        await self.gate.wait()


async def _wait_until(predicate, timeout=1.0):
    async def _loop():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(_loop(), timeout=timeout)


@pytest.mark.asyncio
async def test_hub_broadcasts_to_all_and_drops_disconnected():
    hub = WsHub()
    a, b = FakeWS(), FakeWS()
    await hub.connect(a)
    await hub.connect(b)
    await hub.broadcast({"type": "log", "msg": "hi"})
    await _wait_until(lambda: a.sent and b.sent)
    assert a.sent == [{"type": "log", "msg": "hi"}]
    hub.disconnect(b)
    await hub.broadcast({"type": "done"})
    await _wait_until(lambda: len(a.sent) == 2)
    await asyncio.sleep(0.02)  # give b a chance to (wrongly) receive
    assert len(a.sent) == 2
    assert len(b.sent) == 1


@pytest.mark.asyncio
async def test_broadcast_does_not_block_on_stuck_client():
    """THE regression: a stuck client must not block broadcast or starve others.

    This is the Genre Review 60s-timeout root cause — a browser applying WS
    backpressure stalled the bridge read loop, wedging the worker.
    """
    hub = WsHub()
    stuck, fast = StuckWS(), FakeWS()
    await hub.connect(stuck)
    await hub.connect(fast)
    # Must return promptly despite the stuck client (would hang on old WsHub).
    await asyncio.wait_for(hub.broadcast({"type": "log", "n": 1}), timeout=1.0)
    # And a healthy client keeps receiving.
    await _wait_until(lambda: fast.sent == [{"type": "log", "n": 1}])
    assert stuck.attempts >= 1  # delivery was attempted, just never blocks the hub


@pytest.mark.asyncio
async def test_stuck_client_never_blocks_under_flood():
    """A sustained flood to a stuck client stays bounded and never blocks."""
    hub = WsHub(max_queue=8)
    stuck, fast = StuckWS(), FakeWS()
    await hub.connect(stuck)
    await hub.connect(fast)
    for i in range(500):
        await asyncio.wait_for(hub.broadcast({"type": "progress", "i": i}), timeout=1.0)
    # Fast client still drains the latest; stuck client dropped excess, no hang.
    await _wait_until(lambda: len(fast.sent) >= 1)
    assert {"type": "progress", "i": 499} in fast.sent
