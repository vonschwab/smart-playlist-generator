# tests/unit/test_web_ws.py
import pytest
from src.playlist_web.ws import WsHub


class FakeWS:
    def __init__(self):
        self.sent = []
    async def send_json(self, data):
        self.sent.append(data)


@pytest.mark.asyncio
async def test_hub_broadcasts_to_all_and_drops_disconnected():
    hub = WsHub()
    a, b = FakeWS(), FakeWS()
    await hub.connect(a)
    await hub.connect(b)
    await hub.broadcast({"type": "log", "msg": "hi"})
    assert a.sent == [{"type": "log", "msg": "hi"}]
    hub.disconnect(b)
    await hub.broadcast({"type": "done"})
    assert len(a.sent) == 2
    assert len(b.sent) == 1
