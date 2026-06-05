from __future__ import annotations

from typing import Any, Protocol


class _Sendable(Protocol):
    async def send_json(self, data: Any) -> None: ...


class WsHub:
    def __init__(self) -> None:
        self._clients: set[_Sendable] = set()

    async def connect(self, ws: _Sendable) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: _Sendable) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
