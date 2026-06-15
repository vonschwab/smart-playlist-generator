from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _Sendable(Protocol):
    async def send_json(self, data: Any) -> None: ...


class _ClientSender:
    """Per-client bounded queue + background sender.

    Broadcast NEVER blocks on a client: events are enqueued (oldest dropped on
    overflow) and delivered by this task. A stuck/slow client (a browser
    applying WS backpressure) can therefore never stall the bridge's stdout-read
    loop — the root cause of the Genre Review worker-wedge / 60s-timeout
    (2026-06-12). Dropped live events are acceptable: the GUI reconciles
    authoritative state via /api/jobs polling.
    """

    def __init__(self, ws: _Sendable, max_queue: int) -> None:
        self._ws = ws
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self.dropped = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def offer(self, message: dict) -> None:
        """Non-blocking enqueue; drop the oldest message if the client is behind."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(message)
            except asyncio.QueueEmpty:  # pragma: no cover - race only
                pass

    async def _run(self) -> None:
        try:
            while True:
                message = await self._queue.get()
                await self._ws.send_json(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Client died or send failed; stop delivering. The endpoint's
            # disconnect path removes us from the hub.
            logger.debug("ws client sender stopped", exc_info=True)

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


class WsHub:
    def __init__(self, *, max_queue: int = 256) -> None:
        self._max_queue = max_queue
        self._senders: dict[Any, _ClientSender] = {}

    async def connect(self, ws: _Sendable) -> None:
        sender = _ClientSender(ws, self._max_queue)
        self._senders[ws] = sender
        sender.start()

    def disconnect(self, ws: _Sendable) -> None:
        sender = self._senders.pop(ws, None)
        if sender is not None:
            sender.stop()

    async def broadcast(self, message: dict) -> None:
        """Fan out without ever blocking on a slow client (see _ClientSender)."""
        for sender in list(self._senders.values()):
            sender.offer(message)

    @property
    def client_count(self) -> int:
        return len(self._senders)
