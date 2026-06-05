from __future__ import annotations

import asyncio
import json
import uuid
from typing import Awaitable, Callable, Optional

PROTOCOL_VERSION = 1
EventHandler = Callable[[dict], Awaitable[None]]


class BridgeBusy(RuntimeError):
    """Raised when a request is submitted while another is active."""


class WorkerCommandError(RuntimeError):
    """Raised when a synchronous worker command completes with ok=false."""


class WorkerBridge:
    """Asyncio NDJSON client for the playlist worker subprocess."""

    def __init__(self, worker_cmd: list[str], on_event: EventHandler):
        self._cmd = worker_cmd
        self._on_event = on_event
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._active_request_id: Optional[str] = None
        self._pending: dict[str, asyncio.Future] = {}
        self._results: dict[str, dict] = {}
        self._errors: dict[str, str] = {}

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def busy(self) -> bool:
        return self._active_request_id is not None

    async def start(self) -> None:
        if self.running:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def submit(self, cmd: dict) -> str:
        if not self.running:
            raise RuntimeError("Worker not running")
        if self.busy:
            raise BridgeBusy("Worker is busy with another request")
        request_id = str(uuid.uuid4())
        cmd = dict(cmd)
        cmd["request_id"] = request_id
        cmd["protocol_version"] = PROTOCOL_VERSION
        self._active_request_id = request_id
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        return request_id

    async def command(self, cmd: dict, timeout: float = 60.0) -> dict:
        """Submit a worker command and await its done event.

        Returns the captured `result` event payload. Raises BridgeBusy if the
        worker is already handling a request, or WorkerCommandError if the
        command completes with ok=false.
        """
        if not self.running:
            raise RuntimeError("Worker not running")
        if self.busy:
            raise BridgeBusy("Worker is busy with another request")
        request_id = str(uuid.uuid4())
        cmd = dict(cmd)
        cmd["request_id"] = request_id
        cmd["protocol_version"] = PROTOCOL_VERSION
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        self._active_request_id = request_id
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)
            self._results.pop(request_id, None)
            self._errors.pop(request_id, None)
            if self._active_request_id == request_id:
                self._active_request_id = None
            fut.cancel()

    async def cancel(self) -> bool:
        """Fire-and-forget cancel for the currently running request.

        Returns True if a cancel was dispatched, False if nothing was active.
        """
        if not (self._active_request_id and self._proc and self._proc.stdin):
            return False
        cmd = {"cmd": "cancel", "request_id": self._active_request_id}
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        return True

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                event = {"type": "log", "level": "INFO", "msg": text}
            etype = event.get("type")
            rid = event.get("request_id")
            if rid in self._pending:
                if etype == "result":
                    self._results[rid] = event
                elif etype == "error":
                    self._errors[rid] = event.get("message", "command failed")
                elif etype == "done":
                    fut = self._pending.get(rid)
                    if fut and not fut.done():
                        if event.get("ok"):
                            fut.set_result(self._results.get(rid, {}))
                        else:
                            msg = self._errors.get(rid) or event.get("detail") or "command failed"
                            fut.set_exception(WorkerCommandError(msg))
            if etype == "done" and rid == self._active_request_id:
                self._active_request_id = None
            await self._on_event(event)
