"""
Worker Client - Manages communication with the worker process

Uses QProcess to spawn the worker and communicates via NDJSON protocol.
All events are routed to Qt signals for thread-safe UI updates.

Protocol Version: 1
  - All commands include request_id for correlation
  - All events include request_id to match the originating command
  - Cancel command for cooperative cancellation
  - Busy state management (single active job)
"""
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Slot


# Protocol version
PROTOCOL_VERSION = 1


class WorkerClient(QObject):
    """
    Client for communicating with the worker process.

    Features:
        - Request ID correlation for all commands/events
        - Busy state tracking (single active job)
        - Cancellation support with escalation timer
        - Event filtering by request_id

    Signals:
        log_received: Emitted when a log event is received (level, message)
        progress_received: Emitted when a progress event is received (stage, current, total, detail)
        result_received: Emitted when a result event is received (result_type, data)
        error_received: Emitted when an error event is received (message, traceback)
        done_received: Emitted when a done event is received (cmd, ok, detail, cancelled)
        worker_started: Emitted when the worker process starts
        worker_stopped: Emitted when the worker process stops (exit_code, exit_status)
        busy_changed: Emitted when busy state changes (is_busy)

    Usage:
        client = WorkerClient()
        client.log_received.connect(on_log)
        client.busy_changed.connect(on_busy_changed)
        client.start()
        request_id = client.generate_playlist(...)
    """

    # Signals for event routing
    log_received = Signal(str, str)  # level, message
    progress_received = Signal(str, int, int, str)  # stage, current, total, detail
    result_received = Signal(str, dict)  # result_type, data
    error_received = Signal(str, str)  # message, traceback
    done_received = Signal(str, bool, str, bool)  # cmd, ok, detail, cancelled
    worker_started = Signal()
    worker_stopped = Signal(int, str)  # exit_code, exit_status
    busy_changed = Signal(bool)  # is_busy

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._buffer = ""
        self._running = False

        # Request tracking
        self._active_request_id: Optional[str] = None
        self._active_cmd: Optional[str] = None
        self._busy = False

        # Cancellation escalation
        self._cancel_timer: Optional[QTimer] = None
        self._cancel_grace_ms = 5000  # 5 seconds grace period

    def is_running(self) -> bool:
        """Check if the worker process is running."""
        return self._running and self._process is not None

    def is_busy(self) -> bool:
        """Check if there is an active request being processed."""
        return self._busy

    def get_active_request_id(self) -> Optional[str]:
        """Get the current active request ID, if any."""
        return self._active_request_id

    def _set_busy(self, busy: bool) -> None:
        """Set busy state and emit signal if changed."""
        if self._busy != busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def start(self) -> bool:
        """
        Start the worker process.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            return True

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)

        # Connect signals
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        # Find the worker module
        worker_module = self._find_worker_path()

        # Start the worker process
        python_exe = sys.executable
        self._process.start(python_exe, ["-m", worker_module])

        # Wait briefly for startup
        if self._process.waitForStarted(5000):
            self._running = True
            return True

        return False

    def stop(self) -> None:
        """Stop the worker process gracefully."""
        if not self._running or not self._process:
            return

        # Close stdin to signal EOF
        self._process.closeWriteChannel()

        # Wait for graceful shutdown
        if not self._process.waitForFinished(3000):
            # Force kill if not responding
            self._process.kill()
            self._process.waitForFinished(1000)

        self._running = False

    def send_command(self, cmd_data: Dict[str, Any], track_request: bool = True) -> Optional[str]:
        """
        Send a command to the worker.

        Args:
            cmd_data: Command dictionary with at least a 'cmd' key.
            track_request: If True, generate request_id and track as active.

        Returns:
            The request_id if successful, None otherwise.
        """
        if not self._running or not self._process:
            self.error_received.emit("Worker not running", "")
            return None

        # Check if busy (single active job MVP)
        if track_request and self._busy:
            self.error_received.emit("Worker is busy with another request", "")
            return None

        # Generate and add request_id
        request_id = None
        if track_request:
            request_id = self._generate_request_id()
            cmd_data["request_id"] = request_id
            cmd_data["protocol_version"] = PROTOCOL_VERSION

            # Track this request
            self._active_request_id = request_id
            self._active_cmd = cmd_data.get("cmd")
            self._set_busy(True)

        try:
            line = json.dumps(cmd_data) + "\n"
            data = line.encode("utf-8")
            bytes_written = self._process.write(data)
            # QProcess.write returns -1 on failure; any non-negative value means
            # the bytes were accepted for writing.
            if bytes_written < 0:
                if track_request:
                    self._clear_active_request()
                return None
            return request_id
        except Exception as e:
            self.error_received.emit(f"Failed to send command: {e}", "")
            if track_request:
                self._clear_active_request()
            return None

    def _clear_active_request(self) -> None:
        """Clear the active request state."""
        self._active_request_id = None
        self._active_cmd = None
        self._set_busy(False)
        self._stop_cancel_timer()

    def ping(self) -> Optional[str]:
        """Send a ping command to test worker connectivity."""
        return self.send_command({"cmd": "ping"})

    def cancel(self, request_id: Optional[str] = None) -> bool:
        """
        Request cancellation of the current or specified request.

        Args:
            request_id: Specific request to cancel, or None for active request.

        Returns:
            True if cancel command was sent, False otherwise.
        """
        target_id = request_id or self._active_request_id
        if not target_id:
            self.log_received.emit("WARNING", "No active request to cancel")
            return False

        if not self._running or not self._process:
            self.log_received.emit("WARNING", "Worker not running")
            return False

        # Send cancel command (untracked - doesn't set busy)
        try:
            cmd_data = {"cmd": "cancel", "request_id": target_id}
            line = json.dumps(cmd_data) + "\n"
            data = line.encode("utf-8")
            bytes_written = self._process.write(data)
            if bytes_written >= 0:
                self.log_received.emit("INFO", f"Cancellation requested for {target_id[:8]}...")
                self._start_cancel_timer()
                return True
            return False
        except Exception as e:
            self.error_received.emit(f"Failed to send cancel: {e}", "")
            return False

    def _start_cancel_timer(self) -> None:
        """Start the cancellation escalation timer."""
        self._stop_cancel_timer()
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(self._on_cancel_timeout)
        self._cancel_timer.start(self._cancel_grace_ms)

    def _stop_cancel_timer(self) -> None:
        """Stop the cancellation escalation timer."""
        if self._cancel_timer:
            self._cancel_timer.stop()
            self._cancel_timer.deleteLater()
            self._cancel_timer = None

    @Slot()
    def _on_cancel_timeout(self) -> None:
        """Handle cancel timeout - escalate to process kill."""
        if not self._busy:
            return  # Already completed

        self.log_received.emit("WARNING", "Worker not responding to cancel, terminating...")

        # Force kill the worker
        if self._process:
            self._process.kill()
            self._process.waitForFinished(1000)

        # Clear state
        self._clear_active_request()

        # Auto-restart the worker
        self.log_received.emit("INFO", "Restarting worker process...")
        self.start()

    def force_kill(self) -> None:
        """Force kill the worker process immediately."""
        if self._process:
            self._process.kill()
            self._process.waitForFinished(1000)
        self._clear_active_request()
        self._running = False

    def generate_playlist(
        self,
        config_path: str,
        overrides: Dict[str, Any],
        mode: str = "history",
        artist: Optional[str] = None,
        track: Optional[str] = None,
        tracks: int = 30
    ) -> Optional[str]:
        """
        Send a generate_playlist command.

        Args:
            config_path: Path to base config YAML
            overrides: Override values to merge
            mode: "history" or "artist"
            artist: Artist name (required if mode is "artist")
            track: Optional seed track title
            tracks: Number of tracks to generate

        Returns:
            The request_id if sent successfully, None otherwise
        """
        args = {"mode": mode, "tracks": tracks}
        if artist:
            args["artist"] = artist
        if track:
            args["track"] = track

        return self.send_command({
            "cmd": "generate_playlist",
            "base_config_path": config_path,
            "overrides": overrides,
            "args": args
        })

    def scan_library(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Send a scan_library command. Returns request_id if successful."""
        return self.send_command({
            "cmd": "scan_library",
            "base_config_path": config_path,
            "overrides": overrides or {}
        })

    def update_genres(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Send an update_genres command. Returns request_id if successful."""
        return self.send_command({
            "cmd": "update_genres",
            "base_config_path": config_path,
            "overrides": overrides or {}
        })

    def update_sonic(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Send an update_sonic command. Returns request_id if successful."""
        return self.send_command({
            "cmd": "update_sonic",
            "base_config_path": config_path,
            "overrides": overrides or {}
        })

    def build_artifacts(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Send a build_artifacts command. Returns request_id if successful."""
        return self.send_command({
            "cmd": "build_artifacts",
            "base_config_path": config_path,
            "overrides": overrides or {}
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _find_worker_path(self) -> str:
        """Find the worker module path."""
        # The worker is at src/playlist_gui/worker.py
        # It can be run as: python -m src.playlist_gui.worker
        return "src.playlist_gui.worker"

    @Slot()
    def _on_stdout_ready(self) -> None:
        """Handle stdout data from the worker."""
        if not self._process:
            return

        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")

        # Buffer partial lines
        self._buffer += text

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_event_line(line)

    @Slot()
    def _on_stderr_ready(self) -> None:
        """Handle stderr data from the worker (log as warnings)."""
        if not self._process:
            return

        data = self._process.readAllStandardError()
        text = bytes(data).decode("utf-8", errors="replace")

        # Emit stderr as warning logs
        for line in text.strip().split("\n"):
            if line.strip():
                self.log_received.emit("WARNING", f"[stderr] {line}")

    @Slot()
    def _on_started(self) -> None:
        """Handle worker process start."""
        self._running = True
        self.worker_started.emit()

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Handle worker process finish."""
        self._running = False
        status_str = "normal" if exit_status == QProcess.NormalExit else "crashed"
        self.worker_stopped.emit(exit_code, status_str)

    @Slot(QProcess.ProcessError)
    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Handle process errors."""
        error_messages = {
            QProcess.FailedToStart: "Failed to start worker process",
            QProcess.Crashed: "Worker process crashed",
            QProcess.Timedout: "Worker process timed out",
            QProcess.WriteError: "Error writing to worker",
            QProcess.ReadError: "Error reading from worker",
            QProcess.UnknownError: "Unknown worker error",
        }
        msg = error_messages.get(error, "Unknown error")
        self.error_received.emit(msg, "")

    def _process_event_line(self, line: str) -> None:
        """Parse and route an event line with request_id filtering."""
        line = line.strip()
        if not line:
            return

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Not JSON, treat as raw log
            self.log_received.emit("INFO", line)
            return

        event_type = event.get("type")
        if not event_type:
            return

        # Check request_id correlation
        event_request_id = event.get("request_id")

        # Filter events: only process if request_id matches or is missing (legacy)
        if event_request_id and self._active_request_id:
            if event_request_id != self._active_request_id:
                # Stale event from a previous request - ignore
                self.log_received.emit(
                    "DEBUG",
                    f"Ignoring stale event for {event_request_id[:8]}..."
                )
                return

        if event_type == "log":
            self.log_received.emit(
                event.get("level", "INFO"),
                event.get("msg", "")
            )
        elif event_type == "progress":
            self.progress_received.emit(
                event.get("stage", ""),
                event.get("current", 0),
                event.get("total", 100),
                event.get("detail", "")
            )
        elif event_type == "result":
            result_type = event.get("result_type", "unknown")
            # Remove type and result_type, pass rest as data
            data = {k: v for k, v in event.items() if k not in ("type", "result_type", "request_id")}
            self.result_received.emit(result_type, data)
        elif event_type == "error":
            self.error_received.emit(
                event.get("message", "Unknown error"),
                event.get("traceback", "")
            )
        elif event_type == "done":
            cancelled = event.get("cancelled", False)
            self.done_received.emit(
                event.get("cmd", ""),
                event.get("ok", False),
                event.get("detail", ""),
                cancelled
            )
            # Clear active request when done
            if event_request_id == self._active_request_id or not event_request_id:
                self._clear_active_request()
