"""
Tests for worker protocol: request_id correlation and cancellation.

Run with: pytest tests/unit/test_worker_protocol.py -v
"""
import json
import pytest
import uuid
from unittest.mock import MagicMock, patch


class TestRequestIdGeneration:
    """Tests for request_id generation and inclusion in commands."""

    def test_generate_request_id_is_valid_uuid(self):
        """Test that generated request IDs are valid UUIDs."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        request_id = client._generate_request_id()

        # Should be a valid UUID string
        assert request_id is not None
        assert len(request_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

        # Should be parseable as UUID
        parsed = uuid.UUID(request_id)
        assert str(parsed) == request_id

    def test_send_command_includes_request_id(self):
        """Test that send_command adds request_id to command."""
        from src.playlist_gui.worker_client import WorkerClient, PROTOCOL_VERSION

        client = WorkerClient()
        client._running = True
        client._process = MagicMock()
        client._process.write = MagicMock(return_value=100)

        # Send a command
        request_id = client.send_command({"cmd": "ping"})

        # Should return request_id
        assert request_id is not None

        # Check the written data includes request_id
        call_args = client._process.write.call_args[0][0]
        sent_data = json.loads(call_args.decode('utf-8').strip())

        assert "request_id" in sent_data
        assert sent_data["request_id"] == request_id
        assert sent_data["protocol_version"] == PROTOCOL_VERSION

    def test_send_command_tracks_active_request(self):
        """Test that send_command tracks the active request."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._running = True
        client._process = MagicMock()
        client._process.write = MagicMock(return_value=100)

        assert client._active_request_id is None
        assert client.is_busy() is False

        request_id = client.send_command({"cmd": "ping"})

        assert client._active_request_id == request_id
        assert client.is_busy() is True

    def test_send_command_rejects_when_busy(self):
        """Test that send_command rejects new commands when busy."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._running = True
        client._process = MagicMock()
        client._process.write = MagicMock(return_value=100)

        # Simulate busy state
        client._busy = True
        client._active_request_id = "existing-request"

        # Should reject with None
        result = client.send_command({"cmd": "generate_playlist"})
        assert result is None


class TestEventFiltering:
    """Tests for event filtering by request_id."""

    def test_events_with_matching_request_id_are_processed(self):
        """Test that events with matching request_id are processed."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._active_request_id = "test-request-123"
        client._busy = True

        # Track emitted signals
        log_received = []
        client.log_received.connect(lambda level, msg, job_id: log_received.append((level, msg, job_id)))

        # Process an event with matching request_id
        event_line = json.dumps({
            "type": "log",
            "request_id": "test-request-123",
            "level": "INFO",
            "msg": "Test message"
        })

        client._process_event_line(event_line)

        assert len(log_received) == 1
        assert log_received[0] == ("INFO", "Test message", None)

    def test_log_signal_signature_enforced(self):
        """Signals must include job_id and reject wrong arity."""
        from src.playlist_gui.worker_client import WorkerClient
        import pytest

        client = WorkerClient()
        # Missing job_id should raise
        with pytest.raises(TypeError):
            client.log_received.emit("INFO", "msg")
        # Correct arity passes
        client.log_received.emit("INFO", "msg", None)

    def test_events_with_mismatched_request_id_are_filtered(self):
        """Test that events with different request_id are filtered out."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._active_request_id = "current-request"
        client._busy = True

        # Track emitted signals
        progress_received = []
        client.progress_received.connect(
            lambda stage, cur, tot, det, job_id: progress_received.append((stage, cur, tot, det, job_id))
        )

        # Process an event with different request_id (stale event)
        event_line = json.dumps({
            "type": "progress",
            "request_id": "old-request",
            "stage": "generate",
            "current": 50,
            "total": 100
        })

        client._process_event_line(event_line)

        # Should not emit progress signal (filtered as stale)
        assert len(progress_received) == 0

    def test_events_without_request_id_are_processed_legacy(self):
        """Test that events without request_id are processed (legacy mode)."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._active_request_id = "current-request"
        client._busy = True

        log_received = []
        client.log_received.connect(lambda level, msg, job_id: log_received.append((level, msg, job_id)))

        # Process a legacy event without request_id
        event_line = json.dumps({
            "type": "log",
            "level": "WARNING",
            "msg": "Legacy message"
        })

        client._process_event_line(event_line)

        # Should still be processed
        assert len(log_received) == 1
        assert log_received[0] == ("WARNING", "Legacy message", None)

    def test_done_event_clears_active_request(self):
        """Test that done event clears the active request."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._active_request_id = "test-request"
        client._active_cmd = "generate_playlist"
        client._busy = True

        done_received = []
        client.done_received.connect(
            lambda cmd, ok, detail, cancelled, job_id, summary: done_received.append(
                (cmd, ok, cancelled, job_id, summary)
            )
        )

        # Process done event
        event_line = json.dumps({
            "type": "done",
            "request_id": "test-request",
            "cmd": "generate_playlist",
            "ok": True,
            "detail": "Generated 30 tracks"
        })

        client._process_event_line(event_line)

        # Should clear active request
        assert client._active_request_id is None
        assert client.is_busy() is False
        assert len(done_received) == 1
        assert done_received[0] == ("generate_playlist", True, False, None, "")

    def test_done_signal_signature_enforced(self):
        """Done signal must include job_id and summary."""
        from src.playlist_gui.worker_client import WorkerClient
        import pytest

        client = WorkerClient()
        with pytest.raises(TypeError):
            client.done_received.emit("cmd", True, "detail", False)
        # Correct arity
        client.done_received.emit("cmd", True, "detail", False, None, "")


class TestCancellation:
    """Tests for cancellation functionality."""

    def test_cancel_sends_cancel_command(self):
        """Test that cancel() sends a cancel command."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._running = True
        client._process = MagicMock()
        client._process.write = MagicMock(return_value=100)
        client._active_request_id = "request-to-cancel"
        client._busy = True

        result = client.cancel()

        assert result is True
        call_args = client._process.write.call_args[0][0]
        sent_data = json.loads(call_args.decode('utf-8').strip())

        assert sent_data["cmd"] == "cancel"
        assert sent_data["request_id"] == "request-to-cancel"

    def test_cancel_returns_false_when_no_active_request(self):
        """Test that cancel() returns False when no active request."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._running = True
        client._active_request_id = None

        result = client.cancel()
        assert result is False

    def test_done_event_with_cancelled_flag(self):
        """Test that done event with cancelled=true is handled correctly."""
        from src.playlist_gui.worker_client import WorkerClient

        client = WorkerClient()
        client._active_request_id = "cancelled-request"
        client._busy = True

        done_received = []
        client.done_received.connect(
            lambda cmd, ok, detail, cancelled, job_id, summary: done_received.append(
                (cmd, ok, cancelled, job_id, summary)
            )
        )

        # Process cancelled done event
        event_line = json.dumps({
            "type": "done",
            "request_id": "cancelled-request",
            "cmd": "generate_playlist",
            "ok": False,
            "detail": "Cancelled by user",
            "cancelled": True
        })

        client._process_event_line(event_line)

        assert len(done_received) == 1
        cmd, ok, cancelled, job_id, summary = done_received[0]
        assert cmd == "generate_playlist"
        assert ok is False
        assert cancelled is True
        assert job_id is None
        assert summary == ""


class TestWorkerState:
    """Tests for worker state management."""

    def test_worker_state_start_request(self):
        """Test starting a new request."""
        from src.playlist_gui.worker import WorkerState

        state = WorkerState()
        state.start_request("test-id", "generate_playlist", job_id=None)

        assert state.current_request_id == "test-id"
        assert state.current_cmd == "generate_playlist"
        assert state.cancel_requested is False

    def test_worker_state_cancel_request(self):
        """Test cancelling an active request."""
        from src.playlist_gui.worker import WorkerState

        state = WorkerState()
        state.start_request("test-id", "generate_playlist", job_id=None)

        # Cancel with matching ID should succeed
        result = state.request_cancel("test-id")
        assert result is True
        assert state.is_cancelled() is True

    def test_worker_state_cancel_wrong_request(self):
        """Test cancelling with wrong request_id."""
        from src.playlist_gui.worker import WorkerState

        state = WorkerState()
        state.start_request("test-id", "generate_playlist", job_id=None)

        # Cancel with different ID should fail
        result = state.request_cancel("wrong-id")
        assert result is False
        assert state.is_cancelled() is False

    def test_worker_state_check_cancelled_raises(self):
        """Test that check_cancelled raises CancellationError."""
        from src.playlist_gui.worker import WorkerState, CancellationError

        state = WorkerState()
        state.start_request("test-id", "generate_playlist", job_id=None)
        state.request_cancel("test-id")

        with pytest.raises(CancellationError):
            state.check_cancelled()

    def test_worker_state_end_request_clears_state(self):
        """Test that end_request clears all state."""
        from src.playlist_gui.worker import WorkerState

        state = WorkerState()
        state.start_request("test-id", "generate_playlist", job_id=None)
        state.request_cancel("test-id")
        state.end_request()

        assert state.current_request_id is None
        assert state.current_cmd is None
        assert state.cancel_requested is False


class TestSecretRedaction:
    """Tests for secret redaction in worker."""

    def test_redact_secrets_in_text(self):
        """Test that secrets are redacted from text."""
        from src.playlist_gui.worker import redact_secrets_in_text

        text_with_secrets = "api_key=sk-secret-key-123 and token=abc123"
        redacted = redact_secrets_in_text(text_with_secrets)

        assert "sk-secret-key-123" not in redacted
        assert "abc123" not in redacted
        assert "***REDACTED***" in redacted

    def test_redact_secrets_preserves_non_secrets(self):
        """Test that non-secret text is preserved."""
        from src.playlist_gui.worker import redact_secrets_in_text

        text = "Loading configuration from config.yaml"
        redacted = redact_secrets_in_text(text)

        assert redacted == text
