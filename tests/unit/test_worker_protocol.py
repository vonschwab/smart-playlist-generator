"""
Tests for worker protocol: request_id correlation and cancellation.

Run with: pytest tests/unit/test_worker_protocol.py -v
"""
import json
import logging
import pytest


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


class TestWorkerArtifactGenreSource:
    """Tests for GUI artifact-build genre source resolution."""

    def test_layered_runtime_builds_layered_shadow_artifact(self):
        from src.playlist_gui.worker import _resolve_worker_artifact_genre_source

        assert _resolve_worker_artifact_genre_source(
            {
                "playlists": {
                    "ds_pipeline": {
                        "genre_graph": {"source": "layered"},
                    }
                }
            }
        ) == "layered_shadow"

    def test_layered_shadow_runtime_builds_layered_shadow_artifact(self):
        from src.playlist_gui.worker import _resolve_worker_artifact_genre_source

        assert _resolve_worker_artifact_genre_source(
            {
                "playlists": {
                    "ds_pipeline": {
                        "genre_graph": {"source": "layered_shadow"},
                    }
                }
            }
        ) == "layered_shadow"

    def test_non_layered_runtime_honors_explicit_genre_source(self):
        from src.playlist_gui.worker import _resolve_worker_artifact_genre_source

        assert _resolve_worker_artifact_genre_source(
            {
                "playlists": {
                    "ds_pipeline": {
                        "genre_source": "hybrid_shadow",
                        "genre_graph": {"source": "legacy"},
                    }
                }
            }
        ) == "hybrid_shadow"

    def test_invalid_artifact_genre_source_falls_back_to_legacy(self):
        from src.playlist_gui.worker import _resolve_worker_artifact_genre_source

        assert _resolve_worker_artifact_genre_source(
            {"playlists": {"ds_pipeline": {"genre_source": "bad"}}}
        ) == "legacy"


class TestWorkerLogging:
    """Tests for worker log event filtering."""

    def test_worker_log_handler_remains_info_after_cli_logging_reconfigure(self, tmp_path, capsys):
        """CLI file logging must not make DEBUG library chatter flood the GUI stream."""
        import src.logging_utils as logging_utils
        from src.playlist_gui.worker import setup_worker_logging
        from src.logging_utils import configure_logging

        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_filters = list(root.filters)
        original_level = root.level
        original_configured = logging_utils._logging_configured

        try:
            setup_worker_logging()
            capsys.readouterr()

            configure_logging(
                level="INFO",
                log_file=str(tmp_path / "analyze_worker.log"),
                console=False,
                force=True,
            )

            noisy_logger = logging.getLogger("src.similarity_calculator")
            noisy_logger.debug("Filtered broad tag: rock")
            noisy_logger.info("Analyze run start")

            lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
            events = [json.loads(line) for line in lines]

            assert [event["level"] for event in events] == ["INFO"]
            assert events[0]["msg"] == "Analyze run start"
        finally:
            for handler in list(root.handlers):
                if handler not in original_handlers:
                    root.removeHandler(handler)
                    handler.close()
            root.handlers = original_handlers
            root.filters = original_filters
            root.setLevel(original_level)
            logging_utils._logging_configured = original_configured


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
