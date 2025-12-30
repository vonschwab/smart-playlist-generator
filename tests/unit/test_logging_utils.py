"""Tests for logging utilities."""
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.logging_utils import (
    configure_logging,
    stage_timer,
    redact,
    format_count,
    truncate_list,
    add_logging_args,
    resolve_log_level,
    RunSummary,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        # Reset logging state
        import src.logging_utils
        src.logging_utils._logging_configured = False

        configure_logging(level='DEBUG', force=True)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1

    def test_configure_logging_with_file(self):
        """Test logging configuration with file output."""
        import src.logging_utils
        src.logging_utils._logging_configured = False

        # Use a temp file that we manage ourselves to avoid Windows locking issues
        fd, log_file_path = tempfile.mkstemp(suffix='.log')
        os.close(fd)  # Close the file descriptor
        try:
            configure_logging(level='INFO', log_file=log_file_path, force=True)

            # Log something
            logger = logging.getLogger('test_file')
            logger.info('Test message')

            # Flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contains message
            with open(log_file_path, 'r') as f:
                content = f.read()
            assert 'Test message' in content
        finally:
            # Clean up: close handlers before deleting
            for handler in logging.getLogger().handlers[:]:
                if hasattr(handler, 'baseFilename') and handler.baseFilename == log_file_path:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
            try:
                os.unlink(log_file_path)
            except (PermissionError, OSError):
                pass  # May still be locked on Windows

    def test_configure_logging_idempotent(self):
        """Test that configure_logging is idempotent without force."""
        import src.logging_utils
        src.logging_utils._logging_configured = False

        configure_logging(level='INFO', force=True)
        handler_count = len(logging.getLogger().handlers)

        # Second call should not add handlers
        configure_logging(level='DEBUG')
        assert len(logging.getLogger().handlers) == handler_count


class TestStageTimer:
    """Tests for stage_timer context manager."""

    def test_stage_timer_completes_normally(self):
        """Test that stage_timer completes without error."""
        import io
        import src.logging_utils
        src.logging_utils._logging_configured = False

        # Create a string buffer to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        test_logger = logging.getLogger('test_stage_timer_norm')
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        with stage_timer("Test stage", logger=test_logger):
            pass

        log_output = log_capture.getvalue()
        assert 'Test stage completed' in log_output

        test_logger.removeHandler(handler)

    def test_stage_timer_completes_on_exception(self):
        """Test that stage_timer still logs even on exception."""
        import io
        import src.logging_utils
        src.logging_utils._logging_configured = False

        # Create a string buffer to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        test_logger = logging.getLogger('test_stage_timer_exc')
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        try:
            with stage_timer("Failing stage", logger=test_logger):
                raise ValueError("Test error")
        except ValueError:
            pass

        log_output = log_capture.getvalue()
        assert 'Failing stage completed' in log_output

        test_logger.removeHandler(handler)


class TestRedact:
    """Tests for redact function."""

    def test_redact_api_key(self):
        """Test that API keys matching default patterns are redacted."""
        # The default regex handles patterns like api_key=value
        text = 'api_key=secret123 other=visible'
        result = redact(text)
        assert 'secret123' not in result
        assert 'REDACTED' in result
        assert 'visible' in result  # Non-sensitive fields preserved

    def test_redact_token(self):
        """Test that tokens are redacted."""
        text = 'token=abc123xyz'
        result = redact(text)
        assert 'abc123xyz' not in result
        assert 'REDACTED' in result

    def test_redact_windows_path(self):
        """Test that Windows user paths are redacted."""
        text = r'C:\Users\JohnDoe\Documents\file.txt'
        result = redact(text)
        assert 'JohnDoe' not in result
        assert '***' in result

    def test_redact_unix_path(self):
        """Test that Unix user paths are redacted."""
        text = '/home/johndoe/documents/file.txt'
        result = redact(text)
        assert 'johndoe' not in result
        assert '***' in result

    def test_redact_path_object(self):
        """Test that Path objects on Windows are handled."""
        # On Windows, paths use backslashes so we test the Windows pattern
        text = r'C:\Users\TestUser\Documents\file.txt'
        result = redact(text)
        assert 'TestUser' not in result
        assert '***' in result

    def test_redact_with_custom_keys(self):
        """Test redaction with custom keys."""
        text = 'password=hunter2 username=bob'
        result = redact(text, keys=['password'])
        assert 'hunter2' not in result
        assert 'bob' in result  # username not in keys


class TestFormatCount:
    """Tests for format_count function."""

    def test_format_count_singular(self):
        """Test singular form."""
        assert format_count(1, 'track') == '1 track'

    def test_format_count_plural(self):
        """Test plural form."""
        assert format_count(5, 'track') == '5 tracks'

    def test_format_count_zero(self):
        """Test zero uses plural."""
        assert format_count(0, 'track') == '0 tracks'

    def test_format_count_custom_plural(self):
        """Test custom plural form."""
        assert format_count(5, 'category', 'categories') == '5 categories'

    def test_format_count_with_commas(self):
        """Test large numbers have commas."""
        assert format_count(1000, 'track') == '1,000 tracks'


class TestTruncateList:
    """Tests for truncate_list function."""

    def test_truncate_empty_list(self):
        """Test empty list."""
        assert truncate_list([]) == '(none)'

    def test_truncate_short_list(self):
        """Test list shorter than max."""
        result = truncate_list(['a', 'b'])
        assert result == 'a, b'

    def test_truncate_long_list(self):
        """Test list longer than max is truncated."""
        result = truncate_list(['a', 'b', 'c', 'd', 'e'], max_items=3)
        assert result == 'a, b, c (+2 more)'

    def test_truncate_with_format_fn(self):
        """Test with custom format function."""
        result = truncate_list([1, 2, 3], format_fn=lambda x: f'#{x}')
        assert result == '#1, #2, #3'


class TestLoggingArgs:
    """Tests for CLI argument helpers."""

    def test_add_logging_args(self):
        """Test that logging args are added to parser."""
        import argparse
        parser = argparse.ArgumentParser()
        add_logging_args(parser)

        # Parse with logging args
        args = parser.parse_args(['--log-level', 'DEBUG', '--log-file', 'test.log'])
        assert args.log_level == 'DEBUG'
        assert args.log_file == 'test.log'

    def test_resolve_log_level_default(self):
        """Test default log level resolution."""
        import argparse
        args = argparse.Namespace(debug=False, quiet=False, log_level='INFO')
        assert resolve_log_level(args) == 'INFO'

    def test_resolve_log_level_debug_flag(self):
        """Test --debug flag takes precedence."""
        import argparse
        args = argparse.Namespace(debug=True, quiet=False, log_level='INFO')
        assert resolve_log_level(args) == 'DEBUG'

    def test_resolve_log_level_quiet_flag(self):
        """Test --quiet flag takes precedence over log_level."""
        import argparse
        args = argparse.Namespace(debug=False, quiet=True, log_level='INFO')
        assert resolve_log_level(args) == 'WARNING'

    def test_resolve_log_level_debug_over_quiet(self):
        """Test --debug takes precedence over --quiet."""
        import argparse
        args = argparse.Namespace(debug=True, quiet=True, log_level='INFO')
        assert resolve_log_level(args) == 'DEBUG'


class TestRunSummary:
    """Tests for RunSummary class."""

    def test_run_summary_basic(self):
        """Test basic run summary logging."""
        import io
        import src.logging_utils
        src.logging_utils._logging_configured = False

        # Create a string buffer to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        test_logger = logging.getLogger('test_summary_basic')
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        summary = RunSummary("Test Run", logger=test_logger)
        summary.add("processed", 100)
        summary.add("failed", 5)
        summary.log()

        log_output = log_capture.getvalue()
        assert 'TEST RUN SUMMARY' in log_output
        assert 'Processed: 100' in log_output
        assert 'Failed: 5' in log_output

        test_logger.removeHandler(handler)

    def test_run_summary_increment(self):
        """Test increment method."""
        test_logger = logging.getLogger('test_increment')
        summary = RunSummary("Test", logger=test_logger)
        summary.increment("count")
        summary.increment("count")
        summary.increment("count", 3)
        assert summary.metrics["count"] == 5
