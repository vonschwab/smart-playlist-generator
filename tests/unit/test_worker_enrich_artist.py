"""Test worker.enrich_artist command runs CLI pipeline via subprocess."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_handle_enrich_artist_runs_pipeline_steps_in_order():
    from src.playlist_gui.worker import handle_enrich_artist

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.playlist_gui.worker.subprocess.run", return_value=completed) as mock_run:
        result = handle_enrich_artist(artist="Duster", request_id="req-1")

    assert result["ok"] is True
    expected_commands = [
        "ingest-local",
        "extract-lastfm",
        "extract-bandcamp",
        "classify-tags",
        "build-enriched",
    ]
    actual_commands = []
    for call in mock_run.call_args_list:
        argv = call.args[0]
        # argv = [sys.executable, "scripts/ai_genre_enrich.py", "<command>", ...]
        actual_commands.append(argv[2])
    assert actual_commands == expected_commands


def test_handle_enrich_artist_stops_on_first_failure():
    from src.playlist_gui.worker import handle_enrich_artist

    def fake_run(argv, **kwargs):
        result = MagicMock()
        result.returncode = 0 if argv[2] == "ingest-local" else 1
        result.stdout = ""
        result.stderr = "boom"
        return result

    with patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run:
        result = handle_enrich_artist(artist="Duster", request_id="req-2")

    assert result["ok"] is False
    assert "extract-lastfm" in result.get("error", "")
    assert len(mock_run.call_args_list) == 2
