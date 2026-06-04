"""Test worker.enrich_artist command runs CLI pipeline via subprocess."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_handle_enrich_artist_runs_pipeline_steps_in_order():
    from src.playlist_gui.worker import handle_enrich_artist

    def fake_run(argv, **kwargs):
        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = ""
        if argv[2] == "discover":
            completed.stdout = (
                'Discovered 1 release(s).\n'
                '{"payload":{"artist":"Duster","album":"Stratosphere","normalized_album":"stratosphere"}}\n'
            )
        elif argv[2] == "hybrid-enrich-one":
            completed.stdout = '{"applied_count":2}\n'
        else:
            completed.stdout = ""
        return completed

    with patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run:
        result = handle_enrich_artist(artist="Duster", request_id="req-1")

    assert result["ok"] is True
    assert result["releases"] == 1
    assert result["applied"] == 2
    expected_commands = [
        "ingest-local",
        "extract-lastfm",
        "extract-bandcamp",
        "classify-tags",
        "discover",
        "hybrid-enrich-one",
    ]
    actual_commands = []
    for call in mock_run.call_args_list:
        argv = call.args[0]
        # argv = [sys.executable, "scripts/ai_genre_enrich.py", "<command>", ...]
        actual_commands.append(argv[2])
    assert actual_commands == expected_commands
    hybrid_argv = mock_run.call_args_list[-1].args[0]
    assert "--with-model-prior" in hybrid_argv
    assert "--include-provisional" in hybrid_argv
    assert "--apply" in hybrid_argv
    for call in mock_run.call_args_list[:3]:
        assert "--no-rebuild-signatures" in call.args[0]


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


def test_handle_enrich_artist_continues_after_single_release_hybrid_failure():
    from src.playlist_gui.worker import handle_enrich_artist

    def fake_run(argv, **kwargs):
        completed = MagicMock()
        completed.stderr = ""
        if argv[2] == "discover":
            completed.returncode = 0
            completed.stdout = (
                '{"payload":{"artist":"Mount Eerie","album":"A Crow Looked At Me"}}\n'
                '{"payload":{"artist":"Mount Eerie","album":"Duplicate Live Album"}}\n'
                '{"payload":{"artist":"Mount Eerie","album":"Sauna"}}\n'
            )
        elif argv[2] == "hybrid-enrich-one" and "Duplicate Live Album" in argv:
            completed.returncode = 2
            completed.stdout = "Expected exactly one release, found 2.\n"
        elif argv[2] == "hybrid-enrich-one":
            completed.returncode = 0
            completed.stdout = '{"applied_count":2}\n'
        else:
            completed.returncode = 0
            completed.stdout = ""
        return completed

    with patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run:
        result = handle_enrich_artist(artist="Mount Eerie", request_id="req-3")

    assert result["ok"] is True
    assert result["releases"] == 3
    assert result["applied"] == 4
    assert result["failed"] == 1
    hybrid_albums = [
        call.args[0][call.args[0].index("--album") + 1]
        for call in mock_run.call_args_list
        if call.args[0][2] == "hybrid-enrich-one"
    ]
    assert hybrid_albums == ["A Crow Looked At Me", "Duplicate Live Album", "Sauna"]


def test_handle_enrich_genres_album_runs_exact_release_pipeline():
    from src.playlist_gui.worker import handle_enrich_genres

    def fake_run(argv, **kwargs):
        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = ""
        completed.stdout = '{"applied_count":3}\n' if argv[2] == "hybrid-enrich-one" else ""
        return completed

    with patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run:
        result = handle_enrich_genres(scope="album", artist="Duster", album="Stratosphere")

    assert result["ok"] is True
    assert result["releases"] == 1
    assert result["applied"] == 3
    actual_commands = [call.args[0][2] for call in mock_run.call_args_list]
    assert actual_commands == [
        "ingest-local",
        "extract-lastfm",
        "extract-bandcamp",
        "classify-tags",
        "hybrid-enrich-one",
    ]
    for call in mock_run.call_args_list:
        argv = call.args[0]
        assert "--artist" in argv
        assert "--album" in argv
    for call in mock_run.call_args_list[:3]:
        assert "--no-rebuild-signatures" in call.args[0]


def test_handle_enrich_genres_full_scan_skips_already_enriched_releases():
    from src.playlist_gui.worker import handle_enrich_genres

    def fake_run(argv, **kwargs):
        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = ""
        if argv[2] == "discover":
            completed.stdout = (
                '{"release_key":"duster::stratosphere","payload":{"artist":"Duster","album":"Stratosphere"}}\n'
                '{"release_key":"duster::together","payload":{"artist":"Duster","album":"Together"}}\n'
            )
        elif argv[2] == "hybrid-enrich-one":
            completed.stdout = '{"applied_count":2}\n'
        else:
            completed.stdout = ""
        return completed

    with (
        patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run,
        patch("src.playlist_gui.worker._enriched_release_keys", return_value={"duster::stratosphere"}),
    ):
        result = handle_enrich_genres(scope="all_unenriched")

    assert result["ok"] is True
    assert result["releases"] == 1
    assert result["skipped_enriched"] == 1
    assert result["applied"] == 2
    actual_commands = [call.args[0][2] for call in mock_run.call_args_list]
    assert actual_commands == [
        "discover",
        "ingest-local",
        "extract-lastfm",
        "extract-bandcamp",
        "classify-tags",
        "hybrid-enrich-one",
    ]
    hybrid_argv = mock_run.call_args_list[-1].args[0]
    assert "Together" in hybrid_argv
    assert "Stratosphere" not in hybrid_argv
