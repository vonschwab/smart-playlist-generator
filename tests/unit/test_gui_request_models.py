"""Tests for shared GUI/worker request models."""

from src.playlist.request_models import (
    GeneratePlaylistRequest,
    LibraryOperationRequest,
    LibraryPipelineRequest,
)
from src.playlist_gui.jobs import JobType
from src.playlist_gui.request_models import GeneratePlaylistRequest as GuiGeneratePlaylistRequest


def test_gui_request_model_import_remains_compatible():
    assert GuiGeneratePlaylistRequest is GeneratePlaylistRequest


def test_generate_playlist_request_serializes_only_present_fields():
    request = GeneratePlaylistRequest(
        mode="genre",
        tracks=30,
        genre=" shoegaze ",
        genre_mode="off",
        sonic_mode="strict",
        pace_mode="narrow",
        seed_tracks=["", " Artist - Title "],
        seed_track_ids=[" 123 ", ""],
    )

    assert request.to_worker_args() == {
        "mode": "genre",
        "tracks": 30,
        "genre": "shoegaze",
        "seed_tracks": ["Artist - Title"],
        "seed_track_ids": ["123"],
        "genre_mode": "off",
        "sonic_mode": "strict",
        "pace_mode": "narrow",
    }


def test_generate_playlist_request_parses_worker_args_with_defaults():
    request = GeneratePlaylistRequest.from_worker_args(
        {
            "mode": "history",
            "tracks": "45",
            "seed_tracks": [" A - T ", ""],
            "seed_track_ids": [" 123 "],
            "include_collaborations": True,
            "exclude_seed_tracks_from_recency": True,
            "pace_mode": "strict",
        }
    )

    assert request.mode == "history"
    assert request.tracks == 45
    assert request.seed_tracks == ["A - T"]
    assert request.seed_track_ids == ["123"]
    assert request.include_collaborations is True
    assert request.exclude_seed_tracks_from_recency is True
    assert request.pace_mode == "strict"


def test_generate_playlist_request_serializes_seed_freshness_flag():
    request = GeneratePlaylistRequest(
        mode="artist",
        artist="The Strokes",
        exclude_seed_tracks_from_recency=True,
    )

    assert request.to_worker_args()["exclude_seed_tracks_from_recency"] is True


def test_generate_playlist_request_validates_mode_inputs():
    assert GeneratePlaylistRequest(mode="artist").validation_error() == (
        "Enter an artist before generating."
    )
    assert GeneratePlaylistRequest(mode="genre").validation_error() == (
        "Enter a genre before generating."
    )
    assert GeneratePlaylistRequest(mode="seeds").validation_error() == (
        "Add at least one seed track before generating."
    )
    assert GeneratePlaylistRequest(mode="history").validation_error() is None


def test_generate_playlist_request_builds_from_cli_artist_args():
    class Args:
        artist = "David Bowie"
        genre = None
        track = "Heroes"
        tracks = 30
        anchor_seed_ids = "1, 2,,3"
        artist_only = True

    request = GeneratePlaylistRequest.from_cli_args(
        Args,
        genre_mode="off",
        sonic_mode="strict",
        pace_mode="narrow",
    )

    assert request.mode == "artist"
    assert request.artist == "David Bowie"
    assert request.track == "Heroes"
    assert request.tracks == 30
    assert request.anchor_seed_ids == ["1", "2", "3"]
    assert request.artist_only is True
    assert request.genre_mode == "off"
    assert request.sonic_mode == "strict"
    assert request.pace_mode == "narrow"


def test_generate_playlist_request_builds_from_cli_genre_and_history_args():
    class GenreArgs:
        artist = None
        genre = "shoegaze"
        track = None
        tracks = 45
        anchor_seed_ids = None
        artist_only = False

    class HistoryArgs:
        artist = None
        genre = None
        track = None
        tracks = 20
        anchor_seed_ids = None
        artist_only = False

    genre_request = GeneratePlaylistRequest.from_cli_args(GenreArgs)
    history_request = GeneratePlaylistRequest.from_cli_args(HistoryArgs)

    assert genre_request.mode == "genre"
    assert genre_request.genre == "shoegaze"
    assert genre_request.tracks == 45
    assert history_request.mode == "history"
    assert history_request.tracks == 20


def test_library_operation_request_serializes_worker_command():
    request = LibraryOperationRequest(
        operation=JobType.UPDATE_SONIC,
        config_path="config.yaml",
        overrides={"library": {"database_path": "data/metadata.db"}},
    )

    assert request.to_worker_command() == {
        "cmd": "update_sonic",
        "base_config_path": "config.yaml",
        "overrides": {"library": {"database_path": "data/metadata.db"}},
    }


def test_library_pipeline_request_uses_ordered_operations():
    request = LibraryPipelineRequest(config_path="config.yaml", overrides={"force": True})

    operations = request.operations()

    assert [operation.operation for operation in operations] == [
        JobType.ANALYZE_LIBRARY.value,
    ]
    assert all(operation.config_path == "config.yaml" for operation in operations)
    assert all(operation.overrides == {"force": True} for operation in operations)


def test_library_pipeline_request_serializes_analyze_library_command():
    request = LibraryPipelineRequest(config_path="config.yaml", overrides={"force": True})

    assert request.to_worker_command() == {
        "cmd": "analyze_library",
        "base_config_path": "config.yaml",
        "overrides": {"force": True},
    }


def test_library_pipeline_request_serializes_stage_options():
    request = LibraryPipelineRequest(
        config_path="config.yaml",
        overrides={"playlists": {"count": 1}},
        stages=["sonic", "artifacts"],
        force=True,
        dry_run=True,
    )

    assert request.to_worker_command() == {
        "cmd": "analyze_library",
        "base_config_path": "config.yaml",
        "overrides": {"playlists": {"count": 1}},
        "stages": ["sonic", "artifacts"],
        "force": True,
        "dry_run": True,
    }
