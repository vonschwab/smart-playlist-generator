"""Unit tests for duration filtering."""
import pytest
from src.playlist.filtering import filter_by_duration, is_valid_duration


class TestDurationFiltering:
    """Test hard min/max duration filtering."""

    def test_filter_by_duration_removes_short_tracks(self):
        """Tracks under 47s should be filtered out."""
        tracks = [
            {'rating_key': '1', 'duration': 26000},  # 26s - too short
            {'rating_key': '2', 'duration': 46000},  # 46s - too short
            {'rating_key': '3', 'duration': 47000},  # 47s - OK
            {'rating_key': '4', 'duration': 240000},  # 240s - OK
        ]

        filtered = filter_by_duration(tracks=tracks, min_duration_seconds=47, max_duration_seconds=720)

        assert len(filtered) == 2
        assert filtered[0]['rating_key'] == '3'
        assert filtered[1]['rating_key'] == '4'

    def test_filter_by_duration_removes_long_tracks(self):
        """Tracks over 720s should be filtered out."""
        tracks = [
            {'rating_key': '1', 'duration': 240000},  # 240s - OK
            {'rating_key': '2', 'duration': 720000},  # 720s - OK
            {'rating_key': '3', 'duration': 721000},  # 721s - too long
            {'rating_key': '4', 'duration': 900000},  # 900s - too long
        ]

        filtered = filter_by_duration(tracks=tracks, min_duration_seconds=47, max_duration_seconds=720)

        assert len(filtered) == 2
        assert filtered[0]['rating_key'] == '1'
        assert filtered[1]['rating_key'] == '2'

    def test_filter_by_duration_boundary_cases(self):
        """Boundary values should be handled correctly."""
        tracks = [
            {'rating_key': '1', 'duration': 46999},  # 46.999s - excluded
            {'rating_key': '2', 'duration': 47000},  # 47.0s - included
            {'rating_key': '3', 'duration': 720000},  # 720.0s - included
            {'rating_key': '4', 'duration': 720001},  # 720.001s - excluded
        ]

        filtered = filter_by_duration(tracks=tracks, min_duration_seconds=47, max_duration_seconds=720)

        assert len(filtered) == 2
        assert filtered[0]['rating_key'] == '2'
        assert filtered[1]['rating_key'] == '3'

    def test_filter_by_duration_missing_duration_excluded(self):
        """Tracks with missing or 0 duration should be excluded."""
        tracks = [
            {'rating_key': '1', 'duration': 0},
            {'rating_key': '2'},  # missing duration key
            {'rating_key': '3', 'duration': None},
            {'rating_key': '4', 'duration': 240000},  # OK
        ]

        filtered = filter_by_duration(tracks=tracks, min_duration_seconds=47, max_duration_seconds=720)

        assert len(filtered) == 1
        assert filtered[0]['rating_key'] == '4'

    def test_is_valid_duration_helper(self):
        """Test the is_valid_duration helper function."""
        assert is_valid_duration({'duration': 47000}, min_seconds=47, max_seconds=720) is True
        assert is_valid_duration({'duration': 240000}, min_seconds=47, max_seconds=720) is True
        assert is_valid_duration({'duration': 720000}, min_seconds=47, max_seconds=720) is True

        assert is_valid_duration({'duration': 26000}, min_seconds=47, max_seconds=720) is False
        assert is_valid_duration({'duration': 46999}, min_seconds=47, max_seconds=720) is False
        assert is_valid_duration({'duration': 720001}, min_seconds=47, max_seconds=720) is False
        assert is_valid_duration({'duration': 900000}, min_seconds=47, max_seconds=720) is False

        # Missing/invalid duration
        assert is_valid_duration({'duration': 0}, min_seconds=47, max_seconds=720) is False
        assert is_valid_duration({}, min_seconds=47, max_seconds=720) is False
        assert is_valid_duration({'duration': None}, min_seconds=47, max_seconds=720) is False


class TestDurationFilteringDefaults:
    """Test that default values are correct."""

    def test_default_min_is_47_seconds(self):
        """Default minimum should be 47 seconds."""
        tracks = [
            {'rating_key': '1', 'duration': 46000},  # excluded
            {'rating_key': '2', 'duration': 47000},  # included
        ]

        filtered = filter_by_duration(tracks=tracks)

        assert len(filtered) == 1
        assert filtered[0]['rating_key'] == '2'

    def test_default_max_is_720_seconds(self):
        """Default maximum should be 720 seconds (12 minutes)."""
        tracks = [
            {'rating_key': '1', 'duration': 720000},  # included
            {'rating_key': '2', 'duration': 721000},  # excluded
        ]

        filtered = filter_by_duration(tracks=tracks)

        assert len(filtered) == 1
        assert filtered[0]['rating_key'] == '1'
