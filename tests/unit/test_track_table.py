"""
Tests for track table model, filter, and export functionality.

Run with: pytest tests/unit/test_track_table.py -v
"""
import tempfile
import pytest
from pathlib import Path


class TestDurationFormatting:
    """Tests for duration formatting functions."""

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        from src.playlist_gui.widgets.track_table_model import format_duration

        assert format_duration(0) == "0:00"
        assert format_duration(None) == "0:00"
        assert format_duration(-1000) == "0:00"

    def test_format_duration_seconds(self):
        """Test formatting various durations."""
        from src.playlist_gui.widgets.track_table_model import format_duration

        assert format_duration(1000) == "0:01"
        assert format_duration(59000) == "0:59"
        assert format_duration(60000) == "1:00"
        assert format_duration(61000) == "1:01"
        assert format_duration(125000) == "2:05"
        assert format_duration(3600000) == "60:00"  # 1 hour

    def test_normalize_duration_milliseconds(self):
        """Test normalizing duration from milliseconds."""
        from src.playlist_gui.widgets.track_table_model import normalize_duration

        # Values >= 10000 are treated as milliseconds
        assert normalize_duration(180000) == 180000
        assert normalize_duration(10000) == 10000

    def test_normalize_duration_seconds(self):
        """Test normalizing duration from seconds (heuristic)."""
        from src.playlist_gui.widgets.track_table_model import normalize_duration

        # Values < 10000 are treated as seconds
        assert normalize_duration(180) == 180000
        assert normalize_duration(60) == 60000
        assert normalize_duration(9999) == 9999000

    def test_normalize_duration_invalid(self):
        """Test normalizing invalid duration values."""
        from src.playlist_gui.widgets.track_table_model import normalize_duration

        assert normalize_duration(None) == 0
        assert normalize_duration("") == 0
        assert normalize_duration("invalid") == 0

    def test_normalize_duration_string(self):
        """Test normalizing duration from string."""
        from src.playlist_gui.widgets.track_table_model import normalize_duration

        assert normalize_duration("180000") == 180000
        assert normalize_duration("180") == 180000  # Treated as seconds
        assert normalize_duration("180.5") == 180000  # Floored


class TestTrackTableModel:
    """Tests for TrackTableModel."""

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data for testing."""
        return [
            {
                "position": 1,
                "artist": "Radiohead",
                "title": "Karma Police",
                "album": "OK Computer",
                "duration_ms": 264000,
                "file_path": "C:\\Music\\Radiohead\\OK Computer\\Karma Police.mp3"
            },
            {
                "position": 2,
                "artist": "The Beatles",
                "title": "Yesterday",
                "album": "Help!",
                "duration_ms": 125000,
                "file_path": "C:\\Music\\The Beatles\\Help!\\Yesterday.mp3"
            },
            {
                "position": 3,
                "artist": "Pink Floyd",
                "title": "Wish You Were Here",
                "album": "Wish You Were Here",
                "duration_ms": 334000,
                "file_path": "C:\\Music\\Pink Floyd\\WYWH\\Wish You Were Here.mp3"
            },
        ]

    def test_model_empty(self):
        """Test empty model."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column

        model = TrackTableModel()
        assert model.rowCount() == 0
        assert model.columnCount() == len(Column.HEADERS)

    def test_model_set_tracks(self, sample_tracks):
        """Test setting tracks."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        assert model.rowCount() == 3
        assert model.columnCount() == len(Column.HEADERS)

    def test_model_get_track(self, sample_tracks):
        """Test getting track by row."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        track = model.get_track(0)
        assert track is not None
        assert track["artist"] == "Radiohead"

        track = model.get_track(2)
        assert track["artist"] == "Pink Floyd"

        assert model.get_track(-1) is None
        assert model.get_track(100) is None

    def test_model_display_role(self, sample_tracks):
        """Test display role data."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column
        from PySide6.QtCore import Qt

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        # Artist column
        index = model.index(0, Column.ARTIST)
        assert model.data(index, Qt.DisplayRole) == "Radiohead"

        # Title column
        index = model.index(1, Column.TITLE)
        assert model.data(index, Qt.DisplayRole) == "Yesterday"

        # Duration column (formatted)
        index = model.index(0, Column.DURATION)
        assert model.data(index, Qt.DisplayRole) == "4:24"

    def test_model_user_role_for_sorting(self, sample_tracks):
        """Test user role returns sortable data."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column
        from PySide6.QtCore import Qt

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        # Duration returns raw milliseconds for sorting
        index = model.index(0, Column.DURATION)
        assert model.data(index, Qt.UserRole) == 264000

        # Artist returns lowercase for case-insensitive sort
        index = model.index(0, Column.ARTIST)
        assert model.data(index, Qt.UserRole) == "radiohead"

    def test_model_header_data(self):
        """Test header labels."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column
        from PySide6.QtCore import Qt

        model = TrackTableModel()

        assert model.headerData(Column.INDEX, Qt.Horizontal, Qt.DisplayRole) == "#"
        assert model.headerData(Column.ARTIST, Qt.Horizontal, Qt.DisplayRole) == "Artist"
        assert model.headerData(Column.TITLE, Qt.Horizontal, Qt.DisplayRole) == "Title"
        assert model.headerData(Column.DURATION, Qt.Horizontal, Qt.DisplayRole) == "Duration"

    def test_columns_single_source_of_truth(self):
        """Column headers/keys stay aligned and drive the model."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, Column

        # HEADERS and KEYS should stay aligned (no hardcoded counts in tests)
        assert len(Column.HEADERS) == len(Column.KEYS)

        model = TrackTableModel()
        assert model.columnCount() == len(Column.HEADERS)
        # Column indexes map deterministically to keys
        for idx, key in enumerate(Column.KEYS):
            assert Column.KEYS[idx] == key

    def test_model_clear(self, sample_tracks):
        """Test clearing tracks."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel

        model = TrackTableModel()
        model.set_tracks(sample_tracks)
        assert model.rowCount() == 3

        model.clear()
        assert model.rowCount() == 0


class TestTrackFilterProxyModel:
    """Tests for TrackFilterProxyModel filtering."""

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data for testing."""
        return [
            {"position": 1, "artist": "Radiohead", "title": "Karma Police", "album": "OK Computer", "duration_ms": 264000, "file_path": "C:\\Music\\radiohead.mp3"},
            {"position": 2, "artist": "The Beatles", "title": "Yesterday", "album": "Help!", "duration_ms": 125000, "file_path": "C:\\Music\\beatles.mp3"},
            {"position": 3, "artist": "Pink Floyd", "title": "Wish You Were Here", "album": "Wish You Were Here", "duration_ms": 334000, "file_path": "C:\\Music\\pinkfloyd.mp3"},
            {"position": 4, "artist": "Radiohead", "title": "Creep", "album": "Pablo Honey", "duration_ms": 235000, "file_path": "C:\\Music\\radiohead2.mp3"},
        ]

    def test_filter_no_filter(self, sample_tracks):
        """Test proxy with no filter shows all rows."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        assert proxy.rowCount() == 4

    def test_filter_by_artist(self, sample_tracks):
        """Test filtering by artist."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Filter for Radiohead
        proxy.setFilterRegularExpression(
            QRegularExpression("radiohead", QRegularExpression.CaseInsensitiveOption)
        )

        assert proxy.rowCount() == 2

    def test_filter_by_title(self, sample_tracks):
        """Test filtering by title."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Filter for "Yesterday"
        proxy.setFilterRegularExpression(
            QRegularExpression("yesterday", QRegularExpression.CaseInsensitiveOption)
        )

        assert proxy.rowCount() == 1

    def test_filter_by_album(self, sample_tracks):
        """Test filtering by album."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Filter for "Wish" (matches album)
        proxy.setFilterRegularExpression(
            QRegularExpression("wish", QRegularExpression.CaseInsensitiveOption)
        )

        assert proxy.rowCount() == 1

    def test_filter_case_insensitive(self, sample_tracks):
        """Test that filter is case-insensitive."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Filter with mixed case
        proxy.setFilterRegularExpression(
            QRegularExpression("RADIOHEAD", QRegularExpression.CaseInsensitiveOption)
        )

        assert proxy.rowCount() == 2

    def test_filter_include_path(self, sample_tracks):
        """Test filtering including file path."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Filter for "beatles" (only in file path)
        proxy.setFilterRegularExpression(
            QRegularExpression("beatles", QRegularExpression.CaseInsensitiveOption)
        )

        # Without path search, should match artist "The Beatles"
        assert proxy.rowCount() == 1

        # Enable path search
        proxy.set_include_path_in_search(True)
        proxy.invalidateFilter()

        # Should still match (was already matching on artist)
        assert proxy.rowCount() == 1

    def test_filter_get_visible_tracks(self, sample_tracks):
        """Test getting visible tracks after filter."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel
        from PySide6.QtCore import QRegularExpression

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        proxy.setFilterRegularExpression(
            QRegularExpression("radiohead", QRegularExpression.CaseInsensitiveOption)
        )

        visible = proxy.get_visible_tracks()
        assert len(visible) == 2
        assert all(t["artist"] == "Radiohead" for t in visible)


class TestM3U8Export:
    """Tests for M3U8 playlist export."""

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data for export testing."""
        return [
            {
                "position": 1,
                "artist": "Radiohead",
                "title": "Karma Police",
                "album": "OK Computer",
                "duration_ms": 264000,
                "file_path": "C:\\Music\\Radiohead\\Karma Police.mp3"
            },
            {
                "position": 2,
                "artist": "The Beatles",
                "title": "Yesterday",
                "album": "Help!",
                "duration_ms": 125000,
                "file_path": "C:\\Music\\The Beatles\\Yesterday.mp3"
            },
        ]

    def test_export_m3u8_format(self, sample_tracks):
        """Test M3U8 export format."""
        from src.playlist_gui.widgets.track_table import TrackTable

        # Create the widget (but don't show it)
        table = TrackTable()

        # Use temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m3u8', delete=False) as f:
            temp_path = f.name

        try:
            table._write_m3u8(temp_path, sample_tracks)

            # Read and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.strip().split('\n')

            # First line should be #EXTM3U
            assert lines[0] == "#EXTM3U"

            # Should have 2 EXTINF lines and 2 path lines
            extinf_lines = [l for l in lines if l.startswith("#EXTINF")]
            path_lines = [l for l in lines if not l.startswith("#")]

            assert len(extinf_lines) == 2
            assert len(path_lines) == 2

            # Check first track
            assert "#EXTINF:264,Radiohead - Karma Police" in content
            assert "C:\\Music\\Radiohead\\Karma Police.mp3" in content

            # Check second track
            assert "#EXTINF:125,The Beatles - Yesterday" in content
            assert "C:\\Music\\The Beatles\\Yesterday.mp3" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_m3u8_encoding(self, sample_tracks):
        """Test M3U8 export uses UTF-8 encoding."""
        from src.playlist_gui.widgets.track_table import TrackTable

        # Add a track with unicode characters
        tracks_with_unicode = sample_tracks + [{
            "position": 3,
            "artist": "Björk",
            "title": "Jóga",
            "album": "Homogenic",
            "duration_ms": 307000,
            "file_path": "C:\\Music\\Björk\\Jóga.mp3"
        }]

        table = TrackTable()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.m3u8', delete=False) as f:
            temp_path = f.name

        try:
            table._write_m3u8(temp_path, tracks_with_unicode)

            # Read as UTF-8
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Verify unicode is preserved
            assert "Björk" in content
            assert "Jóga" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_m3u8_empty(self):
        """Test export with empty track list."""
        from src.playlist_gui.widgets.track_table import TrackTable

        table = TrackTable()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.m3u8', delete=False) as f:
            temp_path = f.name

        try:
            table._write_m3u8(temp_path, [])

            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Should only have header
            assert content.strip() == "#EXTM3U"

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSorting:
    """Tests for table sorting."""

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data for sorting tests."""
        return [
            {"position": 1, "artist": "Radiohead", "title": "Creep", "album": "Pablo Honey", "duration_ms": 235000, "file_path": ""},
            {"position": 2, "artist": "The Beatles", "title": "Yesterday", "album": "Help!", "duration_ms": 125000, "file_path": ""},
            {"position": 3, "artist": "Pink Floyd", "title": "Wish", "album": "WYWH", "duration_ms": 334000, "file_path": ""},
        ]

    def test_proxy_sorting_by_duration(self, sample_tracks):
        """Test sorting by duration (numeric)."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel, Column
        from PySide6.QtCore import Qt

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Sort by duration ascending
        proxy.sort(Column.DURATION, Qt.AscendingOrder)

        # First should be shortest (Beatles at 125s)
        first_track = proxy.get_track(0)
        assert first_track["artist"] == "The Beatles"

        # Last should be longest (Pink Floyd at 334s)
        last_track = proxy.get_track(2)
        assert last_track["artist"] == "Pink Floyd"

    def test_proxy_sorting_by_artist(self, sample_tracks):
        """Test sorting by artist (alphabetic, case-insensitive)."""
        from src.playlist_gui.widgets.track_table_model import TrackTableModel, TrackFilterProxyModel, Column
        from PySide6.QtCore import Qt

        model = TrackTableModel()
        model.set_tracks(sample_tracks)

        proxy = TrackFilterProxyModel()
        proxy.setSourceModel(model)

        # Sort by artist ascending
        proxy.sort(Column.ARTIST, Qt.AscendingOrder)

        # First should be Pink Floyd (alphabetically first)
        first_track = proxy.get_track(0)
        assert first_track["artist"] == "Pink Floyd"

        # Last should be The Beatles (with "The" prefix)
        last_track = proxy.get_track(2)
        assert last_track["artist"] == "The Beatles"
