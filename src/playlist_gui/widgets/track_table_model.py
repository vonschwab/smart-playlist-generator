"""
Track Table Model - QAbstractTableModel for playlist tracks

Provides a proper model/view implementation for efficient display of playlist tracks.
Supports sorting by any column and role-based data access.
"""
from typing import Any, Dict, List, Optional

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)


class Column:
    """Column definitions for the track table."""
    INDEX = 0
    ARTIST = 1
    TITLE = 2
    ALBUM = 3
    DURATION = 4
    FILE_PATH = 5

    HEADERS = ["#", "Artist", "Title", "Album", "Duration", "File Path"]
    KEYS = ["position", "artist", "title", "album", "duration_ms", "file_path"]


def format_duration(ms: int) -> str:
    """
    Format milliseconds as M:SS.

    Args:
        ms: Duration in milliseconds

    Returns:
        Formatted string like "3:45"
    """
    if not ms or ms <= 0:
        return "0:00"
    seconds = ms // 1000
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def normalize_duration(value: Any) -> int:
    """
    Normalize duration to milliseconds.

    Handles both milliseconds and seconds input (heuristic: < 10000 = seconds).

    Args:
        value: Duration value (may be int, float, or string)

    Returns:
        Duration in milliseconds
    """
    if value is None:
        return 0

    try:
        num = int(float(value))
        # Heuristic: if under 10000, it's probably seconds
        if 0 < num < 10000:
            return num * 1000
        return num
    except (ValueError, TypeError):
        return 0


class TrackTableModel(QAbstractTableModel):
    """
    Table model for playlist tracks.

    Provides efficient data storage and retrieval for QTableView.
    Supports sorting via custom data roles.

    Usage:
        model = TrackTableModel()
        model.set_tracks(tracks_list)
        view.setModel(model)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: List[Dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return number of tracks."""
        if parent.isValid():
            return 0
        return len(self._tracks)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return number of columns."""
        if parent.isValid():
            return 0
        return len(Column.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """
        Return data for the given index and role.

        Supports:
            - DisplayRole: formatted text for display
            - UserRole: raw value for sorting
            - TextAlignmentRole: alignment
        """
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if row < 0 or row >= len(self._tracks):
            return None

        track = self._tracks[row]
        key = Column.KEYS[col]
        value = track.get(key, "")

        if role == Qt.DisplayRole:
            if col == Column.DURATION:
                # Format duration as M:SS
                ms = normalize_duration(value)
                return format_duration(ms)
            elif col == Column.INDEX:
                return str(value) if value else ""
            else:
                return str(value) if value else ""

        elif role == Qt.UserRole:
            # Raw value for sorting
            if col == Column.DURATION:
                return normalize_duration(value)
            elif col == Column.INDEX:
                try:
                    return int(value) if value else 0
                except (ValueError, TypeError):
                    return 0
            else:
                # String value, lowercase for case-insensitive sort
                return str(value).lower() if value else ""

        elif role == Qt.TextAlignmentRole:
            if col == Column.INDEX or col == Column.DURATION:
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        elif role == Qt.ToolTipRole:
            if col == Column.FILE_PATH:
                return str(value) if value else ""

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> Any:
        """Return header data."""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section < len(Column.HEADERS):
                return Column.HEADERS[section]
        return None

    def set_tracks(self, tracks: List[Dict[str, Any]]) -> None:
        """
        Set the tracks to display.

        Args:
            tracks: List of track dicts with keys: position, artist, title, album, duration_ms, file_path
        """
        self.beginResetModel()
        self._tracks = list(tracks)  # Copy the list
        self.endResetModel()

    def get_track(self, row: int) -> Optional[Dict[str, Any]]:
        """Get track data at the given row."""
        if 0 <= row < len(self._tracks):
            return self._tracks[row]
        return None

    def get_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks."""
        return list(self._tracks)

    def clear(self) -> None:
        """Clear all tracks."""
        self.beginResetModel()
        self._tracks = []
        self.endResetModel()


class TrackFilterProxyModel(QSortFilterProxyModel):
    """
    Proxy model for filtering and sorting tracks.

    Provides:
        - Case-insensitive filtering across artist/title/album (and optionally path)
        - Sorting by any column using UserRole data
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._include_path_in_search = False
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setSortRole(Qt.UserRole)

    def set_include_path_in_search(self, include: bool) -> None:
        """Toggle whether file path is included in filter matching."""
        if self._include_path_in_search != include:
            self._include_path_in_search = include
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """
        Determine if a row matches the current filter.

        Matches against artist, title, album, and optionally file path.
        """
        pattern = self.filterRegularExpression()
        if not pattern.isValid() or pattern.pattern() == "":
            return True

        model = self.sourceModel()
        if not model:
            return True

        # Check columns: Artist, Title, Album
        columns_to_check = [Column.ARTIST, Column.TITLE, Column.ALBUM]
        if self._include_path_in_search:
            columns_to_check.append(Column.FILE_PATH)

        for col in columns_to_check:
            index = model.index(source_row, col, source_parent)
            text = model.data(index, Qt.DisplayRole)
            if text and pattern.match(str(text)).hasMatch():
                return True

        return False

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """
        Compare two indices for sorting.

        Uses UserRole data for proper numeric/case-insensitive sorting.
        """
        left_data = self.sourceModel().data(left, Qt.UserRole)
        right_data = self.sourceModel().data(right, Qt.UserRole)

        # Handle None values
        if left_data is None and right_data is None:
            return False
        if left_data is None:
            return True
        if right_data is None:
            return False

        # Compare based on type
        if isinstance(left_data, (int, float)) and isinstance(right_data, (int, float)):
            return left_data < right_data

        return str(left_data) < str(right_data)

    def get_source_row(self, proxy_row: int) -> int:
        """Map a proxy row to the source model row."""
        proxy_index = self.index(proxy_row, 0)
        source_index = self.mapToSource(proxy_index)
        return source_index.row()

    def get_track(self, proxy_row: int) -> Optional[Dict[str, Any]]:
        """Get track data at the given proxy row."""
        source_row = self.get_source_row(proxy_row)
        source_model = self.sourceModel()
        if isinstance(source_model, TrackTableModel):
            return source_model.get_track(source_row)
        return None

    def get_visible_tracks(self) -> List[Dict[str, Any]]:
        """Get all visible (filtered) tracks."""
        tracks = []
        for row in range(self.rowCount()):
            track = self.get_track(row)
            if track:
                tracks.append(track)
        return tracks

    def get_visible_count(self) -> int:
        """Get the count of visible (filtered) tracks."""
        return self.rowCount()

    def get_total_count(self) -> int:
        """Get the total count of all tracks."""
        source = self.sourceModel()
        return source.rowCount() if source else 0
