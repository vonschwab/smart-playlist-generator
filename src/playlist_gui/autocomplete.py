"""
Autocomplete Provider - Provides predictive text for artists and tracks

Queries the metadata database to provide autocomplete suggestions.
Uses Qt's QCompleter for integration with line edits.
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QStringListModel, Signal, QObject
from PySide6.QtWidgets import QCompleter, QLineEdit
from src.artist_key_db import ensure_artist_key_schema
from src.string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


class DatabaseCompleter(QObject):
    """
    Provides autocomplete data from the metadata database.

    Caches artist and track lists for fast completion.
    """

    data_loaded = Signal()

    def __init__(self, db_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._db_path = db_path or "data/metadata.db"
        self._artists: List[str] = []
        self._artist_entries: List[Tuple[str, str]] = []  # (display, artist_key)
        self._tracks: List[Tuple[str, str, str]] = []  # (title, artist, album)
        self._track_display: List[str] = []  # "Title - Artist" format for display
        self._track_artist_keys: List[str] = []
        self._loaded = False

    def set_database(self, db_path: str) -> None:
        """Set the database path and reload data."""
        self._db_path = db_path
        self._loaded = False
        self.load_data()

    def load_data(self) -> bool:
        """Load artist and track data from the database."""
        if not Path(self._db_path).exists():
            return False

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            ensure_artist_key_schema(conn)

            # Load distinct artists (sorted, deduplicated)
            cursor.execute("""
                SELECT DISTINCT artist, artist_key
                FROM tracks
                WHERE artist IS NOT NULL AND artist != ''
                ORDER BY artist COLLATE NOCASE
            """)
            self._artist_entries = []
            for row in cursor.fetchall():
                artist = row[0]
                artist_key = row[1] or normalize_artist_key(artist)
                if artist:
                    self._artist_entries.append((artist, artist_key))
            self._artists = [artist for artist, _ in self._artist_entries]

            # Load tracks with artist and album
            cursor.execute("""
                SELECT title, artist, album, artist_key
                FROM tracks
                WHERE title IS NOT NULL AND title != ''
                ORDER BY title COLLATE NOCASE
                LIMIT 50000
            """)
            rows = cursor.fetchall()
            self._tracks = [(row[0], row[1], row[2]) for row in rows]
            self._track_artist_keys = [
                row[3] or normalize_artist_key(row[1]) for row in rows
            ]

            # Create display format for tracks
            self._track_display = []
            for title, artist, album in self._tracks:
                if artist:
                    display = f"{title} - {artist}"
                    if album:
                        display += f" ({album})"
                else:
                    display = title
                self._track_display.append(display)

            conn.close()
            self._loaded = True
            self.data_loaded.emit()
            return True

        except Exception as e:
            logger.warning("Failed to load autocomplete data: %s", e)
            return False

    def get_artists(self) -> List[str]:
        """Get list of all artists."""
        if not self._loaded:
            self.load_data()
        return self._artists

    def filter_artists(self, query: str) -> List[str]:
        """Filter artists by normalized query against artist_key."""
        if not self._loaded:
            self.load_data()
        key = normalize_artist_key(query)
        if not key:
            return self._artists
        return [artist for artist, artist_key in self._artist_entries if key in artist_key]

    def get_tracks(self) -> List[str]:
        """Get list of track display strings."""
        if not self._loaded:
            self.load_data()
        return self._track_display

    def get_track_info(self, display: str) -> Optional[Tuple[str, str, str]]:
        """Get (title, artist, album) from a display string."""
        if not self._loaded:
            return None

        try:
            idx = self._track_display.index(display)
            return self._tracks[idx]
        except (ValueError, IndexError):
            return None

    def get_artist_tracks(self, artist: str) -> List[str]:
        """Get track display strings for a specific artist."""
        if not self._loaded:
            self.load_data()

        artist_key = normalize_artist_key(artist)
        if not artist_key:
            return []
        result = []
        for i, key in enumerate(self._track_artist_keys):
            if key == artist_key:
                result.append(self._track_display[i])
        return result

    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    @property
    def artist_count(self) -> int:
        """Get number of artists."""
        return len(self._artists)

    @property
    def track_count(self) -> int:
        """Get number of tracks."""
        return len(self._tracks)


class CaseInsensitiveCompleter(QCompleter):
    """
    Case-insensitive completer that matches anywhere in the string.
    """

    def __init__(self, items: List[str], parent=None):
        self._model = QStringListModel(items)
        super().__init__(self._model, parent)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchContains)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setMaxVisibleItems(15)

    def update_items(self, items: List[str]) -> None:
        """Update the completer items."""
        self._model.setStringList(items)


def setup_artist_completer(line_edit: QLineEdit, completer_data: DatabaseCompleter) -> QCompleter:
    """
    Setup autocomplete for an artist input field.

    Args:
        line_edit: The QLineEdit to attach the completer to
        completer_data: The DatabaseCompleter with loaded data

    Returns:
        The configured QCompleter
    """
    completer = CaseInsensitiveCompleter(completer_data.get_artists(), line_edit)
    # We do our own normalized filtering, so keep the popup unfiltered to avoid
    # Qt's accent-sensitive matching from hiding valid suggestions.
    completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
    line_edit.setCompleter(completer)

    def refresh(query: str) -> None:
        # The completer can be deleted/replaced if the field is re-parented;
        # grab the current instance off the line edit and guard before use.
        comp = line_edit.completer()
        if not comp:
            return
        try:
            if isinstance(comp, CaseInsensitiveCompleter):
                comp.update_items(completer_data.filter_artists(query))
            comp.complete()
        except RuntimeError:
            # Completer was already deleted; ignore.
            return

    line_edit.textChanged.connect(refresh)
    return completer


def setup_track_completer(
    line_edit: QLineEdit,
    completer_data: DatabaseCompleter,
    artist_filter: Optional[str] = None
) -> QCompleter:
    """
    Setup autocomplete for a track input field.

    Args:
        line_edit: The QLineEdit to attach the completer to
        completer_data: The DatabaseCompleter with loaded data
        artist_filter: Optional artist name to filter tracks

    Returns:
        The configured QCompleter
    """
    if artist_filter:
        tracks = completer_data.get_artist_tracks(artist_filter)
    else:
        tracks = completer_data.get_tracks()

    completer = CaseInsensitiveCompleter(tracks, line_edit)
    line_edit.setCompleter(completer)
    return completer


def update_track_completer(
    line_edit: QLineEdit,
    completer_data: DatabaseCompleter,
    artist: str
) -> None:
    """
    Update track completer to show only tracks by the specified artist.

    Args:
        line_edit: The QLineEdit with the completer
        completer_data: The DatabaseCompleter with loaded data
        artist: The artist to filter by
    """
    if artist:
        tracks = completer_data.get_artist_tracks(artist)
    else:
        tracks = completer_data.get_tracks()

    completer = line_edit.completer()
    if completer:
        model = QStringListModel(tracks)
        completer.setModel(model)
