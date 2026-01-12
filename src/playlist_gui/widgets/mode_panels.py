"""
Mode Panels - Mode-specific control panels for the Generate UI.

Each mode (Artist/History/Seeds) has its own panel with controls
specific to that generation mode.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..autocomplete import DatabaseCompleter, setup_artist_completer, setup_track_completer
from ..seed_resolver import resolve_track_from_display
from .seed_chips import SeedChip, SeedChipsList


# Type aliases
PresenceLevel = Literal["low", "medium", "high", "max"]
VarietyLevel = Literal["focused", "balanced", "sprawling"]


class ArtistModePanel(QWidget):
    """
    Artist(s) mode controls.

    Features:
    - Artist input with autocomplete
    - Multi-artist note (backend uses first artist only for now)
    - Presence control (how much of playlist from seed artist)
    - Variety control (stylistic spread)
    """

    artist_changed = Signal(str)  # Emits primary artist name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._db_completer: Optional[DatabaseCompleter] = None
        self._artists: List[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Artist input row
        artist_row = QHBoxLayout()
        artist_row.addWidget(QLabel("Artist:"))

        self._artist_edit = QLineEdit()
        self._artist_edit.setPlaceholderText("Start typing artist name...")
        self._artist_edit.setMinimumWidth(250)
        self._artist_edit.textChanged.connect(self._on_artist_text_changed)
        artist_row.addWidget(self._artist_edit, stretch=1)

        layout.addLayout(artist_row)

        # Multi-artist note (hidden by default)
        self._multi_note = QLabel(
            "<i>Multi-artist journeys coming soon — currently generating from the first artist only.</i>"
        )
        self._multi_note.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        self._multi_note.setWordWrap(True)
        self._multi_note.hide()
        layout.addWidget(self._multi_note)

        # Presence and variety row
        tuning_row = QHBoxLayout()
        tuning_row.setSpacing(20)

        # Presence dropdown
        presence_section = QHBoxLayout()
        presence_section.addWidget(QLabel("Presence:"))
        self._presence_combo = QComboBox()
        self._presence_combo.addItems(["Low (~10%)", "Medium (~25%)", "High (~40%)", "Max (~60%)"])
        self._presence_combo.setCurrentIndex(1)  # Default: Medium
        self._presence_combo.setToolTip(
            "How much of the playlist should feature the seed artist?\n"
            "Low = few tracks, Max = majority of tracks"
        )
        presence_section.addWidget(self._presence_combo)
        tuning_row.addLayout(presence_section)

        # Variety slider
        variety_section = QHBoxLayout()
        variety_section.addWidget(QLabel("Variety:"))

        self._variety_slider = QSlider(Qt.Horizontal)
        self._variety_slider.setMinimum(0)
        self._variety_slider.setMaximum(2)
        self._variety_slider.setValue(1)  # Default: balanced
        self._variety_slider.setTickPosition(QSlider.TicksBelow)
        self._variety_slider.setTickInterval(1)
        self._variety_slider.setFixedWidth(100)
        self._variety_slider.valueChanged.connect(self._on_variety_changed)
        variety_section.addWidget(self._variety_slider)

        self._variety_label = QLabel("Balanced")
        self._variety_label.setFixedWidth(70)
        variety_section.addWidget(self._variety_label)

        tuning_row.addLayout(variety_section)
        tuning_row.addStretch()

        layout.addLayout(tuning_row)

    def _on_artist_text_changed(self, text: str) -> None:
        """Handle artist text change."""
        text = text.strip()
        if text:
            self._artists = [text]
        else:
            self._artists = []

        # For now, multi-artist note is hidden since we only use single input
        # In future, this could be a chip-based multi-input
        self.artist_changed.emit(text)

    def _on_variety_changed(self, value: int) -> None:
        """Update variety label."""
        labels = ["Focused", "Balanced", "Sprawling"]
        self._variety_label.setText(labels[value])

    def set_completer_data(self, completer: DatabaseCompleter) -> None:
        """Set autocomplete data source."""
        self._db_completer = completer
        if completer:
            setup_artist_completer(self._artist_edit, completer)

    def get_artists(self) -> List[str]:
        """Get list of entered artists."""
        text = self._artist_edit.text().strip()
        return [text] if text else []

    def get_primary_artist(self) -> Optional[str]:
        """Get the first/primary artist."""
        artists = self.get_artists()
        return artists[0] if artists else None

    def get_presence(self) -> PresenceLevel:
        """Get selected presence level."""
        index = self._presence_combo.currentIndex()
        levels: List[PresenceLevel] = ["low", "medium", "high", "max"]
        return levels[index]

    def get_variety(self) -> VarietyLevel:
        """Get selected variety level."""
        value = self._variety_slider.value()
        levels: List[VarietyLevel] = ["focused", "balanced", "sprawling"]
        return levels[value]

    def set_artist(self, artist: str) -> None:
        """Set artist text programmatically."""
        self._artist_edit.setText(artist)

    def clear(self) -> None:
        """Clear all inputs."""
        self._artist_edit.clear()
        self._presence_combo.setCurrentIndex(1)
        self._variety_slider.setValue(1)


class HistoryModePanel(QWidget):
    """
    History mode controls.

    Features:
    - Time window selection (how far back to look for listening history)
    """

    window_changed = Signal(int)  # Emits days value

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Time window row
        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Time window:"))

        self._window_combo = QComboBox()
        self._window_combo.addItems(["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"])
        self._window_combo.setCurrentIndex(2)  # Default: 30 days
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        self._window_combo.setToolTip(
            "How far back to look at your listening history.\n"
            "Longer windows include more variety but may be less current."
        )
        window_row.addWidget(self._window_combo)

        window_row.addStretch()
        layout.addLayout(window_row)

        # Info label
        info = QLabel(
            "<i>Generate a playlist based on your recent listening patterns.</i>"
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(info)

        layout.addStretch()

    def _on_window_changed(self, index: int) -> None:
        """Handle window selection change."""
        self.window_changed.emit(self.get_window())

    def get_window(self) -> int:
        """Get selected time window in days."""
        index = self._window_combo.currentIndex()
        windows = [7, 14, 30, 90]
        return windows[index]

    def set_window(self, days: int) -> None:
        """Set time window programmatically."""
        windows = [7, 14, 30, 90]
        if days in windows:
            self._window_combo.setCurrentIndex(windows.index(days))


class SeedsModePanel(QWidget):
    """
    Seed(s) mode controls.

    Features:
    - Track search with autocomplete
    - Seed chips list with drag reordering
    - Auto-order toggle
    - DJ bridging hint (when conditions not met)
    """

    seeds_changed = Signal()  # Emits when seeds modified

    def __init__(self, db_path: str = "data/metadata.db", parent: QWidget | None = None):
        super().__init__(parent)
        self._db_path = db_path
        self._db_completer: Optional[DatabaseCompleter] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Track search row
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Add track:"))

        self._track_edit = QLineEdit()
        self._track_edit.setPlaceholderText("Search for a track to add as seed...")
        self._track_edit.setMinimumWidth(300)
        self._track_edit.returnPressed.connect(self._on_add_seed)
        search_row.addWidget(self._track_edit, stretch=1)

        self._add_btn = QPushButton("Add Seed")
        self._add_btn.setFixedWidth(80)
        self._add_btn.clicked.connect(self._on_add_seed)
        search_row.addWidget(self._add_btn)

        layout.addLayout(search_row)

        # Seed chips list
        self._chips_list = SeedChipsList()
        self._chips_list.seeds_changed.connect(self._on_seeds_changed)
        layout.addWidget(self._chips_list)

        # Auto-order toggle row
        order_row = QHBoxLayout()

        self._auto_order_check = QCheckBox("Auto-order seeds for optimal bridging")
        self._auto_order_check.setChecked(True)
        self._auto_order_check.setToolTip(
            "When enabled, seeds are automatically reordered to create\n"
            "smoother transitions between tracks. Disable to preserve\n"
            "your manual ordering."
        )
        self._auto_order_check.toggled.connect(self._on_auto_order_changed)
        order_row.addWidget(self._auto_order_check)

        order_row.addStretch()

        # Remove selected button
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setFixedWidth(110)
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._chips_list.remove_selected)
        order_row.addWidget(self._remove_btn)

        layout.addLayout(order_row)

        # DJ bridging hint (hidden by default)
        self._dj_hint = QLabel(
            "Genre enrichment requires 2+ seeds from different artists."
        )
        self._dj_hint.setStyleSheet(
            "color: #856404; background: #fff3cd; border: 1px solid #ffc107; "
            "border-radius: 4px; padding: 6px; font-size: 11px;"
        )
        self._dj_hint.setWordWrap(True)
        self._dj_hint.hide()
        layout.addWidget(self._dj_hint)

        # Connect list selection to remove button
        self._chips_list._list.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_add_seed(self) -> None:
        """Add current search text as a seed."""
        display = self._track_edit.text().strip()
        if not display:
            return

        # Resolve to SeedChip
        chip = resolve_track_from_display(display, self._db_path)
        if chip:
            if self._chips_list.add_seed(chip):
                self._track_edit.clear()
            else:
                # Duplicate - visual feedback could be added
                pass
        else:
            # Track not found - could show error feedback
            pass

    def _on_seeds_changed(self) -> None:
        """Handle seeds list change."""
        self._update_dj_hint()
        self.seeds_changed.emit()

    def _on_auto_order_changed(self, checked: bool) -> None:
        """Handle auto-order toggle."""
        self._chips_list.set_auto_order(checked)

    def _on_selection_changed(self) -> None:
        """Handle list selection change."""
        has_selection = self._chips_list._list.currentRow() >= 0
        self._remove_btn.setEnabled(has_selection)

    def _update_dj_hint(self) -> None:
        """Update DJ bridging hint visibility."""
        count = self._chips_list.seed_count()
        unique_artists = self._chips_list.unique_artist_count()

        # Show hint if DJ bridging conditions not met
        show_hint = count > 0 and (count < 2 or unique_artists < 2)
        self._dj_hint.setVisible(show_hint)

    def set_completer_data(self, completer: DatabaseCompleter) -> None:
        """Set autocomplete data source."""
        self._db_completer = completer
        if completer:
            setup_track_completer(self._track_edit, completer, artist_filter=None)

    def set_db_path(self, path: str) -> None:
        """Set database path for seed resolution."""
        self._db_path = path

    def get_seeds(self) -> List[SeedChip]:
        """Get list of seed chips."""
        return self._chips_list.get_seeds()

    def get_seed_track_ids(self) -> List[str]:
        """Get list of track IDs."""
        return self._chips_list.get_seed_track_ids()

    def get_seed_artist_keys(self) -> List[str]:
        """Get list of artist keys."""
        return self._chips_list.get_seed_artist_keys()

    def get_auto_order(self) -> bool:
        """Check if auto-ordering is enabled."""
        return self._auto_order_check.isChecked()

    def seed_count(self) -> int:
        """Get number of seeds."""
        return self._chips_list.seed_count()

    def unique_artist_count(self) -> int:
        """Get number of unique artists."""
        return self._chips_list.unique_artist_count()

    def clear(self) -> None:
        """Clear all seeds."""
        self._chips_list.clear()
        self._track_edit.clear()

    def set_seeds(self, chips: List[SeedChip]) -> None:
        """Set seeds programmatically."""
        self._chips_list.set_seeds(chips)
