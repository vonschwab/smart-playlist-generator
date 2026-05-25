"""
Mode Panels - Mode-specific control panels for the Generate UI.

Each mode (Artist/Seeds) has its own panel with controls
specific to that generation mode.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..autocomplete import DatabaseCompleter, setup_artist_completer, setup_genre_completer, setup_track_completer
from ..seed_resolver import resolve_track_from_display
from .seed_chips import SeedChip, SeedChipsList

logger = logging.getLogger(__name__)


# Type aliases
PresenceLevel = Literal["very_low", "low", "medium", "high", "very_high"]
VarietyLevel = Literal["focused", "balanced", "sprawling"]
MODE_CONTROL_GROUP_HEIGHT = 48


def _create_mode_control_group(
    owner: QWidget,
    key: str,
    title: str,
    content: QWidget,
    *,
    stretch: bool = False,
) -> QFrame:
    """Create a compact labelled card for mode-specific controls."""
    group = QFrame()
    group.setObjectName("modeControlGroup")
    group.setMinimumHeight(MODE_CONTROL_GROUP_HEIGHT)
    group.setMaximumHeight(MODE_CONTROL_GROUP_HEIGHT)
    group.setSizePolicy(
        QSizePolicy.Expanding if stretch else QSizePolicy.Maximum,
        QSizePolicy.Fixed,
    )

    layout = QHBoxLayout(group)
    layout.setContentsMargins(8, 5, 8, 5)
    layout.setSpacing(7)

    title_label = QLabel(title)
    title_label.setObjectName("modeControlGroupTitle")
    layout.addWidget(title_label)
    layout.addWidget(content, stretch=1 if stretch else 0)

    owner._control_groups[key] = group  # type: ignore[attr-defined]
    setattr(owner, f"_{key}_group_title", title_label)
    return group


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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._db_completer: Optional[DatabaseCompleter] = None
        self._artists: List[str] = []
        self._control_groups: dict[str, QFrame] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Artist input row
        self._artist_edit = QLineEdit()
        self._artist_edit.setPlaceholderText("Start typing artist name...")
        self._artist_edit.setMinimumWidth(250)
        self._artist_edit.textChanged.connect(self._on_artist_text_changed)
        self._artist_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(
            _create_mode_control_group(
                self,
                "artist",
                "Artist",
                self._artist_edit,
                stretch=True,
            )
        )

        # Multi-artist note (hidden by default)
        self._multi_note = QLabel(
            "<i>Multi-artist journeys coming soon — currently generating from the first artist only.</i>"
        )
        self._multi_note.setObjectName("modeInlineNote")
        self._multi_note.setWordWrap(True)
        self._multi_note.hide()
        layout.addWidget(self._multi_note)

        # Presence and variety row
        tuning_row = QHBoxLayout()
        tuning_row.setContentsMargins(0, 0, 0, 0)
        tuning_row.setSpacing(8)

        # Presence dropdown
        self._presence_combo = QComboBox()
        self._presence_combo.addItems(
            [
                "Very Low (~5%)",
                "Low (~10%)",
                "Medium (~12.5%)",
                "High (~20%)",
                "Very High (~33%)",
            ]
        )
        self._presence_combo.setCurrentIndex(2)  # Default: Medium
        self._presence_combo.setToolTip(
            "How much of the playlist should feature the seed artist?\n"
            "Very Low = few tracks, Very High = strong seed presence"
        )
        self._presence_combo.setMinimumWidth(180)
        self._presence_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        tuning_row.addWidget(
            _create_mode_control_group(
                self,
                "presence",
                "Presence",
                self._presence_combo,
            )
        )

        # Variety slider
        variety_container = QWidget()
        variety_container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        variety_section = QHBoxLayout(variety_container)
        variety_section.setContentsMargins(0, 0, 0, 0)
        variety_section.setSpacing(4)
        variety_help = (
            "Controls stylistic spread within the seed artist's neighborhood.\n"
            "Focused stays close to the core sound, Balanced mixes nearby styles,\n"
            "Sprawling reaches farther while keeping artist influence.\n"
            "Genre/Sonic modes set strictness; Variety sets spread."
        )

        self._variety_slider = QSlider(Qt.Horizontal)
        self._variety_slider.setMinimum(0)
        self._variety_slider.setMaximum(2)
        self._variety_slider.setValue(1)  # Default: balanced
        self._variety_slider.setTickPosition(QSlider.TicksBelow)
        self._variety_slider.setTickInterval(1)
        self._variety_slider.setMinimumWidth(140)
        self._variety_slider.setMaximumWidth(220)
        self._variety_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._variety_slider.setToolTip(variety_help)
        self._variety_slider.valueChanged.connect(self._on_variety_changed)
        variety_section.addWidget(self._variety_slider)

        self._variety_label = QLabel("Balanced")
        self._variety_label.setMinimumWidth(80)
        self._variety_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        variety_section.addWidget(self._variety_label)

        variety_group = _create_mode_control_group(
            self,
            "variety",
            "Variety",
            variety_container,
        )
        self._variety_group_title.setToolTip(variety_help)
        tuning_row.addWidget(variety_group)

        # Include collaborations checkbox
        self._collabs_check = QCheckBox("Include")
        self._collabs_check.setToolTip(
            "Mix collaboration tracks into the seed pool.\n"
            "Examples: \"Miles Davis Quintet\", \"Greg Foat & Art Themen\",\n"
            "\"Miles Davis & John Coltrane\". Recognises &, and, feat./ft.,\n"
            "with, +, /, comma, and ensemble suffixes (trio, quartet,\n"
            "quintet, sextet, group, band, ensemble, orchestra)."
        )
        tuning_row.addWidget(
            _create_mode_control_group(
                self,
                "collaborations",
                "Collaborations",
                self._collabs_check,
            )
        )

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
        levels: List[PresenceLevel] = [
            "very_low",
            "low",
            "medium",
            "high",
            "very_high",
        ]
        return levels[index]

    def get_variety(self) -> VarietyLevel:
        """Get selected variety level."""
        value = self._variety_slider.value()
        levels: List[VarietyLevel] = ["focused", "balanced", "sprawling"]
        return levels[value]

    def get_include_collaborations(self) -> bool:
        """Whether collaboration tracks should join the seed pool."""
        return self._collabs_check.isChecked()

    def set_include_collaborations(self, checked: bool) -> None:
        """Set the collaboration toggle programmatically."""
        self._collabs_check.setChecked(bool(checked))

    def set_artist(self, artist: str) -> None:
        """Set artist text programmatically."""
        self._artist_edit.setText(artist)

    def set_primary_artist(self, artist: str) -> None:
        """Set primary artist text programmatically."""
        self.set_artist(artist)

    def set_presence(self, level: str) -> None:
        """Set presence level programmatically."""
        levels = ["very_low", "low", "medium", "high", "very_high"]
        if level in levels:
            self._presence_combo.setCurrentIndex(levels.index(level))

    def set_variety(self, level: str) -> None:
        """Set variety level programmatically."""
        levels = ["focused", "balanced", "sprawling"]
        if level in levels:
            self._variety_slider.setValue(levels.index(level))

    def clear(self) -> None:
        """Clear all inputs."""
        self._artist_edit.clear()
        self._presence_combo.setCurrentIndex(2)
        self._variety_slider.setValue(1)
        self._collabs_check.setChecked(False)


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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._db_path = db_path
        self._db_completer: Optional[DatabaseCompleter] = None
        self._control_groups: dict[str, QFrame] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Track search row
        search_content = QWidget()
        search_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_row = QHBoxLayout(search_content)
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(6)

        self._track_edit = QLineEdit()
        self._track_edit.setPlaceholderText("Search for a track to add as seed...")
        self._track_edit.setMinimumWidth(300)
        self._track_edit.returnPressed.connect(self._on_add_seed)
        search_row.addWidget(self._track_edit, stretch=1)

        self._add_btn = QPushButton("Add Seed")
        self._add_btn.setMinimumWidth(100)
        self._add_btn.clicked.connect(self._on_add_seed)
        search_row.addWidget(self._add_btn)

        layout.addWidget(
            _create_mode_control_group(
                self,
                "track",
                "Track",
                search_content,
                stretch=True,
            )
        )

        # Seed chips list
        self._chips_list = SeedChipsList()
        self._chips_list.seeds_changed.connect(self._on_seeds_changed)
        layout.addWidget(self._chips_list)

        # Auto-order toggle row
        order_content = QWidget()
        order_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        order_row = QHBoxLayout(order_content)
        order_row.setContentsMargins(0, 0, 0, 0)
        order_row.setSpacing(8)

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
        self._remove_btn.setMinimumWidth(130)
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._chips_list.remove_selected)
        order_row.addWidget(self._remove_btn)

        layout.addWidget(
            _create_mode_control_group(
                self,
                "seed_order",
                "Seed Order",
                order_content,
                stretch=True,
            )
        )

        # DJ bridging hint (hidden by default)
        self._dj_hint = QLabel(
            "Genre enrichment requires 2+ seeds from different artists."
        )
        self._dj_hint.setObjectName("warningHint")
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

        chip: Optional[SeedChip] = None

        # Preferred: use completer's cached data (no parsing required)
        if self._db_completer and self._db_completer.is_loaded():
            track_data = self._db_completer.get_track_data_by_display(display)
            if track_data:
                track_id, title, artist, album, artist_key = track_data
                # Reconstruct display for consistency
                if artist:
                    final_display = f"{title} - {artist}"
                    if album:
                        final_display += f" ({album})"
                else:
                    final_display = title
                chip = SeedChip(
                    track_id=track_id,
                    display=final_display,
                    artist_key=artist_key,
                    title=title,
                    artist=artist,
                )

        # Fallback: use database query (string parsing required).
        # Reaches here only when the completer is unloaded or when the typed
        # text is ambiguous (multiple tracks share the same title).
        if chip is None:
            logger.debug(
                "Seed resolution using DB fallback for '%s' "
                "(completer unloaded or ambiguous title match)",
                display,
            )
            chip = resolve_track_from_display(display, self._db_path)

        if chip:
            if self._chips_list.add_seed(chip):
                self._track_edit.clear()
            # else: duplicate - visual feedback could be added
        # else: track not found - could show error feedback

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

    def get_seed_display_strings(self) -> List[str]:
        """
        Get list of seed display strings for backend communication.

        Returns "Title - Artist (Album)" format strings that the backend
        can parse and use for track lookup.
        """
        return self._chips_list.get_seed_display_strings()

    def get_seed_artist_keys(self) -> List[str]:
        """Get list of artist keys."""
        return self._chips_list.get_seed_artist_keys()

    def get_auto_order(self) -> bool:
        """Check if auto-ordering is enabled."""
        return self._auto_order_check.isChecked()

    def set_auto_order(self, enabled: bool) -> None:
        """Set auto-order checkbox programmatically."""
        self._auto_order_check.setChecked(enabled)

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


class GenreModePanel(QWidget):
    """Genre mode controls."""

    genre_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._db_completer: Optional[DatabaseCompleter] = None
        self._control_groups: dict[str, QFrame] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._genre_edit = QLineEdit()
        self._genre_edit.setPlaceholderText("Start typing genre name...")
        self._genre_edit.setMinimumWidth(250)
        self._genre_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._genre_edit.textChanged.connect(lambda text: self.genre_changed.emit(text.strip()))

        layout.addWidget(
            _create_mode_control_group(
                self,
                "genre",
                "Genre",
                self._genre_edit,
                stretch=True,
            )
        )

    def set_completer_data(self, completer: DatabaseCompleter) -> None:
        """Set autocomplete data source."""
        self._db_completer = completer
        if completer:
            setup_genre_completer(self._genre_edit, completer)

    def get_genre(self) -> str:
        """Get the entered genre."""
        return self._genre_edit.text().strip()

    def set_genre(self, genre: str) -> None:
        """Set genre text programmatically."""
        self._genre_edit.setText(genre)

    def clear(self) -> None:
        """Clear genre input."""
        self._genre_edit.clear()


class HistoryModePanel(QWidget):
    """History mode controls."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._control_groups: dict[str, QFrame] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel("Uses your configured listening history source.")
        label.setObjectName("historyHint")
        layout.addWidget(
            _create_mode_control_group(
                self,
                "history",
                "History",
                label,
                stretch=True,
            )
        )
