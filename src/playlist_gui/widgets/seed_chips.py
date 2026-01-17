"""
Seed Chips Widget - Draggable list of seed track chips.

Manages a list of seed tracks with support for:
- Adding/removing seeds
- Drag-and-drop reordering (when auto-order is off)
- Auto-ordering for optimal DJ bridging
- Storing stable track IDs and artist keys for policy evaluation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def compute_seed_order(chips: List["SeedChip"]) -> List[int]:
    """
    Compute an ordering of seeds to maximize bridging potential.

    This is a GUI-side heuristic that works without feature vectors.
    The backend may further refine the order if needed.

    Strategy:
    1. Group seeds by artist_key
    2. Interleave groups round-robin style to maximize variety
    3. This creates a sequence like: A1, B1, C1, A2, B2, C2, ...
       which helps bridging find transitions between different artists.

    Args:
        chips: List of SeedChip objects to order

    Returns:
        List of indices representing the new order.
        e.g., [2, 0, 1] means old[2] becomes new[0], etc.
    """
    if len(chips) <= 1:
        return list(range(len(chips)))

    # Group indices by artist_key, preserving order within each group
    artist_groups: Dict[str, List[int]] = {}
    group_order: List[str] = []  # Track order of first appearance

    for i, chip in enumerate(chips):
        key = chip.artist_key or f"__unknown_{i}__"
        if key not in artist_groups:
            artist_groups[key] = []
            group_order.append(key)
        artist_groups[key].append(i)

    # If all same artist, just return original order
    if len(artist_groups) == 1:
        return list(range(len(chips)))

    # Round-robin interleave: pick one from each artist group in rotation
    # This maximizes variety and creates natural bridge points
    ordered_indices: List[int] = []
    group_iterators = [iter(artist_groups[key]) for key in group_order]

    while group_iterators:
        remaining = []
        for it in group_iterators:
            try:
                ordered_indices.append(next(it))
            except StopIteration:
                pass
            else:
                remaining.append(it)
        group_iterators = remaining

    return ordered_indices


@dataclass
class SeedChip:
    """
    Data for a single seed track chip.

    Attributes:
        track_id: Stable database ID (rating_key or equivalent)
        display: Human-readable display string "Title - Artist"
        artist_key: Normalized artist key for policy evaluation
        title: Track title
        artist: Artist name
    """
    track_id: str
    display: str
    artist_key: str
    title: str = ""
    artist: str = ""


class SeedChipsList(QWidget):
    """
    Draggable list of seed track chips with auto-ordering support.

    Emits seeds_changed when seeds are added, removed, or reordered.
    Supports both manual drag reordering and automatic bridging optimization.
    """

    seeds_changed = Signal()  # Emits when seeds modified
    seed_count_changed = Signal(int)  # Emits new count

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._chips: List[SeedChip] = []
        self._auto_order: bool = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header with count
        header_row = QHBoxLayout()
        self._header_label = QLabel("<b>Seeds:</b>")
        header_row.addWidget(self._header_label)

        self._count_label = QLabel("0 tracks")
        self._count_label.setStyleSheet("color: #666;")
        header_row.addWidget(self._count_label)

        header_row.addStretch()

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setMinimumWidth(90)
        self._clear_btn.setStyleSheet("font-size: 11px;")
        self._clear_btn.clicked.connect(self.clear)
        self._clear_btn.setEnabled(False)
        header_row.addWidget(self._clear_btn)

        layout.addLayout(header_row)

        # List widget for chips
        self._list = QListWidget()
        self._list.setMinimumHeight(80)
        self._list.setMaximumHeight(150)
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Use global theme for list styling to match the dark UI.

        # Enable drag-drop reordering
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.model().rowsMoved.connect(self._on_rows_moved)

        layout.addWidget(self._list)

        # Info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        self._info_label.hide()
        layout.addWidget(self._info_label)

    def _on_rows_moved(self) -> None:
        """Handle internal drag-drop reordering."""
        if self._auto_order:
            # When auto-order is on, don't allow manual reordering
            # Rebuild from our internal list
            self._rebuild_list_display()
            return

        # Update internal list to match visual order
        new_chips = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            track_id = item.data(Qt.UserRole)
            for chip in self._chips:
                if chip.track_id == track_id:
                    new_chips.append(chip)
                    break

        self._chips = new_chips
        self.seeds_changed.emit()

    def _rebuild_list_display(self) -> None:
        """Rebuild the list widget from internal chip list."""
        self._list.clear()
        for chip in self._chips:
            item = QListWidgetItem(chip.display)
            item.setData(Qt.UserRole, chip.track_id)
            item.setToolTip(f"Track ID: {chip.track_id}\nArtist: {chip.artist}")
            self._list.addItem(item)

    def _update_ui(self) -> None:
        """Update UI elements after chip changes."""
        count = len(self._chips)
        self._count_label.setText(f"{count} track{'s' if count != 1 else ''}")
        self._clear_btn.setEnabled(count > 0)

        # Update drag-drop based on auto-order setting
        if self._auto_order:
            self._list.setDragDropMode(QAbstractItemView.NoDragDrop)
            self._info_label.setText("Auto-ordering enabled - order optimized for bridging")
            self._info_label.show()
        else:
            self._list.setDragDropMode(QAbstractItemView.InternalMove)
            self._info_label.setText("Drag to reorder seeds")
            self._info_label.show() if count > 1 else self._info_label.hide()

        self.seed_count_changed.emit(count)

    def add_seed(self, chip: SeedChip) -> bool:
        """
        Add a seed chip to the list.

        Returns False if track_id already exists.
        When auto-order is enabled, the list is re-sorted after adding.
        """
        # Check for duplicates
        if any(c.track_id == chip.track_id for c in self._chips):
            return False

        self._chips.append(chip)

        # Apply auto-ordering if enabled (will also rebuild display)
        if self._auto_order and len(self._chips) > 1:
            self._apply_auto_order()
        else:
            # Add to list widget manually
            item = QListWidgetItem(chip.display)
            item.setData(Qt.UserRole, chip.track_id)
            item.setToolTip(f"Track ID: {chip.track_id}\nArtist: {chip.artist}")
            self._list.addItem(item)

        self._update_ui()
        self.seeds_changed.emit()
        return True

    def remove_seed(self, index: int) -> None:
        """Remove seed at given index."""
        if 0 <= index < len(self._chips):
            self._chips.pop(index)
            self._list.takeItem(index)
            self._update_ui()
            self.seeds_changed.emit()

    def remove_selected(self) -> None:
        """Remove currently selected seed."""
        row = self._list.currentRow()
        if row >= 0:
            self.remove_seed(row)

    def clear(self) -> None:
        """Remove all seeds."""
        self._chips.clear()
        self._list.clear()
        self._update_ui()
        self.seeds_changed.emit()

    def get_seeds(self) -> List[SeedChip]:
        """Get list of all seed chips in current order."""
        return list(self._chips)

    def get_seed_track_ids(self) -> List[str]:
        """Get list of track IDs in current order."""
        return [chip.track_id for chip in self._chips]

    def get_seed_display_strings(self) -> List[str]:
        """
        Get list of seed display strings for backend communication.

        The backend expects "Title - Artist" format, not raw track IDs.
        This method provides the format needed for create_playlist_from_seed_tracks().
        """
        return [chip.display for chip in self._chips]

    def get_seed_artist_keys(self) -> List[str]:
        """Get list of artist keys in current order."""
        return [chip.artist_key for chip in self._chips]

    def seed_count(self) -> int:
        """Get number of seeds."""
        return len(self._chips)

    def unique_artist_count(self) -> int:
        """Get number of unique artists."""
        return len(set(chip.artist_key for chip in self._chips))

    def set_auto_order(self, enabled: bool) -> None:
        """
        Enable or disable auto-ordering.

        When enabled, seeds are ordered for optimal DJ bridging.
        When disabled, user can manually drag-reorder seeds.
        """
        was_enabled = self._auto_order
        self._auto_order = enabled
        self._update_ui()

        # Apply auto-ordering when toggled ON
        if enabled and not was_enabled:
            self._apply_auto_order()

    def get_auto_order(self) -> bool:
        """Check if auto-ordering is enabled."""
        return self._auto_order

    def _apply_auto_order(self) -> None:
        """
        Apply automatic ordering to maximize bridging potential.

        Called when auto-order is enabled and seeds change.
        Uses a simple heuristic that interleaves artists for variety.
        """
        if len(self._chips) <= 1:
            return

        try:
            new_order = compute_seed_order(self._chips)

            # Apply the new order when it changes, then refresh the list
            if new_order != list(range(len(self._chips))):
                self._chips = [self._chips[i] for i in new_order]
            self._rebuild_list_display()

            logger.debug(
                "Auto-ordered %d seeds: %s",
                len(self._chips),
                [c.display[:30] for c in self._chips],
            )
        except Exception as e:
            # If ordering fails, log warning and continue with current order
            logger.warning("Auto-ordering failed: %s", e)

    def reorder_for_bridging(self, new_order: List[int]) -> None:
        """
        Reorder seeds according to bridging optimization.

        Args:
            new_order: List of indices representing new order.
                       e.g., [2, 0, 1] means: old[2] becomes new[0], etc.
        """
        if len(new_order) != len(self._chips):
            return

        # Reorder internal list
        new_chips = [self._chips[i] for i in new_order]
        self._chips = new_chips

        # Rebuild display
        self._rebuild_list_display()
        self.seeds_changed.emit()

    def set_seeds(self, chips: List[SeedChip]) -> None:
        """Replace all seeds with a new list."""
        self._chips = list(chips)

        # Apply auto-ordering if enabled
        if self._auto_order and len(self._chips) > 1:
            self._apply_auto_order()
        else:
            self._rebuild_list_display()

        self._update_ui()
        self.seeds_changed.emit()
