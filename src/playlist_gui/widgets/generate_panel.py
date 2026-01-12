"""
Generate Panel - Main generation control panel for the GUI.

Composes mode selection, global controls (cohesion, length, recency, spacing),
mode-specific panels, and action buttons into a cohesive generation UI.

Integrates with Phase 1 PolicyLayer for runtime configuration derivation.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Literal, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..autocomplete import DatabaseCompleter
from ..ui_state import UIStateModel
from .cohesion_dial import CohesionDial
from .mode_panels import ArtistModePanel, HistoryModePanel, SeedsModePanel
from .seed_chips import SeedChip


ModeType = Literal["artist", "history", "seeds"]


class GeneratePanel(QWidget):
    """
    Main generation control panel with mode switching and PolicyLayer integration.

    Emits:
        generate_requested: When user clicks Generate, emits UIStateModel as dict
        regenerate_requested: When user clicks Regenerate
        new_seeds_requested: When user clicks New Seeds
        cancel_requested: When user clicks Cancel
    """

    generate_requested = Signal(dict)  # UIStateModel as dict
    regenerate_requested = Signal(dict)  # UIStateModel as dict
    new_seeds_requested = Signal(dict)  # UIStateModel as dict
    cancel_requested = Signal()

    def __init__(
        self,
        db_completer: Optional[DatabaseCompleter] = None,
        db_path: str = "data/metadata.db",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._db_completer = db_completer
        self._db_path = db_path
        self._is_generating = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ─────────────────────────────────────────────────────────────────────
        # Mode selector row
        # ─────────────────────────────────────────────────────────────────────
        mode_frame = QFrame()
        mode_frame.setStyleSheet("""
            QFrame {
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(8, 4, 8, 4)
        mode_layout.setSpacing(4)

        mode_layout.addWidget(QLabel("<b>Mode:</b>"))

        self._mode_group = QButtonGroup(self)
        modes = [("artist", "Artist(s)"), ("history", "History"), ("seeds", "Seed(s)")]

        for mode_id, mode_label in modes:
            radio = QRadioButton(mode_label)
            radio.setProperty("mode_id", mode_id)
            self._mode_group.addButton(radio)
            mode_layout.addWidget(radio)
            if mode_id == "artist":
                radio.setChecked(True)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)

        mode_layout.addStretch()
        layout.addWidget(mode_frame)

        # ─────────────────────────────────────────────────────────────────────
        # Global controls row
        # ─────────────────────────────────────────────────────────────────────
        global_row = QHBoxLayout()
        global_row.setSpacing(20)

        # Cohesion dial
        self._cohesion_dial = CohesionDial()
        global_row.addWidget(self._cohesion_dial)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #ddd;")
        global_row.addWidget(sep1)

        # Length dropdown
        length_section = QHBoxLayout()
        length_section.addWidget(QLabel("Length:"))
        self._length_combo = QComboBox()
        self._length_combo.addItems(["20", "30", "40", "50"])
        self._length_combo.setCurrentIndex(1)  # Default: 30
        self._length_combo.setFixedWidth(60)
        self._length_combo.setToolTip("Number of tracks in the generated playlist")
        length_section.addWidget(self._length_combo)
        global_row.addLayout(length_section)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #ddd;")
        global_row.addWidget(sep2)

        # Recency filter
        recency_section = QHBoxLayout()
        recency_section.setSpacing(6)

        self._recency_check = QCheckBox("Exclude recent:")
        self._recency_check.setChecked(True)
        self._recency_check.setToolTip(
            "Exclude tracks you've played recently to keep playlists fresh"
        )
        self._recency_check.toggled.connect(self._on_recency_toggled)
        recency_section.addWidget(self._recency_check)

        self._recency_days = QSpinBox()
        self._recency_days.setRange(1, 90)
        self._recency_days.setValue(14)
        self._recency_days.setSuffix(" days")
        self._recency_days.setFixedWidth(80)
        self._recency_days.setToolTip("How far back to check for recent plays")
        recency_section.addWidget(self._recency_days)

        recency_section.addWidget(QLabel("if played"))

        self._recency_plays = QSpinBox()
        self._recency_plays.setRange(1, 10)
        self._recency_plays.setValue(1)
        self._recency_plays.setSuffix("+ times")
        self._recency_plays.setFixedWidth(80)
        self._recency_plays.setToolTip("Minimum play count to be excluded")
        recency_section.addWidget(self._recency_plays)

        global_row.addLayout(recency_section)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #ddd;")
        global_row.addWidget(sep3)

        # Artist spacing
        spacing_section = QHBoxLayout()
        spacing_section.addWidget(QLabel("Artist spacing:"))
        self._spacing_combo = QComboBox()
        self._spacing_combo.addItems(["Normal", "Strong"])
        self._spacing_combo.setCurrentIndex(0)
        self._spacing_combo.setToolTip(
            "How many tracks between repeated artists.\n"
            "Normal = 6 tracks, Strong = 9 tracks"
        )
        spacing_section.addWidget(self._spacing_combo)
        global_row.addLayout(spacing_section)

        global_row.addStretch()
        layout.addLayout(global_row)

        # ─────────────────────────────────────────────────────────────────────
        # Mode-specific panel stack
        # ─────────────────────────────────────────────────────────────────────
        self._mode_stack = QStackedWidget()

        # Artist mode panel
        self._artist_panel = ArtistModePanel()
        if self._db_completer:
            self._artist_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._artist_panel)

        # History mode panel
        self._history_panel = HistoryModePanel()
        self._mode_stack.addWidget(self._history_panel)

        # Seeds mode panel
        self._seeds_panel = SeedsModePanel(db_path=self._db_path)
        if self._db_completer:
            self._seeds_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._seeds_panel)

        layout.addWidget(self._mode_stack)

        # ─────────────────────────────────────────────────────────────────────
        # Action buttons row
        # ─────────────────────────────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        # Generate button
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedWidth(100)
        self._generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86c7;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #5a96d7;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        action_row.addWidget(self._generate_btn)

        # Regenerate button
        self._regenerate_btn = QPushButton("Regenerate")
        self._regenerate_btn.setFixedWidth(100)
        self._regenerate_btn.setToolTip(
            "Re-run generation with the same settings and seeds"
        )
        self._regenerate_btn.clicked.connect(self._on_regenerate)
        action_row.addWidget(self._regenerate_btn)

        # New Seeds button
        self._new_seeds_btn = QPushButton("New Seeds")
        self._new_seeds_btn.setFixedWidth(100)
        self._new_seeds_btn.setToolTip(
            "Artist/History: Re-pick internal seeds and regenerate.\n"
            "Seeds: Re-run auto-ordering if enabled."
        )
        self._new_seeds_btn.clicked.connect(self._on_new_seeds)
        action_row.addWidget(self._new_seeds_btn)

        action_row.addStretch()

        # Cancel button
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #c74a4a;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #d75a5a;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self._cancel_btn.clicked.connect(self._on_cancel)
        action_row.addWidget(self._cancel_btn)

        layout.addLayout(action_row)

        # ─────────────────────────────────────────────────────────────────────
        # Progress row
        # ─────────────────────────────────────────────────────────────────────
        progress_row = QHBoxLayout()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        progress_row.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setFixedWidth(200)
        self._stage_label.setStyleSheet("color: #666;")
        progress_row.addWidget(self._stage_label)

        layout.addLayout(progress_row)

    def _on_mode_changed(self) -> None:
        """Handle mode radio button change."""
        mode = self._get_current_mode()
        index = {"artist": 0, "history": 1, "seeds": 2}.get(mode, 0)
        self._mode_stack.setCurrentIndex(index)

    def _on_recency_toggled(self, checked: bool) -> None:
        """Handle recency checkbox toggle."""
        self._recency_days.setEnabled(checked)
        self._recency_plays.setEnabled(checked)

    def _get_current_mode(self) -> ModeType:
        """Get currently selected mode."""
        checked = self._mode_group.checkedButton()
        if checked:
            mode_id = checked.property("mode_id")
            if mode_id in ("artist", "history", "seeds"):
                return mode_id
        return "artist"

    def build_ui_state(self) -> UIStateModel:
        """Construct UIStateModel from current UI state."""
        mode = self._get_current_mode()

        return UIStateModel(
            mode=mode,
            cohesion=self._cohesion_dial.value(),
            track_count=int(self._length_combo.currentText()),
            recency_enabled=self._recency_check.isChecked(),
            recency_days=self._recency_days.value(),
            recency_plays_threshold=self._recency_plays.value(),
            artist_spacing=self._spacing_combo.currentText().lower(),
            artist_queries=self._artist_panel.get_artists() if mode == "artist" else [],
            artist_presence=self._artist_panel.get_presence() if mode == "artist" else "medium",
            artist_variety=self._artist_panel.get_variety() if mode == "artist" else "balanced",
            history_window_days=self._history_panel.get_window() if mode == "history" else 30,
            seed_track_ids=self._seeds_panel.get_seed_track_ids() if mode == "seeds" else [],
            seed_auto_order=self._seeds_panel.get_auto_order() if mode == "seeds" else True,
        )

    def get_seed_artist_keys(self) -> List[str]:
        """Get artist keys for seeds (for policy evaluation)."""
        if self._get_current_mode() != "seeds":
            return []
        return self._seeds_panel.get_seed_artist_keys()

    @Slot()
    def _on_generate(self) -> None:
        """Handle Generate button click."""
        if self._is_generating:
            return

        ui_state = self.build_ui_state()
        self.generate_requested.emit(asdict(ui_state))

    @Slot()
    def _on_regenerate(self) -> None:
        """Handle Regenerate button click."""
        if self._is_generating:
            return

        ui_state = self.build_ui_state()
        self.regenerate_requested.emit(asdict(ui_state))

    @Slot()
    def _on_new_seeds(self) -> None:
        """Handle New Seeds button click."""
        if self._is_generating:
            return

        ui_state = self.build_ui_state()
        self.new_seeds_requested.emit(asdict(ui_state))

    @Slot()
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.cancel_requested.emit()

    def set_generating(self, is_generating: bool) -> None:
        """Update UI state for generation in progress."""
        self._is_generating = is_generating
        self._generate_btn.setEnabled(not is_generating)
        self._regenerate_btn.setEnabled(not is_generating)
        self._new_seeds_btn.setEnabled(not is_generating)
        self._cancel_btn.setEnabled(is_generating)

        # Disable mode switching during generation
        for btn in self._mode_group.buttons():
            btn.setEnabled(not is_generating)

    def set_progress(self, value: int, stage: str = "") -> None:
        """Update progress bar and stage label."""
        self._progress_bar.setValue(value)
        self._stage_label.setText(stage)

    def reset_progress(self) -> None:
        """Reset progress bar to initial state."""
        self._progress_bar.setValue(0)
        self._stage_label.setText("")

    def set_completer_data(self, completer: DatabaseCompleter) -> None:
        """Set autocomplete data source for all panels."""
        self._db_completer = completer
        self._artist_panel.set_completer_data(completer)
        self._seeds_panel.set_completer_data(completer)

    def set_db_path(self, path: str) -> None:
        """Set database path for seed resolution."""
        self._db_path = path
        self._seeds_panel.set_db_path(path)

    def get_primary_artist(self) -> Optional[str]:
        """Get primary artist for artist mode."""
        return self._artist_panel.get_primary_artist()

    def get_seeds(self) -> List[SeedChip]:
        """Get seed chips for seeds mode."""
        return self._seeds_panel.get_seeds()
