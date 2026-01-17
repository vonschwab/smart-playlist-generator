"""
Generate Panel - Main generation control panel for the GUI.

Composes mode selection, global controls (genre/sonic, length, recency, spacing),
mode-specific panels, and action buttons into a cohesive generation UI.

Integrates with Phase 1 PolicyLayer for runtime configuration derivation.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Literal, Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..autocomplete import DatabaseCompleter
from ..ui_state import UIStateModel
from .mode_sliders import ModeSliders
from .mode_panels import ArtistModePanel, SeedsModePanel
from .seed_chips import SeedChip


ModeType = Literal["artist", "seeds"]


class GeneratePanel(QWidget):
    """
    Main generation control panel with mode switching and PolicyLayer integration.

    Emits:
        generate_requested: When user clicks Generate, emits UIStateModel as dict
        regenerate_requested: When user clicks Regenerate
        new_seeds_requested: When user clicks New Seeds
    """

    generate_requested = Signal(dict)  # UIStateModel as dict
    regenerate_requested = Signal(dict)  # UIStateModel as dict
    new_seeds_requested = Signal(dict)  # UIStateModel as dict
    mode_changed = Signal(str)

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
        self._has_run = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._setup_ui()
        self._update_run_controls()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ─────────────────────────────────────────────────────────────────────
        # Header toolbar - compact single row with all global controls + actions
        # ─────────────────────────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFixedHeight(72)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(10)

        # Mode selector (dropdown)
        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)

        mode_label = QLabel("Mode:")
        mode_label.setObjectName("controlLabel")
        mode_layout.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Artist", "artist")
        self._mode_combo.addItem("Seeds", "seeds")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._mode_combo.setFixedWidth(110)
        mode_layout.addWidget(self._mode_combo)

        header_layout.addWidget(mode_container)

        # Vertical separator
        header_layout.addWidget(self._create_vsep())

        # Genre/Sonic mode sliders (stacked)
        self._mode_sliders = ModeSliders()
        header_layout.addWidget(self._mode_sliders)

        # Vertical separator
        header_layout.addWidget(self._create_vsep())

        # Length dropdown
        length_container = QWidget()
        length_layout = QHBoxLayout(length_container)
        length_layout.setContentsMargins(0, 0, 0, 0)
        length_layout.setSpacing(4)

        length_label = QLabel("Length:")
        length_label.setObjectName("controlLabel")
        length_layout.addWidget(length_label)

        self._length_combo = self._create_menu_button(
            options=[("20", 20), ("30", 30), ("40", 40), ("50", 50)],
            default_value=30,
            width=72,
            tooltip="Number of tracks in the generated playlist",
        )
        length_layout.addWidget(self._length_combo)
        header_layout.addWidget(length_container)

        # Vertical separator
        header_layout.addWidget(self._create_vsep())

        # Recency filter (compact)
        recency_container = QWidget()
        recency_layout = QHBoxLayout(recency_container)
        recency_layout.setContentsMargins(0, 0, 0, 0)
        recency_layout.setSpacing(4)

        self._recency_check = QCheckBox("Freshness:")
        self._recency_check.setChecked(True)
        self._recency_check.setToolTip("Exclude recently played tracks")
        self._recency_check.toggled.connect(self._on_recency_toggled)
        recency_layout.addWidget(self._recency_check)

        self._recency_days = self._create_menu_button(
            options=[("7d", 7), ("14d", 14), ("30d", 30), ("60d", 60), ("90d", 90)],
            default_value=14,
            width=80,
            tooltip="Lookback days",
        )
        recency_layout.addWidget(self._recency_days)

        self._recency_plays = self._create_menu_button(
            options=[("1+", 1), ("2+", 2), ("3+", 3), ("5+", 5), ("10+", 10)],
            default_value=1,
            width=70,
            tooltip="Min plays to exclude",
        )
        recency_layout.addWidget(self._recency_plays)

        header_layout.addWidget(recency_container)

        # Vertical separator
        header_layout.addWidget(self._create_vsep())

        # Artist spacing
        spacing_container = QWidget()
        spacing_layout = QHBoxLayout(spacing_container)
        spacing_layout.setContentsMargins(0, 0, 0, 0)
        spacing_layout.setSpacing(4)

        spacing_label = QLabel("Gap:")
        spacing_label.setObjectName("controlLabel")
        spacing_label.setToolTip("Artist spacing")
        spacing_layout.addWidget(spacing_label)

        self._spacing_combo = self._create_menu_button(
            options=[("Normal", "normal"), ("Strong", "strong")],
            default_value="normal",
            width=96,
            tooltip="Tracks between repeated artists\nNormal=6, Strong=9",
        )
        spacing_layout.addWidget(self._spacing_combo)
        header_layout.addWidget(spacing_container)

        # Diversity bonus
        diversity_container = QWidget()
        diversity_layout = QHBoxLayout(diversity_container)
        diversity_layout.setContentsMargins(0, 0, 0, 0)
        diversity_layout.setSpacing(4)

        diversity_label = QLabel("Diversity:")
        diversity_label.setObjectName("controlLabel")
        diversity_layout.addWidget(diversity_label)

        self._diversity_levels = ["Very Low", "Low", "Normal", "High", "Very High"]
        self._diversity_values = [0.00, 0.02, 0.04, 0.06, 0.08]

        self._diversity_slider = QSlider(Qt.Horizontal)
        self._diversity_slider.setMinimum(0)
        self._diversity_slider.setMaximum(len(self._diversity_levels) - 1)
        self._diversity_slider.setValue(2)
        self._diversity_slider.setTickPosition(QSlider.NoTicks)
        self._diversity_slider.setFixedWidth(90)
        self._diversity_slider.setToolTip(
            "Soft bonus for selecting new artists\n"
            "Higher values encourage more variety"
        )
        self._diversity_slider.valueChanged.connect(self._on_diversity_changed)
        diversity_layout.addWidget(self._diversity_slider)

        self._diversity_value = QLabel(self._diversity_levels[2])
        self._diversity_value.setObjectName("diversityValue")
        self._diversity_value.setFixedWidth(72)
        diversity_layout.addWidget(self._diversity_value)

        header_layout.addWidget(diversity_container)

        # Push action buttons to the right edge to avoid trailing empty space
        header_layout.addStretch()

        # Action buttons in header
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setObjectName("primaryButton")
        self._generate_btn.setMinimumWidth(110)
        self._generate_btn.clicked.connect(self._on_generate)
        header_layout.addWidget(self._generate_btn)

        self._new_seeds_btn = QPushButton("New Seeds")
        self._new_seeds_btn.setObjectName("secondaryButton")
        self._new_seeds_btn.setToolTip("Re-pick internal seeds")
        self._new_seeds_btn.setMinimumWidth(110)
        self._new_seeds_btn.clicked.connect(self._on_new_seeds)
        self._new_seeds_btn.setVisible(False)
        header_layout.addWidget(self._new_seeds_btn)

        layout.addWidget(header_frame)

        # ─────────────────────────────────────────────────────────────────────
        # Mode inputs card - contains mode-specific panels + progress
        # ─────────────────────────────────────────────────────────────────────
        self._inputs_frame = QFrame()
        self._inputs_frame.setObjectName("inputsCard")
        self._inputs_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        inputs_layout = QVBoxLayout(self._inputs_frame)
        inputs_layout.setContentsMargins(12, 8, 12, 8)
        inputs_layout.setSpacing(6)

        # Mode-specific panel stack
        self._mode_stack = QStackedWidget()
        self._mode_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Artist mode panel
        self._artist_panel = ArtistModePanel()
        if self._db_completer:
            self._artist_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._artist_panel)

        # Seeds mode panel
        self._seeds_panel = SeedsModePanel(db_path=self._db_path)
        if self._db_completer:
            self._seeds_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._seeds_panel)

        inputs_layout.addWidget(self._mode_stack)

        # Progress bar (inline at bottom of inputs card)
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.valueChanged.connect(self._update_progress_visibility)
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setObjectName("stageLabel")
        self._stage_label.setFixedWidth(180)
        self._stage_label.setVisible(False)
        progress_row.addWidget(self._stage_label)

        inputs_layout.addLayout(progress_row)

        layout.addWidget(self._inputs_frame)
        QTimer.singleShot(0, self._apply_mode_sizing)

    def _create_vsep(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setObjectName("vsep")
        return sep

    def _create_menu_button(
        self,
        options: list[tuple[str, object]],
        default_value: object,
        width: int,
        tooltip: str,
    ) -> QToolButton:
        """Create a menu-backed button that behaves like a compact dropdown."""
        button = QToolButton()
        button.setObjectName("comboButton")
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setMinimumWidth(width)
        button.setToolTip(tooltip)

        menu = QMenu(button)
        group = QActionGroup(button)
        group.setExclusive(True)

        def apply_action(action: QAction) -> None:
            button.setText(f"{action.text()} ▼")
            button.setProperty("value", action.data())
            action.setChecked(True)

        selected = False
        for label, value in options:
            action = QAction(label, button)
            action.setCheckable(True)
            action.setData(value)
            group.addAction(action)
            menu.addAction(action)
            if value == default_value and not selected:
                apply_action(action)
                selected = True

        if not selected and group.actions():
            apply_action(group.actions()[0])

        group.triggered.connect(apply_action)
        button.setMenu(menu)
        return button

    def _on_mode_changed(self, _: int | None = None) -> None:
        """Handle mode radio button change."""
        mode = self._get_current_mode()
        index = {"artist": 0, "seeds": 1}.get(mode, 0)
        self._mode_stack.setCurrentIndex(index)
        QTimer.singleShot(0, self._apply_mode_sizing)
        if self._has_run:
            self._has_run = False
            self._update_run_controls()
        self.mode_changed.emit(mode)

    def _on_recency_toggled(self, checked: bool) -> None:
        """Handle recency checkbox toggle."""
        self._recency_days.setEnabled(checked)
        self._recency_plays.setEnabled(checked)

    def _get_current_mode(self) -> ModeType:
        """Get currently selected mode."""
        mode_id = self._mode_combo.currentData()
        if mode_id in ("artist", "seeds"):
            return mode_id
        return "artist"

    def get_current_mode(self) -> ModeType:
        """Public accessor for current mode."""
        return self._get_current_mode()

    def _update_progress_visibility(self) -> None:
        """Show progress only while running."""
        show = self._progress_bar.value() > 0
        self._progress_bar.setVisible(show)
        self._stage_label.setVisible(show)
        if show:
            self._apply_mode_sizing()

    def _apply_mode_sizing(self) -> None:
        """Adjust the inputs card height to the current mode content."""
        current = self._mode_stack.currentWidget()
        if not current:
            return
        current.adjustSize()
        self._mode_stack.adjustSize()
        self._inputs_frame.adjustSize()
        self.adjustSize()
        panel_height = current.sizeHint().height()
        if panel_height > 0:
            self._mode_stack.setFixedHeight(panel_height)
        self._inputs_frame.setMaximumHeight(self._inputs_frame.sizeHint().height())
        self.updateGeometry()

    def _update_run_controls(self) -> None:
        mode = self._get_current_mode()
        has_run = self._has_run
        self._generate_btn.setText("Regenerate" if has_run else "Generate")
        self._new_seeds_btn.setVisible(has_run and mode == "artist")

    def _get_diversity_gamma(self) -> float:
        index = self._diversity_slider.value()
        return float(self._diversity_values[index])

    def _on_diversity_changed(self, value: int) -> None:
        self._diversity_value.setText(self._diversity_levels[value])

    def build_ui_state(self) -> UIStateModel:
        """Construct UIStateModel from current UI state."""
        mode = self._get_current_mode()

        return UIStateModel(
            mode=mode,
            genre_mode=self._mode_sliders.get_genre_mode(),
            sonic_mode=self._mode_sliders.get_sonic_mode(),
            track_count=int(self._length_combo.property("value") or 30),
            diversity_gamma=self._get_diversity_gamma(),
            recency_enabled=self._recency_check.isChecked(),
            recency_days=int(self._recency_days.property("value") or 14),
            recency_plays_threshold=int(self._recency_plays.property("value") or 1),
            artist_spacing=str(self._spacing_combo.property("value") or "normal"),
            artist_queries=self._artist_panel.get_artists() if mode == "artist" else [],
            artist_presence=self._artist_panel.get_presence() if mode == "artist" else "medium",
            artist_variety=self._artist_panel.get_variety() if mode == "artist" else "balanced",
            seed_track_ids=self._seeds_panel.get_seed_track_ids() if mode == "seeds" else [],
            seed_auto_order=self._seeds_panel.get_auto_order() if mode == "seeds" else True,
        )

    def get_seed_artist_keys(self) -> List[str]:
        """Get artist keys for seeds (for policy evaluation)."""
        if self._get_current_mode() != "seeds":
            return []
        return self._seeds_panel.get_seed_artist_keys()

    def get_seed_display_strings(self) -> List[str]:
        """
        Get seed display strings for backend communication.

        The backend expects "Title - Artist" format strings, not raw track IDs.
        Use this method when passing seeds to the worker/generator.
        """
        if self._get_current_mode() != "seeds":
            return []
        return self._seeds_panel.get_seed_display_strings()

    @Slot()
    def _on_generate(self) -> None:
        """Handle Generate button click."""
        if self._is_generating:
            return

        ui_state = self.build_ui_state()
        if self._has_run:
            self.regenerate_requested.emit(asdict(ui_state))
        else:
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


    def set_generating(self, is_generating: bool) -> None:
        """Update UI state for generation in progress."""
        self._is_generating = is_generating
        self._generate_btn.setEnabled(not is_generating)
        self._new_seeds_btn.setEnabled(not is_generating)

        # Disable mode switching during generation
        self._mode_combo.setEnabled(not is_generating)

    def mark_run_complete(self) -> None:
        self._has_run = True
        self._update_run_controls()

    def set_progress(self, value: int, stage: str = "") -> None:
        """Update progress bar and stage label."""
        self._progress_bar.setValue(value)
        self._stage_label.setText(stage)
        self._update_progress_visibility()

    def reset_progress(self) -> None:
        """Reset progress bar to initial state."""
        self._progress_bar.setValue(0)
        self._stage_label.setText("")
        self._update_progress_visibility()

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
