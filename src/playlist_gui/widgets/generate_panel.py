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
    QGridLayout,
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
from .cohesion_slider import CohesionSlider
from .mode_panels import ArtistModePanel, GenreModePanel, HistoryModePanel, SeedsModePanel
from .seed_chips import SeedChip


ModeType = Literal["artist", "genre", "seeds", "history"]

CONTROL_GROUP_HEIGHT = 66
CONTROL_GROUP_TITLE_WIDTH = 54
HEADER_BREAKPOINT_WIDTH = 1780
HEADER_NARROW_BREAKPOINT_WIDTH = 1040
HEADER_GROUP_WIDTHS = {
    "mode": 168,
    "cohesion": 226,
    "matching": 292,
    "length": 134,
    "freshness": 296,
    "spacing": 292,
    "actions": 288,
}


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
        self._is_reflowing_header = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._setup_ui()
        self._update_run_controls()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ─────────────────────────────────────────────────────────────────────
        # Header toolbar - compact responsive groups for primary generation controls.
        # ─────────────────────────────────────────────────────────────────────
        self._header_frame = QFrame()
        header_frame = self._header_frame
        header_frame.setObjectName("headerFrame")
        header_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_frame.setMinimumHeight(CONTROL_GROUP_HEIGHT + 12)
        header_frame.setMaximumHeight(CONTROL_GROUP_HEIGHT + 12)
        header_layout = QVBoxLayout(header_frame)
        self._header_layout = header_layout
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(8)
        self._control_groups: dict[str, QFrame] = {}
        self._header_group_order = [
            "mode",
            "cohesion",
            "matching",
            "length",
            "freshness",
            "spacing",
            "actions",
        ]
        self._header_group_positions: dict[str, tuple[int, int, int, int]] = {}
        self._header_row_count = 1
        self._header_rows: list[tuple[QWidget, QHBoxLayout]] = []
        for _ in range(3):
            row_widget = QWidget()
            row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_widget.setMinimumHeight(CONTROL_GROUP_HEIGHT)
            row_widget.setMaximumHeight(CONTROL_GROUP_HEIGHT)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            header_layout.addWidget(row_widget)
            row_widget.hide()
            self._header_rows.append((row_widget, row_layout))

        # Mode selector (dropdown)
        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Artist", "artist")
        self._mode_combo.addItem("Genre", "genre")
        self._mode_combo.addItem("Seeds", "seeds")
        self._mode_combo.addItem("History", "history")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._mode_combo.setMinimumWidth(108)
        self._mode_combo.setMaximumWidth(128)
        self._mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        mode_layout.addWidget(self._mode_combo)

        self._create_control_group("mode", "Mode", mode_container)

        # Overall cohesion (pier-bridge beam tuning)
        self._cohesion_slider = CohesionSlider()
        self._create_control_group("cohesion", "Overall\nCohesion", self._cohesion_slider)

        # Genre/Sonic/Pace mode sliders (stacked)
        self._mode_sliders = ModeSliders()
        self._create_control_group("matching", "Matching", self._mode_sliders)

        # Length dropdown
        length_container = QWidget()
        length_layout = QHBoxLayout(length_container)
        length_layout.setContentsMargins(0, 0, 0, 0)
        length_layout.setSpacing(4)

        self._length_combo = self._create_menu_button(
            options=[("20", 20), ("30", 30), ("40", 40), ("50", 50)],
            default_value=30,
            width=72,
            tooltip="Number of tracks in the generated playlist",
        )
        length_layout.addWidget(self._length_combo)
        self._create_control_group("length", "Length", length_container)

        # Recency filter (compact)
        self._recency_container = QWidget()
        recency_layout = QGridLayout(self._recency_container)
        recency_layout.setContentsMargins(0, 0, 0, 0)
        recency_layout.setHorizontalSpacing(5)
        recency_layout.setVerticalSpacing(4)
        recency_layout.setColumnMinimumWidth(0, 18)

        self._recency_check = QCheckBox("")
        self._recency_check.setObjectName("headerCheckBox")
        self._recency_check.setChecked(True)
        self._recency_check.setToolTip("Exclude recently played tracks")
        self._recency_check.toggled.connect(self._on_recency_toggled)
        recency_layout.addWidget(self._recency_check, 0, 0, alignment=Qt.AlignCenter)

        self._recency_days = self._create_menu_button(
            options=[("7d", 7), ("14d", 14), ("30d", 30), ("60d", 60), ("90d", 90)],
            default_value=14,
            width=72,
            tooltip="Lookback days",
        )
        recency_layout.addWidget(self._recency_days, 0, 1)

        self._recency_plays = self._create_menu_button(
            options=[("1+", 1), ("2+", 2), ("3+", 3), ("5+", 5), ("10+", 10)],
            default_value=1,
            width=62,
            tooltip="Min plays to exclude",
        )
        recency_layout.addWidget(self._recency_plays, 0, 2)

        self._recency_seed_row = QWidget()
        recency_seed_layout = QHBoxLayout(self._recency_seed_row)
        recency_seed_layout.setContentsMargins(5, 0, 0, 0)
        recency_seed_layout.setSpacing(0)

        self._recency_seed_check = QCheckBox("Don't seed from recent plays")
        self._recency_seed_check.setObjectName("headerCheckBox")
        self._recency_seed_check.setToolTip(
            "Also prevent recently played artist seed tracks from being chosen as playlist piers"
        )
        recency_seed_layout.addWidget(self._recency_seed_check)
        recency_seed_layout.addStretch(1)
        recency_layout.addWidget(self._recency_seed_row, 1, 0, 1, 3)

        self._create_control_group("freshness", "Freshness", self._recency_container)

        # Artist Gap + Diversity — combined stacked card
        spacing_combined = QWidget()
        spacing_combined_layout = QVBoxLayout(spacing_combined)
        spacing_combined_layout.setContentsMargins(0, 0, 0, 0)
        spacing_combined_layout.setSpacing(2)

        gap_row = QHBoxLayout()
        gap_row.setContentsMargins(0, 0, 0, 0)
        gap_row.setSpacing(6)
        gap_label = QLabel("Artist Gap:")
        gap_label.setObjectName("controlLabel")
        gap_label.setMinimumWidth(68)
        gap_row.addWidget(gap_label)

        self._spacing_levels = ["loose", "normal", "strong", "very_strong"]
        self._spacing_labels = {"loose": "Loose", "normal": "Normal", "strong": "Strong", "very_strong": "Very Strong"}

        self._spacing_slider = QSlider(Qt.Horizontal)
        self._spacing_slider.setMinimum(0)
        self._spacing_slider.setMaximum(len(self._spacing_levels) - 1)
        self._spacing_slider.setValue(self._spacing_levels.index("normal"))
        self._spacing_slider.setTickPosition(QSlider.NoTicks)
        self._spacing_slider.setMinimumWidth(90)
        self._spacing_slider.setMaximumWidth(130)
        self._spacing_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._spacing_slider.setToolTip("Minimum tracks between repeated artists\nLoose=3, Normal=6, Strong=9, Very Strong=12")
        self._spacing_slider.valueChanged.connect(self._on_spacing_changed)
        gap_row.addWidget(self._spacing_slider)

        self._spacing_value = QLabel(self._spacing_labels["normal"])
        self._spacing_value.setObjectName("modeValue")
        self._spacing_value.setMinimumWidth(72)
        self._spacing_value.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        gap_row.addWidget(self._spacing_value)
        spacing_combined_layout.addLayout(gap_row)

        div_row = QHBoxLayout()
        div_row.setContentsMargins(0, 0, 0, 0)
        div_row.setSpacing(6)
        div_label = QLabel("Diversity:")
        div_label.setObjectName("controlLabel")
        div_label.setMinimumWidth(68)
        div_row.addWidget(div_label)

        self._diversity_levels = ["Very Low", "Low", "Normal", "High", "Very High", "One Each"]
        self._diversity_values = [0.00, 0.02, 0.04, 0.06, 0.08, 0.08]

        self._diversity_slider = QSlider(Qt.Horizontal)
        self._diversity_slider.setMinimum(0)
        self._diversity_slider.setMaximum(len(self._diversity_levels) - 1)
        self._diversity_slider.setValue(2)
        self._diversity_slider.setTickPosition(QSlider.NoTicks)
        self._diversity_slider.setMinimumWidth(90)
        self._diversity_slider.setMaximumWidth(140)
        self._diversity_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._diversity_slider.setToolTip(
            "Soft bonus for selecting new artists\n"
            "Higher values encourage more variety"
        )
        self._diversity_slider.valueChanged.connect(self._on_diversity_changed)
        div_row.addWidget(self._diversity_slider)

        self._diversity_value = QLabel(self._diversity_levels[2])
        self._diversity_value.setObjectName("diversityValue")
        self._diversity_value.setMinimumWidth(72)
        self._diversity_value.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        div_row.addWidget(self._diversity_value)
        spacing_combined_layout.addLayout(div_row)

        self._create_control_group("spacing", "Spacing", spacing_combined)

        # Action buttons in header
        actions_container = QWidget()
        actions_layout = QHBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setObjectName("primaryButton")
        self._generate_btn.setMinimumWidth(118)
        self._generate_btn.clicked.connect(self._on_generate)
        actions_layout.addWidget(self._generate_btn)

        self._new_seeds_btn = QPushButton("New Seeds")
        self._new_seeds_btn.setObjectName("secondaryButton")
        self._new_seeds_btn.setToolTip("Re-pick internal seeds")
        self._new_seeds_btn.setMinimumWidth(86)
        self._new_seeds_btn.clicked.connect(self._on_new_seeds)
        self._new_seeds_btn.setVisible(False)
        actions_layout.addWidget(self._new_seeds_btn)
        self._create_control_group("actions", "Actions", actions_container)
        self._reflow_header_groups(HEADER_BREAKPOINT_WIDTH)

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

        # Genre mode panel
        self._genre_panel = GenreModePanel()
        if self._db_completer:
            self._genre_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._genre_panel)

        # Seeds mode panel
        self._seeds_panel = SeedsModePanel(db_path=self._db_path)
        if self._db_completer:
            self._seeds_panel.set_completer_data(self._db_completer)
        self._mode_stack.addWidget(self._seeds_panel)

        # History mode panel
        self._history_panel = HistoryModePanel()
        self._mode_stack.addWidget(self._history_panel)

        inputs_layout.addWidget(self._mode_stack)

        # Progress bar (inline at bottom of inputs card)
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumHeight(16)
        self._progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._progress_bar.valueChanged.connect(self._update_progress_visibility)
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setObjectName("stageLabel")
        self._stage_label.setMinimumWidth(140)
        self._stage_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._stage_label.setVisible(False)
        progress_row.addWidget(self._stage_label)

        inputs_layout.addLayout(progress_row)

        self._validation_label = QLabel("")
        self._validation_label.setObjectName("validationBanner")
        self._validation_label.setWordWrap(True)
        self._validation_label.setVisible(False)
        inputs_layout.addWidget(self._validation_label)

        layout.addWidget(self._inputs_frame)
        QTimer.singleShot(0, self._apply_mode_sizing)

    def resizeEvent(self, event) -> None:
        """Reflow header cards when the panel width changes."""
        super().resizeEvent(event)
        if not self._is_reflowing_header:
            self._reflow_header_groups(self._header_frame.width() or self.width())

    def _create_vsep(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setObjectName("vsep")
        return sep

    def _create_control_group(self, key: str, title: str, content: QWidget) -> QFrame:
        """Create a compact visual group for one header control category."""
        group = QFrame()
        group.setObjectName("controlGroup")
        group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        group.setMinimumWidth(HEADER_GROUP_WIDTHS.get(key, 140))
        group.setMaximumWidth(HEADER_GROUP_WIDTHS.get(key, 140))
        group.setMinimumHeight(CONTROL_GROUP_HEIGHT)
        group.setMaximumHeight(CONTROL_GROUP_HEIGHT)
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(8, 5, 8, 5)
        group_layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setObjectName("controlGroupTitle")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(CONTROL_GROUP_TITLE_WIDTH)
        title_label.setMaximumWidth(CONTROL_GROUP_TITLE_WIDTH)
        title_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        group_layout.addWidget(title_label)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)
        content_row.addWidget(content, 0, Qt.AlignVCenter)
        content_row.addStretch(1)
        group_layout.addLayout(content_row, 1)

        self._control_groups[key] = group
        setattr(self, f"_{key}_group_title", title_label)
        return group

    def _clear_header_rows(self) -> None:
        for row_widget, row_layout in self._header_rows:
            while row_layout.count():
                item = row_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    row_layout.removeWidget(widget)
            row_widget.hide()

    def _header_required_width(self, keys: list[str]) -> int:
        margins = self._header_layout.contentsMargins()
        spacing = self._header_rows[0][1].spacing()
        return (
            sum(HEADER_GROUP_WIDTHS.get(key, 140) for key in keys)
            + max(0, len(keys) - 1) * spacing
            + margins.left()
            + margins.right()
        )

    def _reflow_header_groups(self, available_width: int) -> None:
        """Pack header cards into left-aligned rows that fit the available width."""
        if self._is_reflowing_header:
            return
        self._is_reflowing_header = True
        try:
            self._clear_header_rows()
            self._header_group_positions.clear()

            medium_rows = [
                self._header_group_order[:4],
                self._header_group_order[4:],
            ]
            narrow_rows = [
                ["mode", "cohesion", "length"],
                ["matching", "spacing"],
                ["freshness", "actions"],
            ]
            if available_width >= self._header_required_width(self._header_group_order):
                rows = [self._header_group_order]
            elif available_width >= max(
                self._header_required_width(row) for row in medium_rows
            ):
                rows = medium_rows
            else:
                rows = narrow_rows

            self._header_row_count = len(rows)
            for row_index, row_keys in enumerate(rows):
                row_widget, row_layout = self._header_rows[row_index]
                row_widget.show()
                for column_index, key in enumerate(row_keys):
                    group = self._control_groups[key]
                    row_layout.addWidget(group, 0, Qt.AlignLeft | Qt.AlignVCenter)
                    self._header_group_positions[key] = (row_index, column_index, 1, 1)
                row_layout.addStretch(1)

            margins = self._header_layout.contentsMargins()
            minimum_height = (
                self._header_row_count * CONTROL_GROUP_HEIGHT
                + max(0, self._header_row_count - 1) * self._header_layout.spacing()
                + margins.top()
                + margins.bottom()
            )
            self._header_frame.setMinimumHeight(minimum_height)
            self._header_frame.setMaximumHeight(minimum_height)
            self._sync_panel_height()
            self._header_frame.updateGeometry()
        finally:
            self._is_reflowing_header = False

    def _sync_panel_height(self) -> None:
        """Keep the generation panel tight to its header and active inputs."""
        if not hasattr(self, "_inputs_frame"):
            return
        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        total_height = (
            self._header_frame.maximumHeight()
            + self._inputs_frame.maximumHeight()
            + layout.spacing()
            + margins.top()
            + margins.bottom()
        )
        if self.minimumHeight() != total_height:
            self.setMinimumHeight(total_height)
        if self.maximumHeight() != total_height:
            self.setMaximumHeight(total_height)
        self.updateGeometry()

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

    @staticmethod
    def _set_menu_button_value(button: QToolButton, value: object) -> None:
        """Set a menu button's selection to the action matching value."""
        for action in button.menu().actions():
            if action.data() == value:
                action.trigger()
                return

    def _on_mode_changed(self, _: int | None = None) -> None:
        """Handle mode radio button change."""
        mode = self._get_current_mode()
        index = {"artist": 0, "genre": 1, "seeds": 2, "history": 3}.get(mode, 0)
        self._mode_stack.setCurrentIndex(index)
        self._update_recency_seed_control()
        self.clear_validation_message()
        QTimer.singleShot(0, self._apply_mode_sizing)
        if self._has_run:
            self._has_run = False
            self._update_run_controls()
        self.mode_changed.emit(mode)

    def _on_recency_toggled(self, checked: bool) -> None:
        """Handle recency checkbox toggle."""
        self._recency_days.setEnabled(checked)
        self._recency_plays.setEnabled(checked)
        self._update_recency_seed_control()

    def _update_recency_seed_control(self) -> None:
        """Show seed freshness only where artist seed selection is automatic."""
        is_artist_mode = self._get_current_mode() == "artist"
        self._recency_seed_row.setVisible(is_artist_mode)
        self._recency_seed_check.setVisible(is_artist_mode)
        self._recency_seed_check.setEnabled(
            is_artist_mode and self._recency_check.isChecked()
        )

    def _get_current_mode(self) -> ModeType:
        """Get currently selected mode."""
        mode_id = self._mode_combo.currentData()
        if mode_id in ("artist", "genre", "seeds", "history"):
            return mode_id
        return "artist"

    def get_current_mode(self) -> ModeType:
        """Public accessor for current mode."""
        return self._get_current_mode()

    def set_current_mode(self, mode: str) -> None:
        """Set the active generation mode by stable mode id."""
        index = self._mode_combo.findData(mode)
        if index < 0:
            return
        self._mode_combo.setCurrentIndex(index)

    def apply_ui_state(self, state: UIStateModel) -> None:
        """Restore all controls from a UIStateModel (inverse of build_ui_state)."""
        self.set_current_mode(state.mode)

        self._cohesion_slider.set_cohesion_mode(state.cohesion_mode)
        self._mode_sliders.set_genre_mode(state.genre_mode)
        self._mode_sliders.set_sonic_mode(state.sonic_mode)
        self._mode_sliders.set_pace_mode(state.pace_mode)

        self._set_menu_button_value(self._length_combo, state.track_count)

        # Diversity: find the slider position matching gamma + mode
        if state.artist_diversity_mode == "one_per_artist":
            self._diversity_slider.setValue(len(self._diversity_levels) - 1)
        else:
            closest = min(
                range(len(self._diversity_values) - 1),
                key=lambda i: abs(self._diversity_values[i] - state.diversity_gamma),
            )
            self._diversity_slider.setValue(closest)

        # Spacing
        if state.artist_spacing in self._spacing_levels:
            self._spacing_slider.setValue(self._spacing_levels.index(state.artist_spacing))

        # Recency
        self._recency_check.setChecked(state.recency_enabled)
        self._set_menu_button_value(self._recency_days, state.recency_days)
        self._set_menu_button_value(self._recency_plays, state.recency_plays_threshold)
        self._recency_seed_check.setChecked(bool(state.exclude_seed_tracks_from_recency))
        self._update_recency_seed_control()

        # Mode-specific controls
        if state.mode == "artist":
            if state.artist_queries:
                self._artist_panel.set_primary_artist(state.artist_queries[0])
            self._artist_panel.set_presence(state.artist_presence)
            self._artist_panel.set_variety(state.artist_variety)
            self._artist_panel.set_include_collaborations(state.include_collaborations)
        elif state.mode == "genre":
            self._genre_panel.set_genre(state.genre_query)
        elif state.mode == "seeds":
            self._seeds_panel.set_auto_order(state.seed_auto_order)

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
            self._mode_stack.setMinimumHeight(panel_height)
            self._mode_stack.setMaximumHeight(panel_height)

            margins = self._inputs_frame.layout().contentsMargins()
            inputs_height = panel_height + margins.top() + margins.bottom()
            if self._progress_bar.isVisible() or self._stage_label.isVisible():
                inputs_height += (
                    self._inputs_frame.layout().spacing()
                    + max(self._progress_bar.sizeHint().height(), self._stage_label.sizeHint().height())
                )
            self._inputs_frame.setMinimumHeight(inputs_height)
            self._inputs_frame.setMaximumHeight(inputs_height)
        self._sync_panel_height()

    def _update_run_controls(self) -> None:
        mode = self._get_current_mode()
        has_run = self._has_run
        self._generate_btn.setText("Regenerate" if has_run else "Generate")
        self._new_seeds_btn.setVisible(has_run and mode == "artist")

    def show_validation_message(self, message: str) -> None:
        """Show inline generation validation feedback."""
        self._validation_label.setText(message)
        self._validation_label.setVisible(bool(message))
        self._apply_mode_sizing()

    def clear_validation_message(self) -> None:
        """Clear inline generation validation feedback."""
        self._validation_label.setText("")
        self._validation_label.setVisible(False)
        self._apply_mode_sizing()

    def _get_diversity_gamma(self) -> float:
        index = self._diversity_slider.value()
        return float(self._diversity_values[index])

    def _get_artist_diversity_mode(self) -> str:
        if self._diversity_slider.value() == len(self._diversity_levels) - 1:
            return "one_per_artist"
        return "weighted"

    def _on_spacing_changed(self, value: int) -> None:
        self._spacing_value.setText(self._spacing_labels[self._spacing_levels[value]])

    def _on_diversity_changed(self, value: int) -> None:
        self._diversity_value.setText(self._diversity_levels[value])

    def build_ui_state(self) -> UIStateModel:
        """Construct UIStateModel from current UI state."""
        mode = self._get_current_mode()

        return UIStateModel(
            mode=mode,
            cohesion_mode=self._cohesion_slider.get_cohesion_mode(),
            genre_mode=self._mode_sliders.get_genre_mode(),
            sonic_mode=self._mode_sliders.get_sonic_mode(),
            pace_mode=self._mode_sliders.get_pace_mode(),
            track_count=int(self._length_combo.property("value") or 30),
            diversity_gamma=self._get_diversity_gamma(),
            artist_diversity_mode=self._get_artist_diversity_mode(),
            recency_enabled=self._recency_check.isChecked(),
            recency_days=int(self._recency_days.property("value") or 14),
            recency_plays_threshold=int(self._recency_plays.property("value") or 1),
            exclude_seed_tracks_from_recency=(
                self._recency_seed_check.isChecked() if mode == "artist" else False
            ),
            artist_spacing=self._spacing_levels[self._spacing_slider.value()],
            artist_queries=self._artist_panel.get_artists() if mode == "artist" else [],
            artist_presence=self._artist_panel.get_presence() if mode == "artist" else "medium",
            artist_variety=self._artist_panel.get_variety() if mode == "artist" else "balanced",
            include_collaborations=(
                self._artist_panel.get_include_collaborations() if mode == "artist" else False
            ),
            genre_query=self._genre_panel.get_genre() if mode == "genre" else "",
            seed_track_ids=self._seeds_panel.get_seed_track_ids() if mode == "seeds" else [],
            seed_auto_order=self._seeds_panel.get_auto_order() if mode == "seeds" else True,
        )

    def get_seed_artist_keys(self) -> List[str]:
        """Get artist keys for seeds (for policy evaluation)."""
        if self._get_current_mode() != "seeds":
            return []
        return self._seeds_panel.get_seed_artist_keys()

    def get_seed_track_ids(self) -> List[str]:
        """Get list of seed track IDs for exact track matching."""
        if self._get_current_mode() != "seeds":
            return []
        return self._seeds_panel.get_seed_track_ids()

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

        self.clear_validation_message()
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

        self.clear_validation_message()
        ui_state = self.build_ui_state()
        self.regenerate_requested.emit(asdict(ui_state))

    @Slot()
    def _on_new_seeds(self) -> None:
        """Handle New Seeds button click."""
        if self._is_generating:
            return

        self.clear_validation_message()
        ui_state = self.build_ui_state()
        self.new_seeds_requested.emit(asdict(ui_state))


    def set_generating(self, is_generating: bool) -> None:
        """Update UI state for generation in progress."""
        self._is_generating = is_generating
        if is_generating:
            self.clear_validation_message()
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
        self._genre_panel.set_completer_data(completer)
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
