"""
Advanced Settings Panel - Schema-driven settings UI with normalized sliders

Renders controls based on the settings schema, automatically handling:
- Grouped controls (collapsible sections)
- Normalized weight groups (sliders that sum to 1.0)
- Different control types (sliders, spinboxes, checkboxes, combos)
- Expandable info text for each setting
- Modified highlighting for overridden settings
- Per-setting and group-level reset functionality
"""
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config.config_model import ConfigModel
from ..config.settings_schema import (
    SettingSpec,
    SettingType,
    get_normalize_groups,
    get_visible_settings_by_group,
)


# Styles for modified state
MODIFIED_LABEL_STYLE = "font-weight: bold; color: #1a5fb4;"
MODIFIED_ROW_STYLE = "background-color: #e8f0fc; border-radius: 3px; padding: 2px;"
NORMAL_LABEL_STYLE = ""
NORMAL_ROW_STYLE = ""

# Direct control styles for modified state (ensures readable text)
MODIFIED_SPINBOX_STYLE = """
    QSpinBox, QDoubleSpinBox {
        color: #333;
        background-color: white;
        border: 1px solid #1a5fb4;
    }
"""
MODIFIED_COMBO_STYLE = """
    QComboBox {
        color: #333;
        background-color: white;
        border: 1px solid #1a5fb4;
    }
    QComboBox QAbstractItemView {
        color: #333;
        background-color: white;
    }
"""
MODIFIED_LINEEDIT_STYLE = """
    QLineEdit {
        color: #333;
        background-color: white;
        border: 1px solid #1a5fb4;
    }
"""
NORMAL_CONTROL_STYLE = ""

RESET_BTN_STYLE = """
    QToolButton {
        background-color: transparent;
        border: none;
        color: #888;
        font-size: 14px;
        padding: 2px;
    }
    QToolButton:hover {
        background-color: #e0e0e0;
        color: #333;
        border-radius: 3px;
    }
    QToolButton:disabled {
        color: #ccc;
    }
"""

INFO_BTN_STYLE = """
    QToolButton {
        background-color: #e8e8e8;
        border: 1px solid #aaa;
        border-radius: 9px;
        font-weight: bold;
        font-size: 10px;
        color: #666;
    }
    QToolButton:hover { background-color: #d8d8d8; }
    QToolButton:checked { background-color: #b0c4de; }
"""

GROUP_RESET_BTN_STYLE = """
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 2px 8px;
        font-size: 11px;
        color: #555;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
        border-color: #999;
    }
    QPushButton:disabled {
        color: #aaa;
        background-color: #f8f8f8;
    }
"""


class NormalizedSliderGroup(QWidget):
    """
    A group of sliders whose values must sum to 1.0.

    When one slider changes, others adjust proportionally.
    Shows numeric values, modified indicators, and reset buttons.
    """

    values_changed = Signal(dict)  # {key_path: value, ...}
    override_changed = Signal()  # Emitted when any override state changes

    def __init__(
        self,
        specs: List[SettingSpec],
        config_model: ConfigModel,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.specs = specs
        self.config_model = config_model
        self._updating = False  # Prevent recursion during normalization

        self._sliders: Dict[str, QSlider] = {}
        self._labels: Dict[str, QLabel] = {}
        self._name_labels: Dict[str, QLabel] = {}
        self._reset_btns: Dict[str, QToolButton] = {}
        self._modified_dots: Dict[str, QLabel] = {}
        self._row_containers: Dict[str, QWidget] = {}
        self._info_labels: Dict[str, QLabel] = {}

        self._setup_ui()
        self._load_values()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for spec in self.specs:
            # Row container for modified styling
            row_container = QWidget()
            self._row_containers[spec.key_path] = row_container
            row_layout = QVBoxLayout(row_container)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(2)

            # Main slider row
            row = QHBoxLayout()
            row.setSpacing(8)

            # Modified indicator dot
            mod_dot = QLabel("●")
            mod_dot.setFixedWidth(12)
            mod_dot.setStyleSheet("color: #1a5fb4; font-size: 10px;")
            mod_dot.setVisible(False)
            mod_dot.setToolTip("Modified from base config")
            row.addWidget(mod_dot)
            self._modified_dots[spec.key_path] = mod_dot

            # Label
            label = QLabel(spec.label)
            label.setFixedWidth(80)
            row.addWidget(label)
            self._name_labels[spec.key_path] = label

            # Slider (0-100 for 0.0-1.0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setSingleStep(int((spec.step or 0.05) * 100))
            slider.setPageStep(10)
            if spec.tooltip:
                slider.setToolTip(spec.tooltip)
            slider.valueChanged.connect(lambda v, s=spec: self._on_slider_changed(s.key_path, v))
            row.addWidget(slider, stretch=1)
            self._sliders[spec.key_path] = slider

            # Value label
            value_label = QLabel("0.00")
            value_label.setFixedWidth(40)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(value_label)
            self._labels[spec.key_path] = value_label

            # Reset button
            reset_btn = QToolButton()
            reset_btn.setText("↺")
            reset_btn.setToolTip("Reset to base config value")
            reset_btn.setFixedSize(20, 20)
            reset_btn.setStyleSheet(RESET_BTN_STYLE)
            reset_btn.setEnabled(False)
            reset_btn.clicked.connect(lambda checked, k=spec.key_path: self._on_reset_single(k))
            row.addWidget(reset_btn)
            self._reset_btns[spec.key_path] = reset_btn

            # Info button (if description exists)
            if spec.description:
                info_btn = QToolButton()
                info_btn.setText("?")
                info_btn.setToolTip("Show details")
                info_btn.setFixedSize(18, 18)
                info_btn.setCheckable(True)
                info_btn.setStyleSheet(INFO_BTN_STYLE)
                info_btn.clicked.connect(lambda checked, k=spec.key_path: self._toggle_info(k))
                row.addWidget(info_btn)

            row_layout.addLayout(row)

            # Description label (hidden by default)
            if spec.description:
                desc_label = QLabel(spec.description)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("""
                    QLabel {
                        background-color: #f8f8f8;
                        border: 1px solid #e0e0e0;
                        border-radius: 4px;
                        padding: 6px;
                        color: #444;
                        font-size: 11px;
                        margin-left: 20px;
                    }
                """)
                desc_label.setVisible(False)
                row_layout.addWidget(desc_label)
                self._info_labels[spec.key_path] = desc_label

            layout.addWidget(row_container)

        # Sum display and normalize button
        sum_row = QHBoxLayout()
        sum_row.setSpacing(8)

        self._sum_label = QLabel("Sum: 1.00")
        self._sum_label.setStyleSheet("font-weight: bold;")
        sum_row.addWidget(self._sum_label)

        sum_row.addStretch()

        normalize_btn = QPushButton("Normalize")
        normalize_btn.setFixedWidth(80)
        normalize_btn.clicked.connect(self._normalize_all)
        normalize_btn.setToolTip("Adjust all weights to sum to 1.0")
        sum_row.addWidget(normalize_btn)

        layout.addLayout(sum_row)

    def _toggle_info(self, key_path: str) -> None:
        """Toggle visibility of info label for a slider."""
        if key_path in self._info_labels:
            label = self._info_labels[key_path]
            label.setVisible(not label.isVisible())

    def _load_values(self) -> None:
        """Load current values from config model."""
        self._updating = True
        try:
            for spec in self.specs:
                value = self.config_model.get(spec.key_path, spec.default or 0.0)
                self._set_slider_value(spec.key_path, value)
                self._update_modified_state(spec.key_path)
            self._update_sum_display()
        finally:
            self._updating = False

    def _set_slider_value(self, key_path: str, value: float) -> None:
        """Set a slider's value without triggering change events."""
        slider = self._sliders.get(key_path)
        label = self._labels.get(key_path)
        if slider and label:
            int_value = int(round(value * 100))
            slider.blockSignals(True)
            slider.setValue(int_value)
            slider.blockSignals(False)
            label.setText(f"{value:.2f}")

    def _update_modified_state(self, key_path: str) -> None:
        """Update the modified visual state for a single slider."""
        is_modified = self.config_model.has_override(key_path)

        # Update dot visibility
        if key_path in self._modified_dots:
            self._modified_dots[key_path].setVisible(is_modified)

        # Update label style
        if key_path in self._name_labels:
            self._name_labels[key_path].setStyleSheet(
                MODIFIED_LABEL_STYLE if is_modified else NORMAL_LABEL_STYLE
            )

        # Update row background
        if key_path in self._row_containers:
            self._row_containers[key_path].setStyleSheet(
                MODIFIED_ROW_STYLE if is_modified else NORMAL_ROW_STYLE
            )

        # Update reset button
        if key_path in self._reset_btns:
            self._reset_btns[key_path].setEnabled(is_modified)

    def _on_slider_changed(self, key_path: str, int_value: int) -> None:
        """Handle slider value change."""
        if self._updating:
            return

        self._updating = True
        try:
            value = int_value / 100.0
            self.config_model.set(key_path, value)

            # Update the label
            if key_path in self._labels:
                self._labels[key_path].setText(f"{value:.2f}")

            # Normalize the group
            normalize_group = None
            for spec in self.specs:
                if spec.key_path == key_path:
                    normalize_group = spec.normalize_group
                    break

            if normalize_group:
                self.config_model.normalize_group(normalize_group, changed_key=key_path)

                # Update other sliders
                for spec in self.specs:
                    if spec.key_path != key_path:
                        new_value = self.config_model.get(spec.key_path, spec.default or 0.0)
                        self._set_slider_value(spec.key_path, new_value)

            # Update modified state for all sliders in group
            for spec in self.specs:
                self._update_modified_state(spec.key_path)

            self._update_sum_display()

            # Emit changed values
            values = {}
            for spec in self.specs:
                values[spec.key_path] = self.config_model.get(spec.key_path)
            self.values_changed.emit(values)
            self.override_changed.emit()

        finally:
            self._updating = False

    def _on_reset_single(self, key_path: str) -> None:
        """Reset a single slider to base config value."""
        if self._updating:
            return

        self._updating = True
        try:
            self.config_model.clear_override(key_path)
            base_value = self.config_model.get_base_value(key_path)

            # Get spec for default
            for spec in self.specs:
                if spec.key_path == key_path:
                    if base_value is None:
                        base_value = spec.default or 0.0
                    break

            self._set_slider_value(key_path, base_value)
            self._update_modified_state(key_path)
            self._update_sum_display()

            # Emit changes
            values = {}
            for spec in self.specs:
                values[spec.key_path] = self.config_model.get(spec.key_path)
            self.values_changed.emit(values)
            self.override_changed.emit()

        finally:
            self._updating = False

    def _update_sum_display(self) -> None:
        """Update the sum display label."""
        total = 0.0
        for spec in self.specs:
            total += self.config_model.get(spec.key_path, spec.default or 0.0)
        self._sum_label.setText(f"Sum: {total:.2f}")

        # Color code: green if 1.0, red otherwise
        if abs(total - 1.0) < 0.01:
            self._sum_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self._sum_label.setStyleSheet("font-weight: bold; color: red;")

    def _normalize_all(self) -> None:
        """Force normalize all sliders."""
        if not self.specs:
            return

        normalize_group = self.specs[0].normalize_group
        if normalize_group:
            self.config_model.normalize_group(normalize_group)
            self._load_values()

            # Emit changed values
            values = {}
            for spec in self.specs:
                values[spec.key_path] = self.config_model.get(spec.key_path)
            self.values_changed.emit(values)
            self.override_changed.emit()

    def refresh(self) -> None:
        """Refresh values from config model."""
        self._load_values()

    def has_any_override(self) -> bool:
        """Check if any slider in this group has an override."""
        return any(self.config_model.has_override(spec.key_path) for spec in self.specs)

    def reset_all(self) -> None:
        """Reset all sliders in this group to base config values."""
        self._updating = True
        try:
            for spec in self.specs:
                self.config_model.clear_override(spec.key_path)
            self._load_values()
            self.override_changed.emit()
        finally:
            self._updating = False


class SettingControl(QWidget):
    """
    A single setting control based on its spec.

    Renders the appropriate widget type (slider, spinbox, checkbox, combo)
    with modified indicator, reset button, and expandable info.
    """

    value_changed = Signal(str, object)  # key_path, value
    override_changed = Signal()  # Emitted when override state changes

    def __init__(
        self,
        spec: SettingSpec,
        config_model: ConfigModel,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.spec = spec
        self.config_model = config_model
        self._control: Optional[QWidget] = None
        self._desc_label: Optional[QLabel] = None
        self._reset_btn: Optional[QToolButton] = None
        self._modified_dot: Optional[QLabel] = None
        self._name_label: Optional[QLabel] = None

        self._setup_ui()
        self._load_value()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        # Control row
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        # Modified indicator dot
        self._modified_dot = QLabel("●")
        self._modified_dot.setFixedWidth(12)
        self._modified_dot.setStyleSheet("color: #1a5fb4; font-size: 10px;")
        self._modified_dot.setVisible(False)
        self._modified_dot.setToolTip("Modified from base config")
        row.addWidget(self._modified_dot)

        # Label
        self._name_label = QLabel(self.spec.label)
        self._name_label.setFixedWidth(140)
        if self.spec.tooltip:
            self._name_label.setToolTip(self.spec.tooltip)
        row.addWidget(self._name_label)

        # Control based on type
        if self.spec.setting_type == SettingType.BOOL:
            self._control = QCheckBox()
            self._control.stateChanged.connect(self._on_bool_changed)

        elif self.spec.setting_type == SettingType.INT:
            self._control = QSpinBox()
            if self.spec.min_value is not None:
                self._control.setMinimum(int(self.spec.min_value))
            if self.spec.max_value is not None:
                self._control.setMaximum(int(self.spec.max_value))
            if self.spec.step:
                self._control.setSingleStep(int(self.spec.step))
            self._control.valueChanged.connect(self._on_int_changed)

        elif self.spec.setting_type == SettingType.FLOAT:
            # Use double spinbox for non-normalized floats
            self._control = QDoubleSpinBox()
            if self.spec.min_value is not None:
                self._control.setMinimum(self.spec.min_value)
            if self.spec.max_value is not None:
                self._control.setMaximum(self.spec.max_value)
            if self.spec.step:
                self._control.setSingleStep(self.spec.step)
            self._control.setDecimals(3)
            self._control.valueChanged.connect(self._on_float_changed)

        elif self.spec.setting_type == SettingType.CHOICE:
            self._control = QComboBox()
            if self.spec.choices:
                self._control.addItems(self.spec.choices)
            self._control.currentTextChanged.connect(self._on_choice_changed)

        elif self.spec.setting_type == SettingType.STRING:
            from PySide6.QtWidgets import QLineEdit
            self._control = QLineEdit()
            self._control.textChanged.connect(self._on_string_changed)

        if self._control:
            if self.spec.tooltip:
                self._control.setToolTip(self.spec.tooltip)
            row.addWidget(self._control, stretch=1)

        # Reset button
        self._reset_btn = QToolButton()
        self._reset_btn.setText("↺")
        self._reset_btn.setToolTip("Reset to base config value")
        self._reset_btn.setFixedSize(20, 20)
        self._reset_btn.setStyleSheet(RESET_BTN_STYLE)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self._reset_btn)

        # Info button (if description exists)
        if self.spec.description:
            info_btn = QToolButton()
            info_btn.setText("?")
            info_btn.setToolTip("Show details")
            info_btn.setFixedSize(18, 18)
            info_btn.setCheckable(True)
            info_btn.setStyleSheet(INFO_BTN_STYLE)
            info_btn.clicked.connect(self._toggle_info)
            row.addWidget(info_btn)

        main_layout.addLayout(row)

        # Description label (hidden by default)
        if self.spec.description:
            self._desc_label = QLabel(self.spec.description)
            self._desc_label.setWordWrap(True)
            self._desc_label.setStyleSheet("""
                QLabel {
                    background-color: #f8f8f8;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                    padding: 8px;
                    color: #444;
                    font-size: 11px;
                    margin-left: 20px;
                }
            """)
            self._desc_label.setVisible(False)
            main_layout.addWidget(self._desc_label)

    def _toggle_info(self) -> None:
        """Toggle visibility of description label."""
        if self._desc_label:
            self._desc_label.setVisible(not self._desc_label.isVisible())

    def _load_value(self) -> None:
        """Load current value from config model."""
        value = self.config_model.get(self.spec.key_path, self.spec.default)

        if self._control is None:
            return

        self._control.blockSignals(True)

        if self.spec.setting_type == SettingType.BOOL:
            self._control.setChecked(bool(value))
        elif self.spec.setting_type == SettingType.INT:
            self._control.setValue(int(value) if value else 0)
        elif self.spec.setting_type == SettingType.FLOAT:
            self._control.setValue(float(value) if value else 0.0)
        elif self.spec.setting_type == SettingType.CHOICE:
            if value in (self.spec.choices or []):
                self._control.setCurrentText(str(value))
        elif self.spec.setting_type == SettingType.STRING:
            self._control.setText(str(value) if value else "")

        self._control.blockSignals(False)
        self._update_modified_state()

    def _update_modified_state(self) -> None:
        """Update the visual modified state."""
        is_modified = self.config_model.has_override(self.spec.key_path)

        if self._modified_dot:
            self._modified_dot.setVisible(is_modified)

        if self._name_label:
            self._name_label.setStyleSheet(
                MODIFIED_LABEL_STYLE if is_modified else NORMAL_LABEL_STYLE
            )

        if self._reset_btn:
            self._reset_btn.setEnabled(is_modified)

        # Update container background
        self.setStyleSheet(MODIFIED_ROW_STYLE if is_modified else NORMAL_ROW_STYLE)

        # Update control style directly (ensures readable text color)
        if self._control:
            if is_modified:
                if self.spec.setting_type == SettingType.INT:
                    self._control.setStyleSheet(MODIFIED_SPINBOX_STYLE)
                elif self.spec.setting_type == SettingType.FLOAT:
                    self._control.setStyleSheet(MODIFIED_SPINBOX_STYLE)
                elif self.spec.setting_type == SettingType.CHOICE:
                    self._control.setStyleSheet(MODIFIED_COMBO_STYLE)
                elif self.spec.setting_type == SettingType.STRING:
                    self._control.setStyleSheet(MODIFIED_LINEEDIT_STYLE)
            else:
                self._control.setStyleSheet(NORMAL_CONTROL_STYLE)

    def _on_reset(self) -> None:
        """Reset this setting to base config value."""
        self.config_model.clear_override(self.spec.key_path)
        self._load_value()
        self.value_changed.emit(self.spec.key_path, self.config_model.get(self.spec.key_path))
        self.override_changed.emit()

    def _on_bool_changed(self, state: int) -> None:
        value = state == Qt.Checked
        self.config_model.set(self.spec.key_path, value)
        self._update_modified_state()
        self.value_changed.emit(self.spec.key_path, value)
        self.override_changed.emit()

    def _on_int_changed(self, value: int) -> None:
        self.config_model.set(self.spec.key_path, value)
        self._update_modified_state()
        self.value_changed.emit(self.spec.key_path, value)
        self.override_changed.emit()

    def _on_float_changed(self, value: float) -> None:
        self.config_model.set(self.spec.key_path, value)
        self._update_modified_state()
        self.value_changed.emit(self.spec.key_path, value)
        self.override_changed.emit()

    def _on_choice_changed(self, value: str) -> None:
        self.config_model.set(self.spec.key_path, value)
        self._update_modified_state()
        self.value_changed.emit(self.spec.key_path, value)
        self.override_changed.emit()

    def _on_string_changed(self, value: str) -> None:
        self.config_model.set(self.spec.key_path, value)
        self._update_modified_state()
        self.value_changed.emit(self.spec.key_path, value)
        self.override_changed.emit()

    def refresh(self) -> None:
        """Refresh value from config model."""
        self._load_value()

    def has_override(self) -> bool:
        """Check if this setting has an override."""
        return self.config_model.has_override(self.spec.key_path)


class AdvancedSettingsPanel(QScrollArea):
    """
    Advanced settings panel with grouped controls.

    Schema-driven rendering of all settings with proper normalization handling,
    modified highlighting, and reset functionality.
    """

    setting_changed = Signal(str, object)  # key_path, value
    override_changed = Signal()  # Emitted when any override state changes

    def __init__(
        self,
        config_model: ConfigModel,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.config_model = config_model
        self._controls: Dict[str, SettingControl] = {}
        self._group_widgets: Dict[str, NormalizedSliderGroup] = {}
        self._group_reset_btns: Dict[str, QPushButton] = {}
        self._group_boxes: Dict[str, QGroupBox] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        # Make scrollable
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main content widget
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 12)  # left, top, right, bottom margins

        # ─────────────────────────────────────────────────────────────────────
        # Global Reset Button
        # ─────────────────────────────────────────────────────────────────────
        global_reset_row = QHBoxLayout()
        global_reset_row.addStretch()

        self._global_reset_btn = QPushButton("Reset All Overrides")
        self._global_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #fff0f0;
                border: 1px solid #d9534f;
                border-radius: 4px;
                padding: 4px 12px;
                color: #d9534f;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffe0e0;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                border-color: #ccc;
                color: #aaa;
            }
        """)
        self._global_reset_btn.clicked.connect(self._on_global_reset)
        self._global_reset_btn.setToolTip("Reset all settings to base config values")
        global_reset_row.addWidget(self._global_reset_btn)

        layout.addLayout(global_reset_row)

        # ─────────────────────────────────────────────────────────────────────
        # Settings Groups
        # ─────────────────────────────────────────────────────────────────────
        # Use visible settings (excludes secrets)
        groups = get_visible_settings_by_group()
        normalize_groups = get_normalize_groups()

        # Track which settings are handled by normalize groups
        normalized_keys = set()
        for specs in normalize_groups.values():
            for spec in specs:
                normalized_keys.add(spec.key_path)

        # Create groups in order
        group_order = [
            "Playlist Settings",
            "Pipeline",
            "Hybrid Weights",
            "Tower Weights (Candidate Selection)",
            "Transition Weights",
            "Candidate Pool",
            "Scoring",
            "Constraints",
            "Genre Similarity",
            "Repair",
        ]

        for group_name in group_order:
            if group_name not in groups:
                continue

            specs = groups[group_name]
            if not specs:
                continue

            group_box = QGroupBox(group_name)
            self._group_boxes[group_name] = group_box
            group_layout = QVBoxLayout(group_box)
            group_layout.setSpacing(4)

            # Check if this group is a normalization group
            normalize_group_name = None
            for spec in specs:
                if spec.normalize_group:
                    normalize_group_name = spec.normalize_group
                    break

            if normalize_group_name and normalize_group_name in normalize_groups:
                # Use NormalizedSliderGroup for weight groups
                norm_specs = normalize_groups[normalize_group_name]
                norm_widget = NormalizedSliderGroup(norm_specs, self.config_model)
                norm_widget.values_changed.connect(self._on_normalized_group_changed)
                norm_widget.override_changed.connect(self._on_override_changed)
                group_layout.addWidget(norm_widget)
                self._group_widgets[normalize_group_name] = norm_widget

                # Add any non-normalized settings in this group
                for spec in specs:
                    if spec.key_path not in normalized_keys:
                        control = SettingControl(spec, self.config_model)
                        control.value_changed.connect(self._on_setting_changed)
                        control.override_changed.connect(self._on_override_changed)
                        group_layout.addWidget(control)
                        self._controls[spec.key_path] = control
            else:
                # Regular controls
                for spec in specs:
                    if spec.key_path not in normalized_keys:
                        control = SettingControl(spec, self.config_model)
                        control.value_changed.connect(self._on_setting_changed)
                        control.override_changed.connect(self._on_override_changed)
                        group_layout.addWidget(control)
                        self._controls[spec.key_path] = control

            # Group reset button
            group_reset_row = QHBoxLayout()
            group_reset_row.addStretch()

            group_reset_btn = QPushButton("Reset Group")
            group_reset_btn.setStyleSheet(GROUP_RESET_BTN_STYLE)
            group_reset_btn.clicked.connect(lambda checked, g=group_name: self._on_group_reset(g))
            group_reset_btn.setToolTip(f"Reset all {group_name} settings to base config")
            group_reset_row.addWidget(group_reset_btn)
            self._group_reset_btns[group_name] = group_reset_btn

            group_layout.addLayout(group_reset_row)

            layout.addWidget(group_box)

        # Stretch at bottom
        layout.addStretch()

        self.setWidget(content)

        # Update button states
        self._update_reset_buttons()

    def _on_setting_changed(self, key_path: str, value: Any) -> None:
        """Handle individual setting change."""
        self.setting_changed.emit(key_path, value)

    def _on_normalized_group_changed(self, values: Dict[str, Any]) -> None:
        """Handle normalized group change."""
        for key_path, value in values.items():
            self.setting_changed.emit(key_path, value)

    def _on_override_changed(self) -> None:
        """Handle any override state change."""
        self._update_reset_buttons()
        self.override_changed.emit()

    def _update_reset_buttons(self) -> None:
        """Update enabled state of reset buttons based on override state."""
        # Global reset button
        has_any_override = self.config_model.override_count() > 0
        self._global_reset_btn.setEnabled(has_any_override)

        # Group reset buttons
        groups = get_visible_settings_by_group()
        normalize_groups = get_normalize_groups()

        for group_name, btn in self._group_reset_btns.items():
            group_has_override = False

            # Check regular controls
            if group_name in groups:
                for spec in groups[group_name]:
                    if self.config_model.has_override(spec.key_path):
                        group_has_override = True
                        break

            # Check normalized groups
            if not group_has_override:
                for norm_name, norm_widget in self._group_widgets.items():
                    if norm_widget.has_any_override():
                        # Check if this normalized group belongs to this UI group
                        for spec in normalize_groups.get(norm_name, []):
                            if spec.group == group_name:
                                group_has_override = True
                                break

            btn.setEnabled(group_has_override)

    def _on_group_reset(self, group_name: str) -> None:
        """Reset all settings in a group."""
        self.config_model.clear_group_overrides(group_name)

        # Also reset normalized groups that belong to this UI group
        normalize_groups = get_normalize_groups()
        for norm_name, norm_widget in self._group_widgets.items():
            for spec in normalize_groups.get(norm_name, []):
                if spec.group == group_name:
                    norm_widget.reset_all()
                    break

        self.refresh()
        self.override_changed.emit()

    def _on_global_reset(self) -> None:
        """Reset all overrides after confirmation."""
        reply = QMessageBox.question(
            self,
            "Reset All Overrides",
            "Reset all overrides? This will revert all Advanced Settings to the base config values.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.config_model.reset()
            self.refresh()
            self.override_changed.emit()

    def refresh(self) -> None:
        """Refresh all controls from config model."""
        for control in self._controls.values():
            if hasattr(control, "refresh"):
                control.refresh()

        for group in self._group_widgets.values():
            group.refresh()

        self._update_reset_buttons()

    def reset_to_base(self) -> None:
        """Reset all settings to base config values."""
        self.config_model.reset()
        self.refresh()
        self.override_changed.emit()
