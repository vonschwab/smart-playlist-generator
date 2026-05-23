"""
Mode Sliders - Dual slider control for genre/sonic mode selection.
"""
from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ModeLevel = Literal["strict", "narrow", "dynamic", "discover", "off"]
PaceModeLevel = Literal["strict", "narrow", "dynamic", "off"]

MODE_LEVELS: list[ModeLevel] = ["off", "strict", "narrow", "dynamic", "discover"]
PACE_MODE_LEVELS: list[PaceModeLevel] = ["strict", "narrow", "dynamic", "off"]
MODE_LABELS = {
    "off": "Off",
    "strict": "Strict",
    "narrow": "Narrow",
    "dynamic": "Dynamic",
    "discover": "Discover",
}
PACE_MODE_LABELS = {
    "strict": "Strict",
    "narrow": "Narrow",
    "dynamic": "Dynamic",
    "off": "Off",
}
MODE_TOOLTIPS = {
    "off": "Disable this matching domain",
    "strict": "Ultra-tight matching; stay very close to the seed",
    "narrow": "Cohesive matching; close to the seed",
    "dynamic": "Balanced exploration; moderate variety",
    "discover": "Exploratory matching; widest variety",
}
PACE_MODE_TOOLTIPS = {
    "strict": "Tight rhythm/tempo fidelity",
    "narrow": "Moderate rhythm/tempo anchoring",
    "dynamic": "Gentle anchoring — catches double-time, allows drift",
    "off": "No pace gate — rhythm still influences via sonic embedding",
}


class ModeSliders(QWidget):
    """
    Compact sliders for selecting genre, sonic, and pace modes.

    Emits:
        genre_mode_changed: New genre mode value
        sonic_mode_changed: New sonic mode value
        pace_mode_changed: New pace mode value
    """

    genre_mode_changed = Signal(str)
    sonic_mode_changed = Signal(str)
    pace_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._genre_value: ModeLevel = "narrow"
        self._sonic_value: ModeLevel = "narrow"
        self._pace_value: PaceModeLevel = "dynamic"
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Genre slider row
        genre_row = QHBoxLayout()
        genre_row.setContentsMargins(0, 0, 0, 0)
        genre_row.setSpacing(6)
        genre_label = QLabel("Genre:")
        genre_label.setObjectName("controlLabel")
        genre_label.setMinimumWidth(46)
        genre_row.addWidget(genre_label)

        self._genre_slider = QSlider(Qt.Horizontal)
        self._genre_slider.setMinimum(0)
        self._genre_slider.setMaximum(len(MODE_LEVELS) - 1)
        self._genre_slider.setValue(MODE_LEVELS.index("narrow"))
        self._genre_slider.setTickPosition(QSlider.TicksBelow)
        self._genre_slider.setTickInterval(1)
        self._genre_slider.setPageStep(1)
        self._genre_slider.setSingleStep(1)
        self._genre_slider.setMinimumWidth(90)
        self._genre_slider.setMaximumWidth(130)
        self._genre_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._genre_slider.setObjectName("modeSlider")
        self._genre_slider.setToolTip(
            "Genre strictness (tighter = fewer genre jumps)"
        )
        self._genre_slider.valueChanged.connect(self._on_genre_changed)
        genre_row.addWidget(self._genre_slider)

        self._genre_value_label = QLabel(MODE_LABELS[self._genre_value])
        self._genre_value_label.setObjectName("modeValue")
        self._genre_value_label.setMinimumWidth(68)
        self._genre_value_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._genre_value_label.setToolTip(MODE_TOOLTIPS[self._genre_value])
        genre_row.addWidget(self._genre_value_label)
        layout.addLayout(genre_row)

        # Sonic slider row
        sonic_row = QHBoxLayout()
        sonic_row.setContentsMargins(0, 0, 0, 0)
        sonic_row.setSpacing(6)
        sonic_label = QLabel("Sonic:")
        sonic_label.setObjectName("controlLabel")
        sonic_label.setMinimumWidth(46)
        sonic_row.addWidget(sonic_label)

        self._sonic_slider = QSlider(Qt.Horizontal)
        self._sonic_slider.setMinimum(0)
        self._sonic_slider.setMaximum(len(MODE_LEVELS) - 1)
        self._sonic_slider.setValue(MODE_LEVELS.index("narrow"))
        self._sonic_slider.setTickPosition(QSlider.TicksBelow)
        self._sonic_slider.setTickInterval(1)
        self._sonic_slider.setPageStep(1)
        self._sonic_slider.setSingleStep(1)
        self._sonic_slider.setMinimumWidth(90)
        self._sonic_slider.setMaximumWidth(130)
        self._sonic_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._sonic_slider.setObjectName("modeSlider")
        self._sonic_slider.setToolTip(
            "Sonic strictness (tighter = closer sound profile)"
        )
        self._sonic_slider.valueChanged.connect(self._on_sonic_changed)
        sonic_row.addWidget(self._sonic_slider)

        self._sonic_value_label = QLabel(MODE_LABELS[self._sonic_value])
        self._sonic_value_label.setObjectName("modeValue")
        self._sonic_value_label.setMinimumWidth(68)
        self._sonic_value_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._sonic_value_label.setToolTip(MODE_TOOLTIPS[self._sonic_value])
        sonic_row.addWidget(self._sonic_value_label)
        layout.addLayout(sonic_row)

        # Pace slider row
        pace_row = QHBoxLayout()
        pace_row.setContentsMargins(0, 0, 0, 0)
        pace_row.setSpacing(6)
        pace_label = QLabel("Pace:")
        pace_label.setObjectName("controlLabel")
        pace_label.setMinimumWidth(46)
        pace_row.addWidget(pace_label)

        self._pace_slider = QSlider(Qt.Horizontal)
        self._pace_slider.setMinimum(0)
        self._pace_slider.setMaximum(len(PACE_MODE_LEVELS) - 1)
        self._pace_slider.setValue(PACE_MODE_LEVELS.index("dynamic"))
        self._pace_slider.setTickPosition(QSlider.TicksBelow)
        self._pace_slider.setTickInterval(1)
        self._pace_slider.setPageStep(1)
        self._pace_slider.setSingleStep(1)
        self._pace_slider.setMinimumWidth(90)
        self._pace_slider.setMaximumWidth(130)
        self._pace_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._pace_slider.setObjectName("modeSlider")
        self._pace_slider.setToolTip(
            "Pace strictness (tighter = closer rhythm/tempo)"
        )
        self._pace_slider.valueChanged.connect(self._on_pace_changed)
        pace_row.addWidget(self._pace_slider)

        self._pace_value_label = QLabel(PACE_MODE_LABELS[self._pace_value])
        self._pace_value_label.setObjectName("modeValue")
        self._pace_value_label.setMinimumWidth(68)
        self._pace_value_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._pace_value_label.setToolTip(PACE_MODE_TOOLTIPS[self._pace_value])
        pace_row.addWidget(self._pace_value_label)
        layout.addLayout(pace_row)

        self.setToolTip(
            "Set genre, sonic, and pace strictness separately.\n"
            "Stricter modes narrow matching; Discover explores more."
        )

    def _on_genre_changed(self, value: int) -> None:
        if 0 <= value < len(MODE_LEVELS):
            new_value = MODE_LEVELS[value]
            if new_value != self._genre_value:
                self._genre_value = new_value
                self._genre_value_label.setText(MODE_LABELS[new_value])
                self._genre_value_label.setToolTip(MODE_TOOLTIPS[new_value])
                self.genre_mode_changed.emit(new_value)

    def _on_sonic_changed(self, value: int) -> None:
        if 0 <= value < len(MODE_LEVELS):
            new_value = MODE_LEVELS[value]
            if new_value != self._sonic_value:
                self._sonic_value = new_value
                self._sonic_value_label.setText(MODE_LABELS[new_value])
                self._sonic_value_label.setToolTip(MODE_TOOLTIPS[new_value])
                self.sonic_mode_changed.emit(new_value)

    def _on_pace_changed(self, value: int) -> None:
        if 0 <= value < len(PACE_MODE_LEVELS):
            new_value = PACE_MODE_LEVELS[value]
            if new_value != self._pace_value:
                self._pace_value = new_value
                self._pace_value_label.setText(PACE_MODE_LABELS[new_value])
                self._pace_value_label.setToolTip(PACE_MODE_TOOLTIPS[new_value])
                self.pace_mode_changed.emit(new_value)

    def get_genre_mode(self) -> ModeLevel:
        return self._genre_value

    def get_sonic_mode(self) -> ModeLevel:
        return self._sonic_value

    def get_pace_mode(self) -> PaceModeLevel:
        return self._pace_value

    def set_genre_mode(self, mode: ModeLevel) -> None:
        if mode in MODE_LEVELS:
            self._genre_slider.setValue(MODE_LEVELS.index(mode))

    def set_sonic_mode(self, mode: ModeLevel) -> None:
        if mode in MODE_LEVELS:
            self._sonic_slider.setValue(MODE_LEVELS.index(mode))

    def set_pace_mode(self, mode: PaceModeLevel) -> None:
        if mode in PACE_MODE_LEVELS:
            self._pace_slider.setValue(PACE_MODE_LEVELS.index(mode))

    def reset(self) -> None:
        self.set_genre_mode("narrow")
        self.set_sonic_mode("narrow")
        self.set_pace_mode("dynamic")
