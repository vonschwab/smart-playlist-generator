"""
Cohesion Slider — single-axis control for playlists.cohesion_mode.

Mirrors the per-row layout of ModeSliders but as a standalone single-slider
widget for use in its own header card ("OVERALL COHESION"). Drives the
pier-bridge beam tuning axis independently of Genre/Sonic/Pace.
"""
from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

CohesionModeLevel = Literal["strict", "narrow", "dynamic", "discover"]

COHESION_MODE_LEVELS: list[CohesionModeLevel] = [
    "strict",
    "narrow",
    "dynamic",
    "discover",
]
COHESION_MODE_LABELS = {
    "strict": "Strict",
    "narrow": "Narrow",
    "dynamic": "Dynamic",
    "discover": "Discover",
}
COHESION_MODE_TOOLTIPS = {
    "strict": "Ultra-cohesive transitions; tightest beam, narrowest bridges",
    "narrow": "Cohesive transitions; tight beam",
    "dynamic": "Balanced beam (default)",
    "discover": "Loosest transitions; widest exploration in bridging",
}


class CohesionSlider(QWidget):
    """
    Single-slider widget for selecting overall cohesion mode.

    Emits:
        cohesion_mode_changed: New cohesion mode value
    """

    cohesion_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: CohesionModeLevel = "dynamic"
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._setup_ui()

    def _setup_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(COHESION_MODE_LEVELS) - 1)
        self._slider.setValue(COHESION_MODE_LEVELS.index(self._value))
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setPageStep(1)
        self._slider.setSingleStep(1)
        self._slider.setMinimumWidth(90)
        self._slider.setMaximumWidth(130)
        self._slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._slider.setObjectName("modeSlider")
        self._slider.setToolTip(
            "Overall cohesion (pier-bridge beam tightness)"
        )
        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider)

        self._value_label = QLabel(COHESION_MODE_LABELS[self._value])
        self._value_label.setObjectName("modeValue")
        self._value_label.setMinimumWidth(68)
        self._value_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self._value_label.setToolTip(COHESION_MODE_TOOLTIPS[self._value])
        row.addWidget(self._value_label)

    def _on_changed(self, slider_value: int) -> None:
        if 0 <= slider_value < len(COHESION_MODE_LEVELS):
            new_value = COHESION_MODE_LEVELS[slider_value]
            if new_value != self._value:
                self._value = new_value
                self._value_label.setText(COHESION_MODE_LABELS[new_value])
                self._value_label.setToolTip(COHESION_MODE_TOOLTIPS[new_value])
                self.cohesion_mode_changed.emit(new_value)

    def get_cohesion_mode(self) -> CohesionModeLevel:
        return self._value

    def set_cohesion_mode(self, mode: CohesionModeLevel) -> None:
        if mode in COHESION_MODE_LEVELS:
            self._slider.setValue(COHESION_MODE_LEVELS.index(mode))

    def reset(self) -> None:
        self.set_cohesion_mode("dynamic")
