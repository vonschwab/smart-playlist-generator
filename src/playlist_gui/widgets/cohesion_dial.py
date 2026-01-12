"""
Cohesion Dial Widget - 4-notch slider for cohesion selection.

Maps user-friendly cohesion levels to backend genre_mode/sonic_mode:
- Tight: strict/strict (very similar tracks)
- Balanced: narrow/narrow (default, moderate similarity)
- Wide: dynamic/dynamic (more variety)
- Discover: discover/discover (maximum exploration)
"""
from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


CohesionLevel = Literal["tight", "balanced", "wide", "discover"]

COHESION_LEVELS: list[CohesionLevel] = ["tight", "balanced", "wide", "discover"]
COHESION_LABELS = {
    "tight": "Tight",
    "balanced": "Balanced",
    "wide": "Wide",
    "discover": "Discover",
}
COHESION_TOOLTIPS = {
    "tight": "Very similar tracks - strict genre and sonic matching",
    "balanced": "Moderate similarity - good balance of cohesion and variety",
    "wide": "More variety - dynamic matching allows stylistic exploration",
    "discover": "Maximum exploration - discover new sounds and genres",
}


class CohesionDial(QWidget):
    """
    4-notch cohesion selector widget.

    Provides a slider with discrete positions for selecting playlist cohesion level.
    Emits cohesion_changed signal when value changes.
    """

    cohesion_changed = Signal(str)  # Emits: "tight", "balanced", "wide", "discover"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_value: CohesionLevel = "balanced"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header with current value
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        header_label = QLabel("<b>Cohesion:</b>")
        header_row.addWidget(header_label)

        self._value_label = QLabel(COHESION_LABELS[self._current_value])
        self._value_label.setStyleSheet("color: #4a86c7; font-weight: bold;")
        header_row.addWidget(self._value_label)

        header_row.addStretch()
        layout.addLayout(header_row)

        # Slider row
        slider_row = QHBoxLayout()
        slider_row.setSpacing(4)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(3)
        self._slider.setValue(1)  # Default: balanced (index 1)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setPageStep(1)
        self._slider.setSingleStep(1)
        self._slider.setFixedWidth(200)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #e0e0e0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a86c7, stop:1 #6aa6e7);
                border: 1px solid #4a86c7;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a86c7;
                border: 1px solid #3a76b7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #5a96d7;
            }
        """)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider)

        layout.addLayout(slider_row)

        # Labels row
        labels_row = QHBoxLayout()
        labels_row.setContentsMargins(0, 0, 0, 0)

        for level in COHESION_LEVELS:
            label = QLabel(COHESION_LABELS[level])
            label.setStyleSheet("font-size: 10px; color: #666;")
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(COHESION_TOOLTIPS[level])
            labels_row.addWidget(label)

        layout.addLayout(labels_row)

        # Set tooltip for the whole widget
        self.setToolTip(
            "Control how similar tracks in the playlist should be.\n"
            "Tight = very similar, Discover = maximum variety."
        )

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        if 0 <= value < len(COHESION_LEVELS):
            new_value = COHESION_LEVELS[value]
            if new_value != self._current_value:
                self._current_value = new_value
                self._value_label.setText(COHESION_LABELS[new_value])
                self._value_label.setToolTip(COHESION_TOOLTIPS[new_value])
                self.cohesion_changed.emit(new_value)

    def value(self) -> CohesionLevel:
        """Return current cohesion level."""
        return self._current_value

    def set_value(self, level: CohesionLevel) -> None:
        """Set cohesion level programmatically."""
        if level in COHESION_LEVELS:
            index = COHESION_LEVELS.index(level)
            self._slider.setValue(index)
            # _on_slider_changed will update internal state

    def reset(self) -> None:
        """Reset to default value (balanced)."""
        self.set_value("balanced")
