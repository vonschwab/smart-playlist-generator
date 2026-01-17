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
    "tight": "Strict genre + sonic similarity (stay very close to the seed)",
    "balanced": "Moderate genre + sonic similarity (balanced cohesion/variety)",
    "wide": "Looser genre + sonic similarity (explore nearby styles)",
    "discover": "Very loose genre + sonic similarity (explore farthest)",
}


class CohesionDial(QWidget):
    """
    4-notch cohesion selector widget.

    Provides a slider with discrete positions for selecting playlist cohesion level.
    Emits cohesion_changed signal when value changes.

    Args:
        compact: If True, use a single-row compact layout suitable for toolbars.
    """

    cohesion_changed = Signal(str)  # Emits: "tight", "balanced", "wide", "discover"

    def __init__(self, compact: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_value: CohesionLevel = "balanced"
        self._compact = compact
        self._setup_ui()

    def _setup_ui(self) -> None:
        if self._compact:
            self._setup_compact_ui()
        else:
            self._setup_full_ui()

    def _setup_compact_ui(self) -> None:
        """Compact single-row layout for toolbars."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Label
        header_label = QLabel("Cohesion:")
        header_label.setObjectName("controlLabel")
        layout.addWidget(header_label)

        # Slider (narrower)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(3)
        self._slider.setValue(1)  # Default: balanced (index 1)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setPageStep(1)
        self._slider.setSingleStep(1)
        self._slider.setFixedWidth(100)
        self._slider.setObjectName("cohesionSlider")
        self._slider.setToolTip(
            "Controls how strictly BOTH genre and sonic similarity are enforced."
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        # Current value label
        self._value_label = QLabel(COHESION_LABELS[self._current_value])
        self._value_label.setObjectName("cohesionValue")
        self._value_label.setFixedWidth(55)
        layout.addWidget(self._value_label)

        # Tooltip
        self.setToolTip(
            "Controls how strictly BOTH genre and sonic similarity are enforced.\n"
            "Tight stays close to the seed; Discover explores much farther."
        )

    def _setup_full_ui(self) -> None:
        """Full vertical layout with tick labels."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header with current value
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        header_label = QLabel("<b>Cohesion:</b>")
        header_row.addWidget(header_label)

        self._value_label = QLabel(COHESION_LABELS[self._current_value])
        self._value_label.setObjectName("cohesionValue")
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
        self._slider.setObjectName("cohesionSlider")
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider)

        layout.addLayout(slider_row)

        # Labels row
        labels_row = QHBoxLayout()
        labels_row.setContentsMargins(0, 0, 0, 0)

        for level in COHESION_LEVELS:
            label = QLabel(COHESION_LABELS[level])
            label.setObjectName("cohesionTickLabel")
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(COHESION_TOOLTIPS[level])
            labels_row.addWidget(label)

        layout.addLayout(labels_row)

        # Set tooltip for the whole widget
        self.setToolTip(
            "Controls how strictly BOTH genre and sonic similarity are enforced.\n"
            "Tight stays close to the seed; Discover explores much farther."
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
