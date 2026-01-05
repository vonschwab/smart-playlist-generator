from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.playlist_gui.autocomplete import setup_track_completer, update_track_completer
from src.playlist_gui.autocomplete import DatabaseCompleter


class SeedTracksInput(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(360)
        self._rows: List[QLineEdit] = []
        self._row_widgets: List[QWidget] = []
        self._completer_data: Optional[DatabaseCompleter] = None
        self._artist_filter: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        layout.addLayout(self._rows_layout)

        self._add_row(show_label=True, show_add=True)

    def set_completer_data(self, completer_data: DatabaseCompleter) -> None:
        self._completer_data = completer_data
        for row in self._rows:
            if row.completer() is None:
                setup_track_completer(row, completer_data, self._artist_filter)
            else:
                update_track_completer(row, completer_data, self._artist_filter or "")

    def set_artist_filter(self, artist: Optional[str]) -> None:
        self._artist_filter = artist
        if not self._completer_data:
            return
        for row in self._rows:
            update_track_completer(row, self._completer_data, artist or "")

    def seed_tracks_raw(self) -> List[str]:
        seeds: List[str] = []
        for row in self._rows:
            value = row.text().strip()
            if not value:
                continue
            seeds.append(value)
        return seeds

    def seed_tracks(self) -> List[str]:
        seeds: List[str] = []
        for value in self.seed_tracks_raw():
            if " - " in value:
                value = value.split(" - ")[0].strip()
            if value:
                seeds.append(value)
        return seeds

    def set_seed_tracks(self, tracks: List[str]) -> None:
        self._clear_rows()
        if not tracks:
            self._add_row()
            return
        for track in tracks:
            self._add_row(str(track))

    def _clear_rows(self) -> None:
        for widget in self._row_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._row_widgets = []
        self._rows = []

    def _on_add_row(self) -> None:
        self._add_row()

    def _add_row(self, text: str = "", *, show_label: bool = False, show_add: bool = False) -> None:
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        if show_label:
            label = QLabel("Track Seeds (optional):")
            label.setFixedWidth(150)
            row_layout.addWidget(label)
        else:
            spacer = QLabel("")
            spacer.setFixedWidth(150)
            row_layout.addWidget(spacer)

        edit = QLineEdit()
        edit.setPlaceholderText('Seed track title (e.g. "pink diamond")')
        edit.setText(text)
        edit.setMinimumWidth(260)
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout.addWidget(edit, stretch=1)

        if show_add:
            add_btn = QPushButton("Add")
            add_btn.setFixedWidth(50)
            add_btn.clicked.connect(self._on_add_row)
            row_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(70)
        remove_btn.clicked.connect(lambda: self._remove_row(row_widget))
        row_layout.addWidget(remove_btn)

        self._rows_layout.addWidget(row_widget)
        self._rows.append(edit)
        self._row_widgets.append(row_widget)

        if self._completer_data:
            setup_track_completer(edit, self._completer_data, self._artist_filter)

    def _remove_row(self, row_widget: QWidget) -> None:
        if row_widget not in self._row_widgets:
            return
        idx = self._row_widgets.index(row_widget)
        edit = self._rows[idx]

        self._row_widgets.pop(idx)
        self._rows.pop(idx)

        row_widget.setParent(None)
        row_widget.deleteLater()
        edit.deleteLater()

        if not self._rows:
            self._add_row(show_label=True, show_add=True)
