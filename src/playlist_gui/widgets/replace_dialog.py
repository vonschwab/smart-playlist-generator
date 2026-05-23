"""Dialog for replacing one generated playlist track."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


SUGGESTION_TABS: tuple[tuple[str, str], ...] = (
    ("Best Match", "best"),
    ("Different Pace", "different_pace"),
    ("Different Genre", "different_genre"),
    ("Different Sound", "different_sound"),
)


class ReplaceTrackDialog(QDialog):
    replacement_chosen = Signal(int, str)
    suggestions_requested = Signal(int, str)

    def __init__(
        self,
        *,
        position: int,
        current_track: Dict[str, Any],
        library_tracks: Optional[Iterable[Dict[str, Any]]] = None,
        completer_data: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.position = int(position)
        self.current_track = dict(current_track or {})
        self._requested_modes: set[str] = set()
        self._selected_track_id: Optional[str] = None
        self._models: dict[str, QStandardItemModel] = {}
        self._tables: dict[str, QTableView] = {}
        self._status_labels: dict[str, QLabel] = {}
        self._library_tracks = list(library_tracks or [])
        self._completer_data = completer_data

        self.setWindowTitle("Replace Track")
        self.setMinimumSize(760, 460)

        layout = QVBoxLayout(self)
        heading = QLabel(
            f"{self.current_track.get('artist', '')} - {self.current_track.get('title', '')}".strip(" -")
            or "Current track"
        )
        heading.setObjectName("replaceTrackHeading")
        layout.addWidget(heading)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._build_search_tab(), "Search")
        for label, mode in SUGGESTION_TABS:
            self.tab_widget.addTab(self._build_suggestion_tab(mode), label)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tab_widget, stretch=1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_selected)
        self.button_box.addButton(self.apply_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _build_search_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search library tracks")
        if self._completer_data is not None:
            try:
                from src.playlist_gui.autocomplete import setup_track_completer

                setup_track_completer(self.search_edit, self._completer_data)
            except Exception:
                pass
        self.search_button = QPushButton("Select")
        self.search_button.clicked.connect(self._select_search_match)
        row.addWidget(self.search_edit, stretch=1)
        row.addWidget(self.search_button)
        layout.addLayout(row)
        self.search_status = QLabel("")
        layout.addWidget(self.search_status)
        layout.addStretch(1)
        return widget

    def _build_suggestion_tab(self, mode: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        status = QLabel("Select this tab to load suggestions.")
        table = QTableView()
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        table.setSortingEnabled(True)
        table.doubleClicked.connect(lambda index, m=mode: self._pick_row(m, index.row()))
        table.clicked.connect(lambda index, m=mode: self._select_row(m, index.row()))

        model = QStandardItemModel(0, 6, self)
        model.setHorizontalHeaderLabels(["Title", "Artist", "T_prev", "T_next", "BPM", "Top genres"])
        table.setModel(model)
        table.horizontalHeader().setStretchLastSection(True)

        self._models[mode] = model
        self._tables[mode] = table
        self._status_labels[mode] = status
        layout.addWidget(status)
        layout.addWidget(table, stretch=1)
        return widget

    def _on_tab_changed(self, index: int) -> None:
        if index <= 0:
            return
        mode = SUGGESTION_TABS[index - 1][1]
        if mode in self._requested_modes:
            return
        self._requested_modes.add(mode)
        self._status_labels[mode].setText("Loading...")
        self.suggestions_requested.emit(self.position, mode)

    def populate_suggestions(self, mode: str, candidates: List[Dict[str, Any]]) -> None:
        if mode not in self._models:
            return
        model = self._models[mode]
        model.removeRows(0, model.rowCount())
        for candidate in candidates:
            track_id = str(candidate.get("track_id") or candidate.get("rating_key") or "")
            row = [
                self._item(str(candidate.get("title") or track_id), track_id),
                self._item(str(candidate.get("artist") or "")),
                self._item(f"{float(candidate.get('t_prev', 0.0)):.3f}"),
                self._item(f"{float(candidate.get('t_next', 0.0)):.3f}"),
                self._item(self._format_bpm(candidate.get("perceptual_bpm"))),
                self._item(", ".join(str(g) for g in candidate.get("genres", [])[:3])),
            ]
            model.appendRow(row)
        self._status_labels[mode].setText(
            f"{len(candidates)} suggestion(s)" if candidates else "No suggestions found"
        )

    def _item(self, text: str, track_id: Optional[str] = None) -> QStandardItem:
        item = QStandardItem(text)
        item.setEditable(False)
        if track_id:
            item.setData(track_id, Qt.ItemDataRole.UserRole)
        return item

    def _format_bpm(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except Exception:
            return ""

    def _select_search_match(self) -> None:
        query_text = self.search_edit.text().strip()
        if self._completer_data is not None:
            try:
                track_id = self._completer_data.get_track_id_by_display(query_text)
                if track_id:
                    self._pick_track(str(track_id))
                    return
                matches = self._completer_data.filter_tracks(query_text, limit=1)
                if matches:
                    track_id = self._completer_data.get_track_id_by_display(matches[0])
                    if track_id:
                        self._pick_track(str(track_id))
                        return
            except Exception:
                pass

        query = query_text.casefold()
        if not query:
            return
        for track in self._library_tracks:
            label = f"{track.get('artist', '')} {track.get('title', '')}".casefold()
            if query in label:
                track_id = str(track.get("track_id") or track.get("rating_key") or "")
                if track_id:
                    self._pick_track(track_id)
                return
        self.search_status.setText("No matching track found")

    def _select_row(self, mode: str, row: int) -> None:
        model = self._models.get(mode)
        if model is None or row < 0 or row >= model.rowCount():
            return
        track_id = model.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if track_id:
            self._selected_track_id = str(track_id)
            self.apply_button.setEnabled(True)

    def _pick_row(self, mode: str, row: int) -> None:
        self._select_row(mode, row)
        self._apply_selected()

    def _pick_track(self, track_id: str) -> None:
        self._selected_track_id = str(track_id)
        self.apply_button.setEnabled(True)
        self._apply_selected()

    def _apply_selected(self) -> None:
        if not self._selected_track_id:
            return
        self.replacement_chosen.emit(self.position, self._selected_track_id)
        self.accept()
