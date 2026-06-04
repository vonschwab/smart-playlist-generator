"""Dialog for editing the enriched genre signature of a release."""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EditGenresDialog(QDialog):
    """Modal dialog for editing enriched genres for a single release.

    Shows one genre per line. On save, deduplicates by casefold (first
    occurrence wins) and emits genres_committed. The caller computes the
    add/remove diff and persists via the worker.
    """

    genres_committed = Signal(str, str, list)  # artist, album, genres

    def __init__(
        self,
        *,
        artist: str,
        album: str,
        current_genres: Iterable[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._artist = artist
        self._album = album
        self.setWindowTitle(f"Edit genres — {artist} / {album}")
        self.setMinimumSize(420, 360)

        layout = QVBoxLayout(self)
        label = QLabel(
            f"<b>{artist} / {album}</b><br>"
            "One genre per line. Lines you remove will be marked as user-removed; "
            "lines you add will be marked as user-added. Edits are preserved across "
            "future hybrid enrichment runs.<br>"
            "<i>After saving, run <b>Tools → Build Artifacts</b> to apply changes "
            "to playlist generation.</i>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        self._text = QPlainTextEdit(self)
        self._text.setPlainText("\n".join(list(current_genres)))
        layout.addWidget(self._text, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_clicked)
        button_box.addButton(save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def set_text(self, text: str) -> None:
        self._text.setPlainText(text)

    def _on_save_clicked(self) -> None:
        raw_lines = [line.strip() for line in self._text.toPlainText().splitlines()]
        seen: set[str] = set()
        cleaned: list[str] = []
        for line in raw_lines:
            if not line:
                continue
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(line)
        if not cleaned:
            QMessageBox.warning(
                self,
                "No genres",
                "At least one genre is required. "
                "To remove enriched genres entirely, delete the override via the pipeline.",
            )
            return
        self.genres_committed.emit(self._artist, self._album, cleaned)
        self.accept()
