"""PySide6 review panel for human-in-the-loop genre tag review."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ai_genre_enrichment.storage import SidecarStore


class ReviewPanel(QWidget):
    """Single-keystroke review panel for genre tag classification."""

    review_completed = Signal()
    vocab_graduated = Signal()

    def __init__(self, sidecar_db_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = SidecarStore(sidecar_db_path)
        self._store.initialize()
        self._queue: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._index = 0
        self._stats = {"genre_style": 0, "descriptor": 0, "instrument": 0, "place": 0, "rejected": 0, "skipped": 0}

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Review Queue</b>"))
        header.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "review_only", "Low confidence"])
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        header.addWidget(self._filter_combo)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.load_queue)
        header.addWidget(self._refresh_btn)
        layout.addLayout(header)

        self._release_label = QLabel()
        self._release_label.setWordWrap(True)
        layout.addWidget(self._release_label)

        self._source_label = QLabel()
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        self._current_label = QLabel()
        layout.addWidget(self._current_label)

        self._tag_label = QLabel()
        self._tag_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        layout.addWidget(self._tag_label)

        self._context_label = QLabel()
        self._context_label.setWordWrap(True)
        layout.addWidget(self._context_label)

        layout.addStretch()

        buttons = QHBoxLayout()
        for key, label, classification in [
            ("A", "Accept genre", "genre_style"),
            ("D", "Descriptor", "descriptor"),
            ("I", "Instrument", "instrument"),
            ("P", "Place", "place"),
            ("S", "Skip", None),
            ("R", "Reject", "rejected"),
        ]:
            btn = QPushButton(f"[{key}] {label}")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda checked=False, c=classification: self._decide(c))
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        action_row = QHBoxLayout()
        self.graduate_button = QPushButton("Graduate to YAML")
        self.graduate_button.setToolTip("Promote AI- and human-reviewed tags into the vocabulary YAML.")
        self.graduate_button.clicked.connect(self._on_graduate_clicked)
        action_row.addWidget(self.graduate_button)

        self.cli_review_button = QPushButton("Open CLI review")
        self.cli_review_button.setToolTip("Launch an interactive terminal for the review CLI.")
        self.cli_review_button.clicked.connect(self._on_cli_review_clicked)
        action_row.addWidget(self.cli_review_button)
        layout.addLayout(action_row)

        self._progress_label = QLabel()
        layout.addWidget(self._progress_label)

    def _setup_shortcuts(self) -> None:
        for key, classification in [
            ("A", "genre_style"),
            ("D", "descriptor"),
            ("I", "instrument"),
            ("P", "place"),
            ("R", "rejected"),
        ]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda c=classification: self._decide(c))

        skip_shortcut = QShortcut(QKeySequence("S"), self)
        skip_shortcut.activated.connect(lambda: self._decide(None))

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._undo)

    def load_queue(self) -> None:
        filter_text = self._filter_combo.currentText()
        classification = "review_only" if filter_text == "review_only" else None
        max_confidence = 0.50 if filter_text == "Low confidence" else 0.80
        self._queue = self._store.get_review_queue(
            classification=classification,
            max_confidence=max_confidence,
        )
        self._index = 0
        self._show_current()

    def _on_filter_changed(self) -> None:
        self.load_queue()

    def _show_current(self) -> None:
        if self._index >= len(self._queue):
            self._release_label.setText("")
            self._source_label.setText("")
            self._current_label.setText("")
            self._tag_label.setText("Review complete — no more tags in queue.")
            self._context_label.setText("")
            self._update_progress()
            self.review_completed.emit()
            return

        item = self._queue[self._index]
        self._release_label.setText(f"Release: {item['normalized_artist']} — {item['normalized_album']}")
        self._source_label.setText(f"Source: {item['source_url']}")
        self._current_label.setText(f"Current: {item['classification']} ({item['confidence']:.2f})")
        self._tag_label.setText(f'"{item["normalized_tag"]}"')

        context = self._store.get_review_context(item["release_key"])
        context_lines = [
            f"  {c['normalized_tag']} ({c['classification']}, {c['confidence']:.2f})"
            for c in context
            if c["normalized_tag"] != item["normalized_tag"]
        ]
        self._context_label.setText("Context:\n" + "\n".join(context_lines[:8]) if context_lines else "")
        self._update_progress()

    def _decide(self, classification: str | None) -> None:
        if self._index >= len(self._queue):
            return
        item = self._queue[self._index]

        if classification is not None:
            self._store.record_review_decision(
                source_tag_id=item["source_tag_id"],
                release_key=item["release_key"],
                raw_tag=item["raw_tag"],
                normalized_tag=item["normalized_tag"],
                original_classification=item["classification"],
                reviewed_classification=classification,
            )
            self._store.rebuild_enriched_genres_for_release(item["release_key"])
            stat_key = classification if classification in self._stats else "accepted"
            self._stats[stat_key] = self._stats.get(stat_key, 0) + 1
            self._history.append({
                "source_tag_id": item["source_tag_id"],
                "classification": classification,
                "release_key": item["release_key"],
            })
        else:
            self._stats["skipped"] += 1

        self._index += 1
        self._show_current()

    def _undo(self) -> None:
        if not self._history:
            return
        last = self._history.pop()
        self._store.undo_review_decision(last["source_tag_id"])
        self._store.rebuild_enriched_genres_for_release(last["release_key"])
        stat_key = last["classification"] if last["classification"] in self._stats else "genre_style"
        self._stats[stat_key] = max(0, self._stats.get(stat_key, 0) - 1)
        self._index = max(0, self._index - 1)
        self._show_current()

    def _update_progress(self) -> None:
        total = len(self._queue)
        reviewed = self._index
        parts = [f"{reviewed} / {total} reviewed"]
        for key, count in self._stats.items():
            if count > 0:
                parts.append(f"{count} {key}")
        self._progress_label.setText(" | ".join(parts))

    def _on_graduate_clicked(self) -> None:
        for command in ("graduate-ai", "graduate-reviewed"):
            argv = [sys.executable, "scripts/ai_genre_enrich.py", command]
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return
        self.vocab_graduated.emit()
        if hasattr(self, "load_queue"):
            self.load_queue()

    def _on_cli_review_clicked(self) -> None:
        argv = [sys.executable, "scripts/ai_genre_enrich.py", "review"]
        if sys.platform == "win32":
            subprocess.Popen(argv, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(argv)
