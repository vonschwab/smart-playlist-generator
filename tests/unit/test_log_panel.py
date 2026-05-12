from src.playlist_gui.widgets.log_panel import LogPanel


def test_log_panel_appends_incrementally(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)
    calls = []
    original_append = panel._append_to_view

    def _counting_append(level, message):
        calls.append((level, message))
        original_append(level, message)

    panel._append_to_view = _counting_append

    for idx in range(5):
        panel.append_log("INFO", f"message-{idx}")

    assert len(calls) == 5
    assert panel._text_edit.toPlainText().count("INFO:") == 5


def test_log_panel_stores_filtered_debug_without_rendering_until_enabled(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)

    panel.append_log("DEBUG", "hidden detail")

    assert "hidden detail" not in panel._text_edit.toPlainText()

    panel._level_checks["DEBUG"].setChecked(True)

    assert "DEBUG: hidden detail" in panel._text_edit.toPlainText()


def test_log_panel_visible_document_is_bounded(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel._max_entries = 3
    panel._text_edit.document().setMaximumBlockCount(panel._max_entries + 1)

    for idx in range(5):
        panel.append_log("INFO", f"message-{idx}")

    text = panel._text_edit.toPlainText()
    assert "message-0" not in text
    assert "message-1" not in text
    assert "message-2" in text
    assert "message-4" in text
    assert len(panel._entries) == 3


def test_log_panel_uses_operator_console_surface(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)

    assert panel._toolbar_frame.objectName() == "logToolbar"
    assert panel._title_label.text() == "Logs"
    assert panel._status_label.objectName() == "logStatusLabel"
    assert panel._status_label.text() == "0 entries"
    assert panel._text_edit.objectName() == "logText"
    assert panel._copy_btn.objectName() == "logActionButton"
    assert panel._open_btn.objectName() == "logActionButton"
    assert panel._clear_btn.objectName() == "logActionButton"


def test_log_panel_status_counts_total_and_visible_entries(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)

    panel.append_log("INFO", "visible info")
    panel.append_log("DEBUG", "hidden debug")
    panel.append_log("ERROR", "visible error")

    assert panel._status_label.text() == "3 entries (2 visible)"

    panel._level_checks["DEBUG"].setChecked(True)
    assert panel._status_label.text() == "3 entries"

    panel._search_edit.setText("not present")
    assert panel._status_label.text() == "3 entries (0 visible)"
