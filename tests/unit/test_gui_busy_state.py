"""Tests for GUI busy-state helpers."""

from PySide6.QtWidgets import QMenu

from src.playlist_gui.main_window import _set_menu_actions_enabled
from src.playlist_gui.main_window import MainWindow


def test_set_menu_actions_enabled_updates_all_actions(qtbot):
    menu = QMenu()
    qtbot.addWidget(menu)
    first = menu.addAction("First")
    second = menu.addAction("Second")

    _set_menu_actions_enabled(menu, False)

    assert first.isEnabled() is False
    assert second.isEnabled() is False

    _set_menu_actions_enabled(menu, True)

    assert first.isEnabled() is True
    assert second.isEnabled() is True


def test_update_tools_busy_state_labels_disabled_actions(qtbot):
    window = MainWindow.__new__(MainWindow)
    window._tools_menu = QMenu("&Tools")
    qtbot.addWidget(window._tools_menu)
    analyze = window._tools_menu.addAction("&Analyze Library")
    scan = window._tools_menu.addAction("Scan &Library")
    window._tool_actions = {
        "analyze_library": analyze,
        "scan_library": scan,
    }
    window._tool_action_tips = {
        analyze: "Run the default library analysis pipeline.",
        scan: "Scan the configured music library.",
    }

    MainWindow._update_tools_busy_state(window, True)

    assert window._tools_menu.title() == "&Tools (Busy)"
    assert analyze.isEnabled() is False
    assert scan.isEnabled() is False
    assert analyze.statusTip() == "Unavailable while another operation is running."
    assert scan.toolTip() == "Unavailable while another operation is running."

    MainWindow._update_tools_busy_state(window, False)

    assert window._tools_menu.title() == "&Tools"
    assert analyze.isEnabled() is True
    assert scan.isEnabled() is True
    assert analyze.statusTip() == "Run the default library analysis pipeline."
    assert scan.toolTip() == "Scan the configured music library."
