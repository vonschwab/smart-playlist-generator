from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from src.playlist_gui.widgets.seed_tracks_input import SeedTracksInput


def _ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_seed_tracks_input_cleaning():
    _ensure_app()
    widget = SeedTracksInput()
    widget.set_seed_tracks(
        ["pink diamond - Charli XCX", "  ", "party 4 u", "forever - Charli XCX"]
    )
    assert widget.seed_tracks() == ["pink diamond", "party 4 u", "forever"]
    assert widget.seed_tracks_raw() == [
        "pink diamond - Charli XCX",
        "party 4 u",
        "forever - Charli XCX",
    ]


def test_seed_tracks_input_empty():
    _ensure_app()
    widget = SeedTracksInput()
    widget.set_seed_tracks([])
    assert widget.seed_tracks() == []


def test_seed_tracks_input_uses_responsive_row_controls():
    _ensure_app()
    widget = SeedTracksInput()

    labels = widget.findChildren(QLabel)
    buttons = widget.findChildren(QPushButton)

    assert widget.minimumWidth() == 0
    assert all(label.maximumWidth() > label.minimumWidth() for label in labels)
    assert all(button.maximumWidth() > button.minimumWidth() for button in buttons)
