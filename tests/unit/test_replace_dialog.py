from src.playlist_gui.widgets.replace_dialog import ReplaceTrackDialog


def test_dialog_has_five_tabs(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)

    assert dialog.tab_widget.count() == 5
    expected = ["Search", "Best Match", "Different Pace", "Different Genre", "Different Sound"]
    actual = [dialog.tab_widget.tabText(i) for i in range(5)]
    assert actual == expected


def test_dialog_emits_replacement_chosen_on_apply(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)
    chosen = []
    dialog.replacement_chosen.connect(lambda pos, tid: chosen.append((pos, tid)))

    dialog._pick_track("new-track-id-xyz")

    assert chosen == [(3, "new-track-id-xyz")]


def test_dialog_requests_suggestions_lazily(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)
    requests = []
    dialog.suggestions_requested.connect(lambda pos, mode: requests.append((pos, mode)))

    dialog.tab_widget.setCurrentIndex(1)
    dialog.tab_widget.setCurrentIndex(2)
    dialog.tab_widget.setCurrentIndex(1)

    assert requests == [(3, "best"), (3, "different_pace")]


def test_populate_suggestions_adds_candidate_rows(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)

    dialog.populate_suggestions(
        "best",
        [
            {
                "track_id": "t1",
                "title": "New",
                "artist": "Artist",
                "t_prev": 0.81,
                "t_next": 0.77,
                "perceptual_bpm": 121.2,
                "genres": ["post-punk", "new wave"],
            }
        ],
    )

    model = dialog._models["best"]
    assert model.rowCount() == 1
    assert model.item(0, 0).text() == "New"
    assert model.item(0, 5).text() == "post-punk, new wave"


def test_search_uses_completer_track_ids(qtbot, qapp):
    class FakeCompleterData:
        def get_track_id_by_display(self, display):
            return "track-from-completer" if display == "Song - Artist" else None

        def filter_tracks(self, query, artist_filter=None, limit=None):
            return []

    dialog = ReplaceTrackDialog(
        position=2,
        current_track={"title": "Foo", "artist": "Bar"},
        completer_data=FakeCompleterData(),
    )
    qtbot.addWidget(dialog)
    chosen = []
    dialog.replacement_chosen.connect(lambda pos, tid: chosen.append((pos, tid)))

    dialog.search_edit.setText("Song - Artist")
    dialog._select_search_match()

    assert chosen == [(2, "track-from-completer")]
