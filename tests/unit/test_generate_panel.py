"""Tests for the GUI generate panel."""

from pathlib import Path

from src.playlist_gui.widgets.generate_panel import GeneratePanel


def test_generate_panel_exposes_cli_parity_modes(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    modes = [
        panel._mode_combo.itemData(index)
        for index in range(panel._mode_combo.count())
    ]

    assert modes == ["artist", "genre", "seeds", "history"]


def test_generate_panel_builds_genre_state(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    genre_index = panel._mode_combo.findData("genre")
    panel._mode_combo.setCurrentIndex(genre_index)
    panel._genre_panel.set_genre("ambient")

    state = panel.build_ui_state()

    assert state.mode == "genre"
    assert state.genre_query == "ambient"


def test_pace_mode_selector_defaults_to_dynamic(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    assert panel._mode_sliders.get_pace_mode() == "dynamic"
    assert panel._mode_sliders._pace_value_label.text() == "Dynamic"
    assert panel.build_ui_state().pace_mode == "dynamic"


def test_pace_mode_selector_is_a_slider_with_three_modes(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    assert panel._mode_sliders._pace_slider.minimum() == 0
    assert panel._mode_sliders._pace_slider.maximum() == 2
    panel._mode_sliders.set_pace_mode("strict")
    assert panel.build_ui_state().pace_mode == "strict"


def test_generate_panel_restores_saved_artist_state(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel.apply_saved_state(mode="artist", artist="Slowdive")

    assert panel.get_current_mode() == "artist"
    assert panel.get_primary_artist() == "Slowdive"
    assert panel.build_ui_state().artist_queries == ["Slowdive"]


def test_generate_panel_header_can_expand_to_container(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel.resize(1800, panel.sizeHint().height())
    panel.show()

    assert panel._header_frame.maximumWidth() >= 16777215
    assert not panel._header_frame.hasHeightForWidth()


def test_generate_panel_header_uses_responsive_two_row_layout(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._reflow_header_groups(2000)
    assert panel._header_row_count == 1
    assert [panel._header_group_positions[key][0] for key in panel._header_group_order] == [0] * 7

    panel._reflow_header_groups(900)
    assert panel._header_row_count == 2
    assert [key for key, pos in panel._header_group_positions.items() if pos[0] == 0] == [
        "mode",
        "matching",
        "length",
        "freshness",
    ]
    assert [key for key, pos in panel._header_group_positions.items() if pos[0] == 1] == [
        "spacing",
        "diversity",
        "actions",
    ]
    assert [panel._header_group_positions[key][3] for key in panel._header_group_order[:4]] == [3, 3, 3, 3]
    assert [panel._header_group_positions[key][3] for key in panel._header_group_order[4:]] == [4, 4, 4]
    assert sum(panel._header_group_positions[key][3] for key in panel._header_group_order[:4]) == 12
    assert sum(panel._header_group_positions[key][3] for key in panel._header_group_order[4:]) == 12


def test_generate_panel_header_uses_named_control_groups(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    assert set(panel._control_groups) == {
        "mode",
        "matching",
        "length",
        "freshness",
        "spacing",
        "diversity",
        "actions",
    }
    for group in panel._control_groups.values():
        assert group.objectName() == "controlGroup"

    assert panel._mode_group_title.text() == "Mode"
    assert panel._matching_group_title.text() == "Matching"
    assert panel._actions_group_title.text() == "Actions"


def test_generate_panel_header_control_groups_have_equal_height(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    heights = {group.minimumHeight() for group in panel._control_groups.values()}

    assert len(heights) == 1
    assert heights.pop() >= panel._control_groups["matching"].minimumSizeHint().height()


def test_artist_mode_panel_uses_grouped_input_controls(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    artist_panel = panel._artist_panel

    assert set(artist_panel._control_groups) == {
        "artist",
        "presence",
        "variety",
        "collaborations",
    }
    for group in artist_panel._control_groups.values():
        assert group.objectName() == "modeControlGroup"

    assert artist_panel._artist_group_title.text() == "Artist"
    assert artist_panel._presence_group_title.text() == "Presence"
    assert artist_panel._variety_group_title.text() == "Variety"
    assert artist_panel._collaborations_group_title.text() == "Collaborations"


def test_artist_mode_panel_control_groups_have_equal_height(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    heights = {group.minimumHeight() for group in panel._artist_panel._control_groups.values()}

    assert len(heights) == 1
    assert heights.pop() >= panel._artist_panel._control_groups["presence"].minimumSizeHint().height()


def test_genre_mode_panel_uses_compact_grouped_input(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    genre_panel = panel._genre_panel

    assert set(genre_panel._control_groups) == {"genre"}
    assert genre_panel._control_groups["genre"].objectName() == "modeControlGroup"
    assert genre_panel._genre_group_title.text() == "Genre"
    assert genre_panel._control_groups["genre"].maximumHeight() == genre_panel._control_groups["genre"].minimumHeight()


def test_seeds_mode_panel_uses_grouped_input_controls(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    seeds_panel = panel._seeds_panel

    assert set(seeds_panel._control_groups) == {"track", "seed_order"}
    for group in seeds_panel._control_groups.values():
        assert group.objectName() == "modeControlGroup"

    assert seeds_panel._track_group_title.text() == "Track"
    assert seeds_panel._seed_order_group_title.text() == "Seed Order"


def test_history_mode_panel_uses_compact_grouped_input(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    history_panel = panel._history_panel

    assert set(history_panel._control_groups) == {"history"}
    assert history_panel._control_groups["history"].objectName() == "modeControlGroup"
    assert history_panel._history_group_title.text() == "History"
    assert history_panel._control_groups["history"].maximumHeight() == history_panel._control_groups["history"].minimumHeight()


def test_generate_panel_core_controls_are_not_width_pinned(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    controls = [
        panel._mode_combo,
        panel._diversity_slider,
        panel._diversity_value,
        panel._progress_bar,
        panel._stage_label,
        panel._mode_sliders._genre_slider,
        panel._mode_sliders._genre_value_label,
        panel._mode_sliders._sonic_slider,
        panel._mode_sliders._sonic_value_label,
        panel._mode_sliders._pace_slider,
        panel._mode_sliders._pace_value_label,
        panel._artist_panel._presence_combo,
        panel._artist_panel._variety_slider,
        panel._artist_panel._variety_label,
    ]

    for control in controls:
        assert control.maximumWidth() > control.minimumWidth()


def test_diversity_slider_extreme_sets_one_per_artist_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._diversity_slider.setValue(panel._diversity_slider.maximum())
    state = panel.build_ui_state()

    assert state.diversity_gamma == 0.08
    assert state.artist_diversity_mode == "one_per_artist"
    assert panel._diversity_value.text() == "One Each"


def test_diversity_slider_has_room_for_handle(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._diversity_slider.setValue(panel._diversity_slider.maximum())

    assert panel._diversity_slider.minimumWidth() >= 90
    assert panel._diversity_value.minimumWidth() >= 72


def test_slider_styles_inset_groove_for_handle_clearance():
    theme = Path("src/playlist_gui/theme.qss").read_text(encoding="utf-8")

    assert "QSlider:horizontal" in theme
    assert "padding: 0 10px;" in theme
    assert "QSlider::groove:horizontal" in theme
    assert "width: 16px;" in theme
    assert "height: 16px;" in theme
    assert "margin: -7px -9px;" in theme



def test_generate_panel_collapses_from_seeds_to_genre(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)
    panel.show()

    seeds_index = panel._mode_combo.findData("seeds")
    genre_index = panel._mode_combo.findData("genre")

    panel._mode_combo.setCurrentIndex(seeds_index)
    panel._apply_mode_sizing()
    seeds_height = panel._inputs_frame.maximumHeight()

    panel._mode_combo.setCurrentIndex(genre_index)
    panel._apply_mode_sizing()
    genre_height = panel._inputs_frame.maximumHeight()

    assert genre_height < seeds_height
    assert genre_height <= panel._genre_panel.sizeHint().height() + 40


def test_generate_panel_shows_and_clears_validation_message(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel.show_validation_message("Enter an artist before generating.")

    assert panel._validation_label.isVisibleTo(panel) is True
    assert panel._validation_label.text() == "Enter an artist before generating."

    panel.clear_validation_message()

    assert panel._validation_label.isVisibleTo(panel) is False
    assert panel._validation_label.text() == ""


def test_generate_panel_clears_validation_on_mode_change(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel.show_validation_message("Enter an artist before generating.")
    genre_index = panel._mode_combo.findData("genre")
    panel._mode_combo.setCurrentIndex(genre_index)

    assert panel._validation_label.isVisibleTo(panel) is False
