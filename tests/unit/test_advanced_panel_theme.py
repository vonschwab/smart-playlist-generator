"""Tests for Advanced Settings panel theming hooks."""

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from src.playlist_gui.config.config_model import ConfigModel
from src.playlist_gui.widgets.advanced_panel import AdvancedSettingsPanel


def test_advanced_panel_uses_theme_object_names(qtbot):
    panel = AdvancedSettingsPanel(ConfigModel())
    qtbot.addWidget(panel)

    assert panel.objectName() == "advancedSettingsPanel"
    assert panel._global_reset_btn.objectName() == "advancedDangerButton"
    assert panel._global_reset_btn.styleSheet() == ""

    assert panel._group_reset_btns
    for button in panel._group_reset_btns.values():
        assert button.objectName() == "advancedResetButton"
        assert button.styleSheet() == ""


def test_advanced_modified_state_uses_properties_not_inline_styles(qtbot):
    model = ConfigModel()
    panel = AdvancedSettingsPanel(model)
    qtbot.addWidget(panel)

    key_path, control = next(iter(panel._controls.items()))
    widget = control._control
    model.set(key_path, model.get(key_path))

    control.refresh()

    assert control.property("modified") is True
    assert control.styleSheet() == ""
    assert control._name_label.property("modified") is True
    assert control._modified_dot.objectName() == "modifiedDot"
    assert control._reset_btn.objectName() == "advancedIconButton"
    if isinstance(widget, (QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox)):
        assert widget.styleSheet() == ""
