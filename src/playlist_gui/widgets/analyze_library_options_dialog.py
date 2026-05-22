from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QFrame,
    QVBoxLayout,
    QWidget,
)

from src.playlist.analyze_library_results import (
    format_analyze_library_action_label,
    format_analyze_library_action_list,
)
from src.playlist.request_models import (
    ANALYZE_LIBRARY_STAGE_ORDER,
    LibraryPipelineRequest,
)


PRESET_STAGES: dict[str, list[str]] = {
    "Full Analyze": list(ANALYZE_LIBRARY_STAGE_ORDER),
    "Quick Verify": ["verify"],
    "Rebuild Sonic + Artifacts": ["sonic", "genre-sim", "artifacts", "verify"],
    "Custom": list(ANALYZE_LIBRARY_STAGE_ORDER),
}

class AnalyzeLibraryOptionsDialog(QDialog):
    """Native options dialog for the unified Analyze Library pipeline."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("analyzeLibraryOptionsDialog")
        self.setWindowTitle("Analyze Library")
        self.setModal(True)
        self._stage_checks: dict[str, QCheckBox] = {}
        self._setup_ui()
        self._apply_preset("Full Analyze")

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._title_label = QLabel("Analyze Library")
        self._title_label.setObjectName("dialogTitle")
        layout.addWidget(self._title_label)

        self._summary_label = QLabel(
            "Use the default analysis for normal maintenance. Customize stages only when troubleshooting or rebuilding specific outputs."
        )
        self._summary_label.setObjectName("dialogSummary")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._preset_frame = QFrame()
        self._preset_frame.setObjectName("dialogControlFrame")
        preset_row = QHBoxLayout(self._preset_frame)
        preset_row.setContentsMargins(8, 6, 8, 6)
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESET_STAGES.keys())
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self._preset_combo, 1)
        layout.addWidget(self._preset_frame)

        self._preview_label = QLabel("")
        self._preview_label.setObjectName("dialogPreviewCard")
        self._preview_label.setWordWrap(True)
        self._preview_label.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._preview_label)

        self._stages_group = QGroupBox("Advanced: Custom stages")
        self._stages_group.setObjectName("dialogSection")
        stages_layout = QVBoxLayout(self._stages_group)
        for stage in ANALYZE_LIBRARY_STAGE_ORDER:
            checkbox = QCheckBox(format_analyze_library_action_label(stage))
            checkbox.setObjectName("dialogOptionToggle")
            checkbox.toggled.connect(self._on_stage_toggled)
            self._stage_checks[stage] = checkbox
            stages_layout.addWidget(checkbox)
        layout.addWidget(self._stages_group)

        self._force_check = QCheckBox("Force rebuild")
        self._force_check.setObjectName("dialogOptionToggle")
        self._force_check.setToolTip("Run selected stages even when cached outputs appear current")
        self._force_check.toggled.connect(self._update_preview)
        layout.addWidget(self._force_check)

        self._dry_run_check = QCheckBox("Dry run")
        self._dry_run_check.setObjectName("dialogOptionToggle")
        self._dry_run_check.setToolTip("Show the selected run plan without making changes")
        self._dry_run_check.toggled.connect(self._update_preview)
        layout.addWidget(self._dry_run_check)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _apply_preset(self, preset: str) -> None:
        stages = set(PRESET_STAGES.get(preset, PRESET_STAGES["Full Analyze"]))
        for stage, checkbox in self._stage_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(stage in stages)
            checkbox.setEnabled(preset == "Custom")
            checkbox.blockSignals(False)
        self._stages_group.setVisible(preset == "Custom")
        self._update_preview()

    def _on_stage_toggled(self) -> None:
        if self._preset_combo.currentText() != "Custom":
            return
        if any(checkbox.isChecked() for checkbox in self._stage_checks.values()):
            return
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            sender.blockSignals(True)
            sender.setChecked(True)
            sender.blockSignals(False)
        self._update_preview()

    def selected_stages(self) -> list[str]:
        return [
            stage
            for stage in ANALYZE_LIBRARY_STAGE_ORDER
            if self._stage_checks[stage].isChecked()
        ]

    def build_request(
        self,
        config_path: str,
        overrides: dict[str, Any],
    ) -> LibraryPipelineRequest:
        return LibraryPipelineRequest(
            config_path=config_path,
            overrides=dict(overrides or {}),
            stages=self.selected_stages(),
            force=self._force_check.isChecked(),
            dry_run=self._dry_run_check.isChecked(),
        )

    def _update_preview(self) -> None:
        stages = self.selected_stages()
        stage_text = format_analyze_library_action_list(stages)
        preview = f"Will run: {stage_text}" if stage_text else "Will run: no stages selected"
        options = []
        if self._force_check.isChecked():
            options.append("force rebuild")
        if self._dry_run_check.isChecked():
            options.append("dry run")
        if options:
            preview += f"\nOptions: {', '.join(options)}"
        self._preview_label.setText(preview)

    def set_preset_for_testing(self, preset: str) -> None:
        self._preset_combo.setCurrentText(preset)

    def set_stage_checked_for_testing(self, stage: str, checked: bool) -> None:
        self._stage_checks[stage].setChecked(checked)

    def set_force_for_testing(self, checked: bool) -> None:
        self._force_check.setChecked(checked)

    def set_dry_run_for_testing(self, checked: bool) -> None:
        self._dry_run_check.setChecked(checked)
