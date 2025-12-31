"""
Main Window - Primary GUI window for the playlist generator

Layout:
- Top: Simple controls (mode, artist, track, generate button, progress)
- Center: Track table
- Right dock: Advanced settings panel (includes config file selection)
- Bottom dock: Log panel
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QThreadPool, QRunnable, QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .autocomplete import DatabaseCompleter, setup_artist_completer, update_track_completer
from .config.config_model import ConfigModel
from .config.presets import PresetManager, install_builtin_presets
from .gui_logging import LogEmitter
from .jobs import JobManager, JobType
from .widgets.advanced_panel import AdvancedSettingsPanel
from .widgets.export_dialog import ExportLocalDialog, ExportPlexDialog
from .widgets.jobs_panel import JobsPanel
from .widgets.log_panel import LogPanel
from .widgets.track_table import TrackTable
from .worker_client import WorkerClient
from .diagnostics.checks import CheckResult
from .diagnostics.manager import DiagnosticsManager


class DebugReportSignals(QObject):
    finished = Signal(str, object)  # text, save_path
    failed = Signal(str)


class DebugReportTask(QRunnable):
    """QRunnable to build (and optionally save) a debug report off the UI thread."""

    def __init__(self, args: dict, save_path: Optional[Path] = None):
        super().__init__()
        self.signals = DebugReportSignals()
        self._args = args
        self._save_path = save_path

    def run(self) -> None:  # pragma: no cover - exercised via UI
        from .diagnostics.report import build_debug_report

        try:
            report = build_debug_report(**self._args)
            if self._save_path:
                self._save_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_path.write_text(report, encoding="utf-8")
            self.signals.finished.emit(report, self._save_path)
        except Exception as e:  # pragma: no cover - defensive
            self.signals.failed.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window.

    Features:
    - Artist mode as default
    - Artist input with autocomplete
    - Optional track input with autocomplete (for single-track seeding)
    - Generate button with progress
    - Track table for results
    - Advanced settings dock (includes config file selection)
    - Log panel dock
    - Override status indicator showing active overrides and preset state
    """

    def __init__(self, log_emitter: Optional[LogEmitter] = None, log_buffer=None, log_path: Optional[Path] = None):
        super().__init__()

        self.setWindowTitle("Playlist Generator")
        self.setMinimumSize(1200, 800)

        # State
        self._config_path = "config.yaml"
        self._config_model: Optional[ConfigModel] = None
        self._worker_client: Optional[WorkerClient] = None
        self._job_manager: Optional[JobManager] = None
        self._preset_manager = PresetManager()
        self._is_generating = False
        self._db_completer: Optional[DatabaseCompleter] = None
        self._jobs_panel: Optional[JobsPanel] = None
        self._jobs_dock: Optional[QDockWidget] = None
        self._log_emitter = log_emitter
        self._log_buffer = log_buffer
        self._log_path = log_path
        self._settings = QSettings("PlaylistGenerator", "PlaylistGeneratorGUI")
        self._logger = logging.getLogger("playlist_gui.main_window")
        self._diag_pool = QThreadPool.globalInstance()
        self._diagnostics: Optional[DiagnosticsManager] = None
        self._last_diagnostics: List[CheckResult] = []
        self._last_diagnostics_checked_at = None
        self._diagnostics_banner_on_fail = False
        self._banner_frame: Optional[QFrame] = None
        self._banner_label: Optional[QLabel] = None
        self._banner_time_label: Optional[QLabel] = None
        self._report_tasks: List[DebugReportTask] = []

        # Preset/override tracking state
        self._active_preset_name: Optional[str] = None
        self._preset_overrides_snapshot: dict = {}  # Snapshot of overrides when preset loaded
        self._dirty_overrides = False  # True when overrides differ from loaded preset
        self._pending_preset_name: Optional[str] = None

        # Current playlist state (for export naming)
        self._current_playlist_name: str = ""
        self._current_artist_name: str = ""
        self._current_tracks: list = []

        # Install built-in presets
        install_builtin_presets(self._preset_manager)

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_worker()
        self._diagnostics = DiagnosticsManager(
            base_config_provider=lambda: self._config_path,
            config_model_provider=lambda: self._config_model,
            worker_client=self._worker_client,
        )
        self._diagnostics.diagnostics_updated.connect(self._on_diagnostics_updated)
        if self._worker_client:
            self._worker_client.busy_changed.connect(self._diagnostics.handle_busy_changed)
        self._job_manager = JobManager(self._worker_client)
        self._setup_jobs_dock()
        self._restore_settings()

        # Try to load default config
        self._load_config(self._config_path)

        # Setup autocomplete after a brief delay to allow config load
        QTimer.singleShot(500, self._setup_autocomplete)
        QTimer.singleShot(1200, lambda: self._run_diagnostics(show_banner_on_fail=True, include_worker=False))

    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # Diagnostics banner (hidden by default)
        self._banner_frame = QFrame()
        self._banner_frame.setStyleSheet("QFrame { background: #fff4e5; border: 1px solid #f0c36d; border-radius: 4px; }")
        banner_layout = QHBoxLayout(self._banner_frame)
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self._banner_label = QLabel("")
        banner_layout.addWidget(self._banner_label, stretch=1)
        self._banner_time_label = QLabel("")
        self._banner_time_label.setStyleSheet("color: #555; font-size: 11px;")
        banner_layout.addWidget(self._banner_time_label)
        btn_jobs = QPushButton("Open Jobs")
        btn_jobs.clicked.connect(lambda: self._jobs_dock.show() if self._jobs_dock else None)
        banner_layout.addWidget(btn_jobs)
        btn_scan = QPushButton("Scan Library")
        btn_scan.clicked.connect(self._on_scan_library)
        banner_layout.addWidget(btn_scan)
        btn_art = QPushButton("Build Artifacts")
        btn_art.clicked.connect(self._on_build_artifacts)
        banner_layout.addWidget(btn_art)
        btn_debug = QPushButton("Copy Debug Report")
        btn_debug.clicked.connect(self._on_copy_debug_report)
        banner_layout.addWidget(btn_debug)
        btn_rerun = QPushButton("Re-run")
        btn_rerun.clicked.connect(lambda: self._run_diagnostics(force=True, show_banner_on_fail=True))
        banner_layout.addWidget(btn_rerun)
        btn_retry = QPushButton("Retry Queue")
        btn_retry.clicked.connect(self._on_retry_queue)
        banner_layout.addWidget(btn_retry)
        btn_dismiss = QPushButton("Dismiss")
        btn_dismiss.clicked.connect(self._hide_banner)
        banner_layout.addWidget(btn_dismiss)
        self._banner_frame.hide()
        main_layout.addWidget(self._banner_frame)

        # ─────────────────────────────────────────────────────────────────────
        # Top controls - Clean, artist-focused interface
        # ─────────────────────────────────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Mode selector
        top_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Artist", "History"])  # Artist first (default)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_row.addWidget(self._mode_combo)

        top_row.addSpacing(10)

        # Artist input with autocomplete
        self._artist_label = QLabel("Artist:")
        top_row.addWidget(self._artist_label)
        self._artist_edit = QLineEdit()
        self._artist_edit.setPlaceholderText("Start typing artist name...")
        self._artist_edit.setFixedWidth(220)
        self._artist_edit.textChanged.connect(self._on_artist_changed)
        top_row.addWidget(self._artist_edit)

        top_row.addSpacing(10)

        # Track input (optional, for single-track seeding) with autocomplete
        self._track_label = QLabel("Track (optional):")
        top_row.addWidget(self._track_label)
        self._track_edit = QLineEdit()
        self._track_edit.setPlaceholderText("Seed from specific track...")
        self._track_edit.setFixedWidth(250)
        top_row.addWidget(self._track_edit)

        top_row.addStretch()

        # Generate button
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedWidth(100)
        self._generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86c7;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #5a96d7;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        top_row.addWidget(self._generate_btn)

        # Cancel button
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.setEnabled(False)  # Disabled by default
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #c74a4a;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #d75a5a;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self._cancel_btn.clicked.connect(self._on_cancel)
        top_row.addWidget(self._cancel_btn)

        main_layout.addLayout(top_row)

        # ─────────────────────────────────────────────────────────────────────
        # Progress bar
        # ─────────────────────────────────────────────────────────────────────
        progress_row = QHBoxLayout()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        progress_row.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setFixedWidth(200)
        progress_row.addWidget(self._stage_label)

        main_layout.addLayout(progress_row)

        # ─────────────────────────────────────────────────────────────────────
        # Track table (center)
        # ─────────────────────────────────────────────────────────────────────
        self._track_table = TrackTable()
        main_layout.addWidget(self._track_table)

        # ─────────────────────────────────────────────────────────────────────
        # Export buttons row
        # ─────────────────────────────────────────────────────────────────────
        export_row = QHBoxLayout()
        export_row.setSpacing(10)

        export_row.addStretch()

        self._export_local_btn = QPushButton("Export to Local (M3U8)")
        self._export_local_btn.setEnabled(False)
        self._export_local_btn.clicked.connect(self._on_export_local)
        self._export_local_btn.setToolTip("Export playlist to M3U8 file")
        export_row.addWidget(self._export_local_btn)

        self._export_plex_btn = QPushButton("Export to Plex")
        self._export_plex_btn.setStyleSheet("""
            QPushButton {
                background-color: #e5a00d;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #cc8a00;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self._export_plex_btn.setEnabled(False)
        self._export_plex_btn.clicked.connect(self._on_export_plex)
        self._export_plex_btn.setToolTip("Export playlist to Plex")
        export_row.addWidget(self._export_plex_btn)

        main_layout.addLayout(export_row)

        # ─────────────────────────────────────────────────────────────────────
        # Log panel (bottom dock)
        # ─────────────────────────────────────────────────────────────────────
        self._log_panel = LogPanel()
        if self._log_emitter:
            self._log_emitter.log_ready.connect(self._log_panel.append_log)
        log_dock = QDockWidget("Logs", self)
        log_dock.setWidget(self._log_panel)
        log_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # ─────────────────────────────────────────────────────────────────────
        # Advanced settings (right dock) - includes config file selection
        # ─────────────────────────────────────────────────────────────────────
        self._advanced_panel: Optional[AdvancedSettingsPanel] = None
        self._advanced_dock = QDockWidget("Advanced Settings", self)
        self._advanced_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self._advanced_dock)

        # ─────────────────────────────────────────────────────────────────────
        # Status bar with override indicator
        # ─────────────────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # Override status indicator (permanent widget on right side)
        self._override_status_label = QLabel("Base config")
        self._override_status_label.setStyleSheet("""
            QLabel {
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
            }
        """)
        self._status_bar.addPermanentWidget(self._override_status_label)

        self._status_bar.showMessage("Ready")

        # Initial state - Artist mode is default
        self._update_mode_ui()

    def _create_advanced_panel_content(self) -> QWidget:
        """Create the advanced settings panel with config selector at top."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # ─────────────────────────────────────────────────────────────────────
        # Config file selector (at top of advanced settings)
        # ─────────────────────────────────────────────────────────────────────
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(4)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Config file:"))

        self._config_path_edit = QLineEdit(self._config_path)
        self._config_path_edit.setReadOnly(True)
        config_row.addWidget(self._config_path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_config)
        config_row.addWidget(browse_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.setFixedWidth(60)
        reload_btn.clicked.connect(self._on_reload_config)
        config_row.addWidget(reload_btn)

        config_layout.addLayout(config_row)

        # Database info
        self._db_info_label = QLabel("Database: Not loaded")
        self._db_info_label.setStyleSheet("color: #666; font-size: 11px;")
        config_layout.addWidget(self._db_info_label)

        layout.addWidget(config_group)

        # ─────────────────────────────────────────────────────────────────────
        # Advanced settings panel (scrollable)
        # ─────────────────────────────────────────────────────────────────────
        if self._config_model:
            self._advanced_panel = AdvancedSettingsPanel(self._config_model)
            self._advanced_panel.override_changed.connect(self._on_override_changed)
            layout.addWidget(self._advanced_panel, stretch=1)
        else:
            placeholder = QLabel("Load a config file to see settings")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder, stretch=1)

        return container

    def _update_override_status(self) -> None:
        """Update the override status indicator in the status bar."""
        if not self._config_model:
            self._override_status_label.setText("No config")
            self._override_status_label.setStyleSheet("""
                QLabel {
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    color: #666;
                }
            """)
            return

        override_count = self._config_model.override_count()

        if self._active_preset_name:
            # Preset is loaded
            if self._dirty_overrides:
                text = f"Preset: {self._active_preset_name} (modified)"
                style = """
                    QLabel {
                        padding: 2px 8px;
                        border-radius: 3px;
                        font-size: 11px;
                        background-color: #fff3cd;
                        color: #856404;
                        border: 1px solid #ffc107;
                    }
                """
            else:
                text = f"Preset: {self._active_preset_name}"
                style = """
                    QLabel {
                        padding: 2px 8px;
                        border-radius: 3px;
                        font-size: 11px;
                        background-color: #d4edda;
                        color: #155724;
                        border: 1px solid #28a745;
                    }
                """
        elif override_count > 0:
            # Overrides active but no preset
            text = f"Overrides active ({override_count})"
            style = """
                QLabel {
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    background-color: #cce5ff;
                    color: #004085;
                    border: 1px solid #007bff;
                }
            """
        else:
            # Base config
            text = "Base config"
            style = """
                QLabel {
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    color: #666;
                }
            """

        self._override_status_label.setText(text)
        self._override_status_label.setStyleSheet(style)

    def _check_dirty_overrides(self) -> None:
        """Check if current overrides differ from the loaded preset snapshot."""
        if not self._active_preset_name:
            self._dirty_overrides = False
            return

        if not self._config_model:
            self._dirty_overrides = False
            return

        current_overrides = self._config_model.list_overrides()
        snapshot_flat = {}
        self._config_model._flatten_dict(self._preset_overrides_snapshot, "", snapshot_flat)

        # Compare current overrides with snapshot
        self._dirty_overrides = current_overrides != snapshot_flat

    @Slot()
    def _on_override_changed(self) -> None:
        """Handle override state changes from the advanced panel."""
        self._check_dirty_overrides()
        self._update_override_status()

    def _setup_menu(self) -> None:
        """Setup the menu bar."""
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # File menu
        file_menu = QMenu("&File", self)
        menubar.addMenu(file_menu)

        file_menu.addAction("&Open Config...", self._on_browse_config)
        file_menu.addAction("&Reload Config", self._on_reload_config)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        # Presets menu
        presets_menu = QMenu("&Presets", self)
        menubar.addMenu(presets_menu)

        presets_menu.addAction("&Save Preset...", self._on_save_preset)
        presets_menu.addAction("&Load Preset...", self._on_load_preset)
        presets_menu.addSeparator()
        presets_menu.addAction("&Reset to Base Config", self._on_reset_overrides)

        # Tools menu
        self._tools_menu = QMenu("&Tools", self)
        tools_menu = self._tools_menu
        menubar.addMenu(tools_menu)

        tools_menu.addAction("Run &Diagnostics", self._on_run_diagnostics)
        tools_menu.addAction("Run &Pipeline (Scan->Genres->Sonic->Artifacts)", self._on_run_pipeline)
        tools_menu.addAction("Scan &Library", self._on_scan_library)
        tools_menu.addAction("Update &Genres", self._on_update_genres)
        tools_menu.addAction("Update &Sonic Features", self._on_update_sonic)
        tools_menu.addAction("&Build Artifacts", self._on_build_artifacts)

        # View menu
        view_menu = QMenu("&View", self)
        menubar.addMenu(view_menu)

        view_menu.addAction("&Advanced Settings", lambda: self._advanced_dock.show())
        view_menu.addAction("&Logs", lambda: self._log_panel.parentWidget().show())
        view_menu.addAction("&Jobs", lambda: self._jobs_dock.show() if self._jobs_dock else None)
        view_menu.addAction("Reset &UI Layout", self._reset_ui_layout)

        # Help menu
        help_menu = QMenu("&Help", self)
        menubar.addMenu(help_menu)

        help_menu.addAction("&About", self._on_about)
        help_menu.addAction("Copy Debug Report", self._on_copy_debug_report)
        help_menu.addAction("Save Debug Report...", self._on_save_debug_report)

    def _setup_worker(self) -> None:
        """Setup the worker client."""
        self._worker_client = WorkerClient(self)

        # Connect signals
        self._worker_client.log_received.connect(self._on_worker_log)
        self._worker_client.progress_received.connect(self._on_worker_progress)
        self._worker_client.result_received.connect(self._on_worker_result)
        self._worker_client.error_received.connect(self._on_worker_error)
        self._worker_client.done_received.connect(self._on_worker_done)
        self._worker_client.worker_started.connect(self._on_worker_started)
        self._worker_client.worker_stopped.connect(self._on_worker_stopped)
        self._worker_client.busy_changed.connect(self._on_busy_changed)

    def _setup_jobs_dock(self) -> None:
        """Create the Jobs dock/panel."""
        if not self._job_manager:
            return

        self._jobs_panel = JobsPanel(
            self._job_manager,
            on_run_pipeline=self._on_run_pipeline,
            on_cancel_active=self._on_cancel_jobs,
            on_cancel_pending=self._on_clear_pending_jobs,
        )
        self._jobs_dock = QDockWidget("Jobs", self)
        self._jobs_dock.setWidget(self._jobs_panel)
        self._jobs_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self._jobs_dock)

    def _setup_autocomplete(self) -> None:
        """Setup autocomplete for artist and track inputs."""
        if not self._config_model:
            return

        # Get database path from config
        db_path = self._config_model.get("library.database_path", "data/metadata.db")

        if not Path(db_path).exists():
            self._log_panel.append_log("WARNING", f"Database not found: {db_path}")
            return

        self._db_completer = DatabaseCompleter(db_path)

        if self._db_completer.load_data():
            # Setup artist autocomplete
            setup_artist_completer(self._artist_edit, self._db_completer)

            # Update database info
            if hasattr(self, '_db_info_label'):
                self._db_info_label.setText(
                    f"Database: {self._db_completer.artist_count} artists, "
                    f"{self._db_completer.track_count} tracks"
                )

            self._log_panel.append_log(
                "INFO",
                f"Loaded autocomplete: {self._db_completer.artist_count} artists, "
                f"{self._db_completer.track_count} tracks"
            )
        else:
            self._log_panel.append_log("WARNING", "Failed to load autocomplete data")

    def _load_config(self, path: str) -> bool:
        """Load configuration from file."""
        try:
            if not Path(path).exists():
                self._log_panel.append_log("WARNING", f"Config not found: {path}")
                return False

            self._config_model = ConfigModel(path)
            self._config_path = path

            # Clear preset state when loading new config
            self._active_preset_name = None
            self._preset_overrides_snapshot = {}
            self._dirty_overrides = False

            # Update config path display if it exists
            if hasattr(self, '_config_path_edit'):
                self._config_path_edit.setText(path)

            # Rebuild advanced panel with new config model
            advanced_content = self._create_advanced_panel_content()
            self._advanced_dock.setWidget(advanced_content)

            self._log_panel.append_log("INFO", f"Loaded config: {path}")
            self._status_bar.showMessage(f"Config: {path}")

            # Update override status indicator
            self._update_override_status()

            # Reload autocomplete with new config
            self._setup_autocomplete()

            # Restore preset if pending
            if self._pending_preset_name:
                presets = [p["name"] for p in self._preset_manager.list_presets()]
                if self._pending_preset_name in presets:
                    overrides = self._preset_manager.load_preset(self._pending_preset_name)
                    if overrides:
                        self._config_model.set_overrides(overrides)
                        if self._advanced_panel:
                            self._advanced_panel.refresh()
                        self._active_preset_name = self._pending_preset_name
                        self._preset_overrides_snapshot = overrides.copy()
                        self._dirty_overrides = False
                        self._update_override_status()
                self._pending_preset_name = None

            return True

        except Exception as e:
            self._log_panel.append_log("ERROR", f"Failed to load config: {e}")
            return False

    def _update_mode_ui(self) -> None:
        """Update UI based on selected mode."""
        is_artist_mode = self._mode_combo.currentText() == "Artist"
        self._artist_label.setVisible(is_artist_mode)
        self._artist_edit.setVisible(is_artist_mode)
        self._track_label.setVisible(is_artist_mode)
        self._track_edit.setVisible(is_artist_mode)

    def _ensure_artifacts_ready(self) -> bool:
        """
        Guardrail: warn if required artifacts are missing before playlist generation.

        Returns True if it's safe to proceed, False if the operation should stop.
        """
        if not self._config_model:
            return True

        artifact_path = self._config_model.get("playlists.ds_pipeline.artifact_path")
        if not artifact_path:
            return True

        path = Path(artifact_path)
        base_dir = Path(self._config_path).parent if self._config_path else Path.cwd()
        if not path.is_absolute():
            path = base_dir / path

        if path.exists():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Artifacts Not Found")
        box.setText(f"Required artifacts were not found:\n{path}")
        box.setInformativeText("Run the library pipeline now? This queues Scan -> Genres -> Sonic -> Artifacts.")
        run_btn = box.addButton("Run Pipeline", QMessageBox.AcceptRole)
        continue_btn = box.addButton("Continue Anyway", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(run_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked == run_btn:
            self._on_run_pipeline()
            return False
        if clicked == continue_btn:
            return True
        return False

    def _show_banner(self, message: str, checked_at: Optional[datetime] = None) -> None:
        if self._banner_frame and self._banner_label:
            timestamp = ""
            if checked_at:
                local_time = checked_at.astimezone()
                timestamp = f"Last checked: {local_time.strftime('%H:%M')}"
            if self._banner_time_label:
                self._banner_time_label.setText(timestamp)
            self._banner_label.setText(message)
            self._banner_frame.show()

    def _hide_banner(self) -> None:
        if self._banner_frame:
            self._banner_frame.hide()
        if self._banner_time_label:
            self._banner_time_label.setText("")

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_artist_changed(self, artist: str) -> None:
        """Handle artist input change - update track autocomplete."""
        if self._db_completer and artist:
            update_track_completer(self._track_edit, self._db_completer, artist)

    @Slot()
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        if self._worker_client and self._worker_client.is_busy():
            self._log_panel.append_log("INFO", "Requesting cancellation...")
            self._worker_client.cancel()
            self._cancel_btn.setEnabled(False)  # Disable until we get a response
            self._stage_label.setText("Cancelling...")

    @Slot(bool)
    def _on_busy_changed(self, is_busy: bool) -> None:
        """Handle busy state change from worker client."""
        if self._diagnostics:
            self._diagnostics.handle_busy_changed(is_busy)
        self._is_generating = is_busy
        self._generate_btn.setEnabled(not is_busy)
        self._cancel_btn.setEnabled(is_busy)

        # Disable/enable tools menu items when busy
        # (Tools menu is at index 2 in the menu bar)
        if hasattr(self, '_tools_menu'):
            for action in self._tools_menu.actions():
                action.setEnabled(True)

        if not is_busy:
            # Reset progress when not busy
            self._progress_bar.setValue(0)
            self._stage_label.setText("")

    @Slot()
    def _on_browse_config(self) -> None:
        """Browse for config file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            str(Path.cwd()),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if path:
            self._load_config(path)

    @Slot()
    def _on_reload_config(self) -> None:
        """Reload current config file."""
        if self._config_path:
            self._load_config(self._config_path)

    @Slot(str)
    def _on_mode_changed(self, mode: str) -> None:
        """Handle mode change."""
        self._update_mode_ui()

    @Slot()
    def _on_generate(self) -> None:
        """Start playlist generation."""
        if self._is_generating:
            return

        if not self._config_model:
            QMessageBox.warning(self, "No Config", "Please load a configuration file first.")
            return

        # Refresh diagnostics but avoid worker doctor when busy
        include_worker = not (self._worker_client and self._worker_client.is_busy())
        self._run_diagnostics(show_banner_on_fail=True, include_worker=include_worker)

        if not self._ensure_artifacts_ready():
            return

        # Start worker if not running
        if not self._worker_client.is_running():
            if not self._worker_client.start():
                QMessageBox.critical(self, "Worker Error", "Failed to start worker process.")
                return

        # Get parameters
        mode = "artist" if self._mode_combo.currentText() == "Artist" else "history"
        artist = self._artist_edit.text().strip() if mode == "artist" else None
        track = self._track_edit.text().strip() if mode == "artist" else None
        # Track count is now controlled via config's playlists.tracks_per_playlist
        # (visible in Advanced Settings panel), not a separate spinner
        tracks_per_playlist = self._config_model.get("playlists.tracks_per_playlist", 30)

        if mode == "artist" and not artist:
            QMessageBox.warning(self, "Missing Artist", "Please enter an artist name.")
            return

        # Parse track title from autocomplete format "Title - Artist (Album)"
        if track and " - " in track:
            track = track.split(" - ")[0].strip()

        # Clear previous results
        self._track_table.clear()
        self._progress_bar.setValue(0)
        self._stage_label.setText("Starting...")
        self._is_generating = True
        self._generate_btn.setEnabled(False)

        # Get overrides from config model
        overrides = self._config_model.get_overrides()

        # Send command - tracks parameter comes from config (playlists.tracks_per_playlist)
        self._worker_client.generate_playlist(
            config_path=self._config_path,
            overrides=overrides,
            mode=mode,
            artist=artist,
            track=track,
            tracks=tracks_per_playlist
        )

        log_msg = f"Starting generation (mode={mode}, tracks={tracks_per_playlist})"
        if artist:
            log_msg += f", artist={artist}"
        if track:
            log_msg += f", seed_track={track}"
        self._log_panel.append_log("INFO", log_msg)

    def _run_diagnostics(self, show_banner_on_fail: bool = False, force: bool = False, include_worker: bool = True) -> None:
        """Trigger diagnostics via the diagnostics manager."""
        if not self._diagnostics:
            return
        self._diagnostics_banner_on_fail = self._diagnostics_banner_on_fail or show_banner_on_fail
        self._diagnostics.run_checks(force=force, include_worker=include_worker)

    def _on_diagnostics_updated(self, results: List[CheckResult], checked_at: Optional[datetime]) -> None:
        """Handle diagnostics completion."""
        self._last_diagnostics = results or []
        self._last_diagnostics_checked_at = checked_at
        failures = [r for r in self._last_diagnostics if not r.ok]
        if failures:
            msg = failures[0].detail or f"{failures[0].name} failed"
            self._show_banner(f"Diagnostics: {msg}", checked_at=checked_at)
        elif self._diagnostics_banner_on_fail:
            self._hide_banner()
        self._diagnostics_banner_on_fail = False

    @Slot()
    def _on_save_preset(self) -> None:
        """Save current settings as a preset."""
        if not self._config_model:
            return

        from PySide6.QtWidgets import QInputDialog

        overrides = self._config_model.get_overrides()
        if not overrides:
            QMessageBox.information(self, "No Changes", "No settings have been modified.")
            return

        # If a preset is already loaded, offer to update it
        default_name = self._active_preset_name or ""
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            text=default_name
        )

        if ok and name:
            self._preset_manager.save_preset(name, overrides)
            self._log_panel.append_log("INFO", f"Saved preset: {name}")

            # Update preset state - now this preset is active and not dirty
            self._active_preset_name = name
            self._preset_overrides_snapshot = overrides.copy()
            self._dirty_overrides = False
            self._update_override_status()

    @Slot()
    def _on_load_preset(self) -> None:
        """Load a preset."""
        if not self._config_model:
            return

        presets = self._preset_manager.list_presets()
        if not presets:
            QMessageBox.information(self, "No Presets", "No presets available.")
            return

        from PySide6.QtWidgets import QInputDialog
        names = [p["name"] for p in presets]
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select preset:", names, 0, False)

        if ok and name:
            overrides = self._preset_manager.load_preset(name)
            if overrides:
                self._config_model.set_overrides(overrides)
                if self._advanced_panel:
                    self._advanced_panel.refresh()
                self._log_panel.append_log("INFO", f"Loaded preset: {name}")

                # Track preset state
                self._active_preset_name = name
                self._preset_overrides_snapshot = overrides.copy()
                self._dirty_overrides = False
                self._update_override_status()

    @Slot()
    def _on_reset_overrides(self) -> None:
        """Reset all overrides to base config."""
        if self._config_model:
            self._config_model.reset()
            if self._advanced_panel:
                self._advanced_panel.refresh()
            self._log_panel.append_log("INFO", "Reset to base configuration")

            # Clear preset state
            self._active_preset_name = None
            self._preset_overrides_snapshot = {}
            self._dirty_overrides = False
            self._update_override_status()

    @Slot()
    def _on_scan_library(self) -> None:
        """Run library scan."""
        self._enqueue_job(JobType.SCAN_LIBRARY)

    @Slot()
    def _on_update_genres(self) -> None:
        """Run genre update."""
        self._enqueue_job(JobType.UPDATE_GENRES)

    @Slot()
    def _on_update_sonic(self) -> None:
        """Run sonic feature extraction."""
        self._enqueue_job(JobType.UPDATE_SONIC)

    @Slot()
    def _on_build_artifacts(self) -> None:
        """Run artifact building."""
        self._enqueue_job(JobType.BUILD_ARTIFACTS)

    def _enqueue_job(self, job_type: JobType) -> None:
        """Queue a job through the JobManager."""
        if not self._config_model:
            QMessageBox.warning(self, "No Config", "Load a configuration before queuing jobs.")
            return
        if not self._job_manager:
            QMessageBox.warning(self, "Jobs Unavailable", "Job manager is not initialized.")
            return

        overrides = dict(self._config_model.get_overrides())
        job = self._job_manager.enqueue_job(job_type, self._config_path, overrides)
        self._log_panel.append_log("INFO", f"Queued job: {job_type.label()} ({job.job_id[:8]})")
        self._status_bar.showMessage(f"Queued {job_type.label()} job")

    @Slot()
    def _on_run_pipeline(self) -> None:
        """Queue the full library pipeline."""
        if not self._config_model:
            QMessageBox.warning(self, "No Config", "Load a configuration before queuing jobs.")
            return
        if not self._job_manager:
            QMessageBox.warning(self, "Jobs Unavailable", "Job manager is not initialized.")
            return

        overrides = dict(self._config_model.get_overrides())
        self._job_manager.enqueue_pipeline(self._config_path, overrides)
        self._log_panel.append_log("INFO", "Queued pipeline: Scan -> Genres -> Sonic -> Artifacts")
        self._status_bar.showMessage("Pipeline queued")

    def _on_cancel_jobs(self) -> None:
        """Cancel the active job if running."""
        if self._job_manager:
            self._job_manager.cancel_active_job()

    def _on_clear_pending_jobs(self) -> None:
        """Clear pending jobs from the queue."""
        if self._job_manager:
            self._job_manager.cancel_pending()

    @Slot()
    def _on_retry_queue(self) -> None:
        """Retry previously skipped jobs after a crash."""
        if not self._job_manager:
            return
        self._job_manager.retry_skipped()
        self._hide_banner()
    @Slot()
    def _on_run_diagnostics(self) -> None:
        """Manually trigger diagnostics."""
        self._run_diagnostics(show_banner_on_fail=True, force=True)

    @Slot()
    def _on_copy_debug_report(self) -> None:
        """Generate and copy a debug report."""
        args = self._collect_debug_report_args()
        task = DebugReportTask(args)
        task.signals.finished.connect(lambda text, path, task=task: self._on_debug_report_ready(text, True, path, task))
        task.signals.failed.connect(lambda msg, task=task: self._on_debug_report_failed(msg, task))
        self._report_tasks.append(task)
        self._diag_pool.start(task)

    @Slot()
    def _on_save_debug_report(self) -> None:
        """Generate and save a debug report to disk."""
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Debug Report",
            str(Path.cwd() / "debug_report.txt"),
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path_str:
            return
        save_path = Path(path_str)
        args = self._collect_debug_report_args()
        task = DebugReportTask(args, save_path=save_path)
        task.signals.finished.connect(lambda text, path, task=task: self._on_debug_report_ready(text, False, path, task))
        task.signals.failed.connect(lambda msg, task=task: self._on_debug_report_failed(msg, task))
        self._report_tasks.append(task)
        self._diag_pool.start(task)

    def _collect_debug_report_args(self) -> dict:
        """Gather context for the debug report builder."""
        preset_label = "Base config"
        if self._active_preset_name:
            preset_label = self._active_preset_name
            if self._dirty_overrides:
                preset_label += " (modified)"

        mode = self._mode_combo.currentText()
        artist = self._artist_edit.text()
        worker_status = "running" if self._worker_client and self._worker_client.is_running() else "not running"
        if self._worker_client and self._worker_client.get_pid():
            worker_status += f" (pid {self._worker_client.get_pid()})"

        last_job_summary = ""
        last_job_error = ""
        if self._job_manager and self._job_manager.jobs():
            last_job = self._job_manager.jobs()[-1]
            last_job_summary = last_job.summary or f"{last_job.job_type.label()} {last_job.status}"
            last_job_error = last_job.error_message

        return {
            "base_config_path": self._config_path,
            "preset_label": preset_label,
            "mode": mode,
            "artist": artist,
            "worker_status": worker_status,
            "last_job_summary": last_job_summary,
            "last_job_error": last_job_error,
            "readiness": self._last_diagnostics,
            "gui_log_path": self._log_path if self._log_path else Path(),
            "worker_events": self._worker_client.get_event_buffer() if self._worker_client else [],
        }

    def _on_debug_report_ready(
        self, text: str, copy_to_clipboard: bool, save_path: Optional[Path], task: Optional[DebugReportTask] = None
    ) -> None:
        """Handle debug report completion."""
        try:
            if task:
                self._report_tasks.remove(task)
        except ValueError:
            pass

        if copy_to_clipboard:
            QApplication.clipboard().setText(text)
            self._status_bar.showMessage("Debug report copied to clipboard", 5000)
        elif save_path:
            self._status_bar.showMessage(f"Debug report saved to {save_path}", 5000)
        self._logger.info("Debug report generated (%s)", "copied" if copy_to_clipboard else "saved")

    def _on_debug_report_failed(self, message: str, task: Optional[DebugReportTask] = None) -> None:
        """Handle debug report failure."""
        try:
            if task:
                self._report_tasks.remove(task)
        except ValueError:
            pass
        self._logger.error("Debug report failed: %s", message)
        self._status_bar.showMessage(f"Debug report failed: {message}", 6000)

    @Slot()
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Playlist Generator",
            "Playlist Generator v1.0\n\n"
            "AI-powered playlist generation using sonic and genre similarity.\n\n"
            "Architecture: Two-process model with PySide6 GUI and worker process."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Worker Signal Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @Slot(str, str, object)
    def _on_worker_log(self, level: str, message: str, job_id: object = None) -> None:
        """Handle log from worker."""
        self._logger.info("worker: %s", message)

    @Slot(str, int, int, str, object)
    def _on_worker_progress(self, stage: str, current: int, total: int, detail: str, job_id: object = None) -> None:
        """Handle progress from worker."""
        if total > 0:
            percent = int((current / total) * 100)
            self._progress_bar.setValue(percent)

        stage_text = detail if detail else stage
        self._stage_label.setText(stage_text)
        self._logger.info("worker progress: %s %s/%s %s", stage, current, total, detail)

    @Slot(str, dict, object)
    def _on_worker_result(self, result_type: str, data: dict, job_id: object = None) -> None:
        """Handle result from worker."""
        if result_type == "playlist":
            playlist = data.get("playlist", {})
            tracks = playlist.get("tracks", [])
            playlist_name = playlist.get("name", "")
            self._track_table.set_tracks(tracks, playlist_name=playlist_name)

            # Store playlist info for export
            self._current_playlist_name = playlist_name
            self._current_tracks = tracks

            # Prefer artist input in Artist mode; fall back to first track
            artist_name = ""
            if self._mode_combo.currentText() == "Artist":
                artist_name = self._artist_edit.text().strip()
            elif tracks:
                artist_name = tracks[0].get("artist", "")
            self._current_artist_name = artist_name

            # Enable export buttons if we have tracks
            has_tracks = len(tracks) > 0
            self._export_local_btn.setEnabled(has_tracks)
            self._export_plex_btn.setEnabled(has_tracks)

            self._log_panel.append_log("INFO", f"Received playlist: {playlist.get('name', 'Unknown')}")
        elif result_type == "doctor":
            checks_raw = data.get("checks", [])
            if self._diagnostics:
                self._diagnostics.handle_worker_doctor(checks_raw)

    @Slot(str, str, object)
    def _on_worker_error(self, message: str, tb: str, job_id: object = None) -> None:
        """Handle error from worker."""
        self._logger.error("worker error: %s", message)
        self._log_panel.append_log("ERROR", message)
        if tb:
            self._log_panel.append_log("DEBUG", f"Traceback: {tb[:500]}...")

    @Slot(str, bool, str, bool, object, str)
    def _on_worker_done(self, cmd: str, ok: bool, detail: str, cancelled: bool, job_id: object = None, summary: str = "") -> None:
        """Handle done signal from worker."""
        # Note: busy state is now managed by _on_busy_changed
        # but we still update UI elements here for the specific completion state

        if cancelled:
            self._progress_bar.setValue(0)
            self._stage_label.setText("Cancelled")
            self._status_bar.showMessage(f"Cancelled: {cmd}")
            self._log_panel.append_log("INFO", f"Operation cancelled: {cmd}")
        elif ok:
            self._progress_bar.setValue(100)
            final_text = summary or detail or "Complete"
            self._stage_label.setText(final_text)
            self._status_bar.showMessage(f"Completed: {cmd} - {final_text}")
        else:
            self._stage_label.setText("Failed")
            self._status_bar.showMessage(f"Failed: {cmd} - {detail}")
        self._logger.info("worker done: %s ok=%s cancelled=%s summary=%s", cmd, ok, cancelled, summary or detail)

    @Slot()
    def _on_worker_started(self) -> None:
        """Handle worker start."""
        self._log_panel.append_log("INFO", "Worker process started")
        self._run_diagnostics(force=True, include_worker=True)

    @Slot(int, str)
    def _on_worker_stopped(self, exit_code: int, status: str) -> None:
        """Handle worker stop."""
        self._log_panel.append_log("INFO", f"Worker stopped (code={exit_code}, status={status})")
        # Worker client handles busy state, but ensure UI is reset if worker crashes
        if self._is_generating:
            self._is_generating = False
            self._generate_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            self._progress_bar.setValue(0)
            self._stage_label.setText("Worker stopped")
        if self._worker_client and self._worker_client.was_busy_on_last_exit():
            self._show_banner("Worker exited unexpectedly. Copy Debug Report?")

    # ─────────────────────────────────────────────────────────────────────────
    # Export Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @Slot()
    def _on_export_local(self) -> None:
        """Export playlist to local M3U8 file."""
        tracks = self._current_tracks
        if not tracks:
            QMessageBox.warning(self, "No Tracks", "No playlist to export.")
            return

        # Get default export directory from config
        default_dir = "E:\\PLAYLISTS"
        if self._config_model:
            default_dir = self._config_model.get(
                "playlists.m3u_export_path",
                default_dir
            )

        # Show export dialog
        dialog = ExportLocalDialog(
            parent=self,
            default_name="",  # Will generate based on artist + date
            default_directory=default_dir,
            artist_name=self._current_artist_name
        )

        if dialog.exec() == QDialog.Accepted:
            export_path = dialog.get_export_path()
            playlist_name = dialog.get_playlist_name()

            if export_path:
                try:
                    self._write_m3u8(export_path, tracks, playlist_name)
                    self._log_panel.append_log("INFO", f"Exported playlist to: {export_path}")
                    self._status_bar.showMessage(f"Exported: {export_path}")

                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Playlist exported to:\n{export_path}"
                    )
                except Exception as e:
                    self._log_panel.append_log("ERROR", f"Export failed: {e}")
                    QMessageBox.critical(
                        self,
                        "Export Failed",
                        f"Failed to export playlist:\n{e}"
                    )

    @Slot()
    def _on_export_plex(self) -> None:
        """Export playlist to Plex."""
        tracks = self._current_tracks
        if not tracks:
            QMessageBox.warning(self, "No Tracks", "No playlist to export.")
            return

        # Check if Plex is configured
        plex_configured = False
        plex_base_url = None
        plex_token = None

        if self._config_model:
            plex_enabled = self._config_model.get("plex.enabled", False)
            plex_base_url = self._config_model.get("plex.base_url")
            import os
            plex_token = os.getenv("PLEX_TOKEN") or self._config_model.get("plex.token")
            plex_configured = plex_enabled and plex_base_url and plex_token

        # Show export dialog
        dialog = ExportPlexDialog(
            parent=self,
            default_name="",  # Will generate based on artist + date
            artist_name=self._current_artist_name,
            plex_configured=plex_configured
        )

        if dialog.exec() == QDialog.Accepted:
            playlist_name = dialog.get_playlist_name()

            if not plex_configured:
                QMessageBox.warning(
                    self,
                    "Plex Not Configured",
                    "Please configure Plex settings in config.yaml:\n"
                    "- plex.enabled: true\n"
                    "- plex.base_url: http://...\n"
                    "- PLEX_TOKEN environment variable or plex.token"
                )
                return

            try:
                # Import and initialize Plex exporter
                from src.plex_exporter import PlexExporter

                plex_section = self._config_model.get("plex.music_section")
                plex_verify_ssl = self._config_model.get("plex.verify_ssl", True)
                plex_replace = self._config_model.get("plex.replace_existing", True)
                plex_path_map = self._config_model.get("plex.path_map")

                exporter = PlexExporter(
                    plex_base_url,
                    plex_token,
                    music_section=plex_section,
                    verify_ssl=plex_verify_ssl,
                    replace_existing=plex_replace,
                    path_map=plex_path_map,
                )

                # Convert tracks to format expected by PlexExporter
                plex_tracks = []
                for track in tracks:
                    plex_tracks.append({
                        "artist": track.get("artist", ""),
                        "title": track.get("title", ""),
                        "album": track.get("album", ""),
                        "file_path": track.get("file_path", ""),
                        "duration": track.get("duration_ms", 0),
                    })

                # Export to Plex
                plex_key = exporter.export_playlist(playlist_name, plex_tracks)

                if plex_key:
                    self._log_panel.append_log("INFO", f"Exported playlist to Plex: {playlist_name}")
                    self._status_bar.showMessage(f"Exported to Plex: {playlist_name}")

                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Playlist '{playlist_name}' exported to Plex."
                    )
                else:
                    raise Exception("Plex export returned no playlist key")

            except ImportError:
                self._log_panel.append_log("ERROR", "PlexExporter not available")
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Plex exporter module not available.\n"
                    "Please ensure plexapi is installed: pip install plexapi"
                )
            except Exception as e:
                self._log_panel.append_log("ERROR", f"Plex export failed: {e}")
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export playlist to Plex:\n{e}"
                )

    def _write_m3u8(self, file_path: Path, tracks: list, playlist_name: str) -> None:
        """
        Write tracks to M3U8 file.

        Args:
            file_path: Output file path
            tracks: List of track dicts with artist, title, duration_ms, file_path
            playlist_name: Name of the playlist for the header
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            f.write(f"#PLAYLIST:{playlist_name}\n")

            for track in tracks:
                artist = track.get("artist", "Unknown")
                title = track.get("title", "Unknown")
                duration_ms = track.get("duration_ms", 0)
                path = track.get("file_path", "")

                # Convert duration from ms to seconds
                duration_sec = duration_ms // 1000 if duration_ms else 0

                f.write(f"#EXTINF:{duration_sec},{artist} - {title}\n")
                f.write(f"{path}\n")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        self._save_settings()
        if self._worker_client:
            self._worker_client.stop()
        event.accept()

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _restore_settings(self) -> None:
        """Restore persisted window/layout and form state."""
        try:
            geo = self._settings.value("ui/geometry")
            if geo:
                self.restoreGeometry(geo)
            state = self._settings.value("ui/state")
            if state:
                self.restoreState(state)
            cfg = self._settings.value("state/config_path")
            if cfg:
                self._config_path = str(cfg)
            mode = self._settings.value("state/mode")
            if mode:
                self._mode_combo.setCurrentText(mode)
            artist = self._settings.value("state/artist")
            if artist:
                self._artist_edit.setText(str(artist))
            preset = self._settings.value("state/preset")
            if preset:
                self._pending_preset_name = str(preset)
            filt = self._settings.value("state/filter")
            if filt:
                self._track_table.set_filter_text(str(filt))
        except Exception as e:
            self._logger.warning("Failed to restore settings: %s", e)

    def _save_settings(self) -> None:
        """Persist window/layout and form state."""
        try:
            self._settings.setValue("ui/geometry", self.saveGeometry())
            self._settings.setValue("ui/state", self.saveState())
            self._settings.setValue("state/config_path", self._config_path)
            self._settings.setValue("state/mode", self._mode_combo.currentText())
            self._settings.setValue("state/artist", self._artist_edit.text())
            self._settings.setValue("state/filter", self._track_table.get_filter_text())
            if self._active_preset_name:
                self._settings.setValue("state/preset", self._active_preset_name)
            else:
                self._settings.remove("state/preset")
        except Exception as e:
            self._logger.warning("Failed to save settings: %s", e)

    def _reset_ui_layout(self) -> None:
        """Reset layout-related persisted state."""
        for key in ["ui/geometry", "ui/state"]:
            self._settings.remove(key)
        self.resize(1200, 800)
        self.showNormal()
        self._logger.info("UI layout reset to defaults")
