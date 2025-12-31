"""
Playlist Generator GUI - Application Entry Point

Usage:
    python -m playlist_gui.app

This launches the native Windows desktop GUI for the playlist generator.
"""
import os
import sys
import logging
from pathlib import Path
from .gui_logging import setup_gui_logging

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup Python path for running from various locations."""
    # Add project root to path if not already there
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent
    project_root = src_dir.parent

    for path in [str(project_root), str(src_dir)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Set working directory to project root
    os.chdir(project_root)


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        import platformdirs
    except ImportError:
        # Optional, just warn
        logger.warning("Note: platformdirs not installed, using fallback paths")

    if missing:
        logger.error("Missing required dependencies: %s", ", ".join(missing))
        logger.error("Install with: pip install PySide6 pyyaml platformdirs")
        sys.exit(1)


def main():
    """Main entry point for the GUI application."""
    setup_environment()
    check_dependencies()
    emitter, log_buffer, log_path = setup_gui_logging()

    # Import Qt after environment setup
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Playlist Generator")
    app.setOrganizationName("PlaylistGenerator")

    # Apply a clean style
    app.setStyle("Fusion")

    # Import and create main window
    from .main_window import MainWindow
    window = MainWindow(log_emitter=emitter, log_buffer=log_buffer, log_path=log_path)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
