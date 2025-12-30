"""
Playlist Generator GUI Package

Native Windows desktop GUI for the playlist generator using PySide6/Qt Widgets.
Architecture: Two-process model with GUI process and worker process communicating
via NDJSON protocol over stdin/stdout.
"""
__version__ = "1.0.0"
