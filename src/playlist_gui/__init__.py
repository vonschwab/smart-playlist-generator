"""Worker-process package (NDJSON protocol over stdin/stdout) plus the
policy layer shared with the web GUI.

The PySide6 desktop GUI that used to live here was removed 2026-06-10
(see docs/DEAD_CODE_AUDIT_2026-06-10.md); the browser GUI under
src/playlist_web + web/ is the maintained front-end. Survivors: worker.py
(spawned by the web bridge), policy.py + ui_state.py (imported by the web
app), utils/redaction.py.
"""
__version__ = "4.0.0"
