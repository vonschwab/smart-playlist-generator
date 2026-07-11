"""Unit tests for the data-safety PreToolUse guard hook.

Verifies destructive ops against irreplaceable data (metadata.db, the MERT
archive, the audio library) are denied, while reads and backups pass. Loads the
hook by path like tests/test_reuse_first_reminder.py. The audio root normally
comes from a gitignored config.yaml, so audio cases monkeypatch `_audio_root`
to stay deterministic on any machine.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

_HOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "data_safety_guard.py"
)
_spec = importlib.util.spec_from_file_location("data_safety_guard", _HOOK)
assert _spec is not None and _spec.loader is not None, f"hook not found at {_HOOK}"
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _cmd_denied(command):
    """Mirror main(): normalize, then run the command detector."""
    return hook.command_denied(hook._norm(command))


# --------------------------- metadata.db: DENY ---------------------------

def test_rm_db_denied():
    assert _cmd_denied("rm data/metadata.db") is not None


def test_powershell_remove_item_db_denied():
    assert _cmd_denied(r"Remove-Item data\metadata.db") is not None


def test_sqlite_delete_denied():
    assert _cmd_denied('sqlite3 data/metadata.db "DELETE FROM tracks"') is not None


def test_sqlite_update_denied():
    assert _cmd_denied('sqlite3 data/metadata.db "UPDATE tracks SET x=1"') is not None


def test_redirect_onto_db_denied():
    assert _cmd_denied('echo "" > data/metadata.db') is not None


def test_move_db_denied():
    assert _cmd_denied("mv data/metadata.db /tmp/old.db") is not None


def test_python_open_write_db_denied():
    assert _cmd_denied("python -c \"open('data/metadata.db','w')\"") is not None


def test_rm_wal_denied():
    # -wal is live DB state, must be protected too
    assert _cmd_denied("rm data/metadata.db-wal") is not None


def test_cp_live_db_denied():
    # raw file copy of the LIVE db can capture a torn WAL state -> denied (use copy_db_safe.py)
    assert _cmd_denied("cp data/metadata.db data/metadata.db.bak.20260706") is not None
    assert _cmd_denied("cp data/metadata.db /tmp/snapshot.db") is not None


def test_powershell_copy_item_live_db_denied():
    assert _cmd_denied(r"Copy-Item data\metadata.db D:\backup\metadata.db") is not None


def test_inline_python_copyfile_db_denied():
    assert _cmd_denied("python -c \"import shutil; shutil.copyfile('data/metadata.db','x')\"") is not None


# --------------------------- metadata.db: ALLOW ---------------------------

def test_sqlite_select_allowed():
    assert _cmd_denied('sqlite3 data/metadata.db "SELECT * FROM tracks LIMIT 5"') is None


def test_select_with_redirect_elsewhere_allowed():
    # redirect target is out.txt, not the db — must not false-positive
    assert _cmd_denied('sqlite3 data/metadata.db "SELECT 1" > out.txt') is None


def test_cp_backup_source_allowed():
    # copying an EXISTING .bak (e.g. to restore) is fine — only the live db's raw copy is unsafe
    assert _cmd_denied("cp data/metadata.db.bak.20260706 /tmp/restore.db") is None


def test_copy_db_safe_tool_allowed():
    # the blessed atomic tool must not be caught by the copy guard (even naming --src)
    assert _cmd_denied("python tools/copy_db_safe.py data/snapshot.db") is None
    assert _cmd_denied("python tools/copy_db_safe.py /tmp/x.db --src data/metadata.db") is None


def test_rm_backup_allowed():
    # deleting an old .bak is fine; only the live db is protected
    assert _cmd_denied("rm data/metadata.db.bak.20260101") is None


def test_benign_command_allowed():
    assert _cmd_denied("python scripts/analyze_library.py --stages publish") is None


def test_redirect_to_unprotected_path_allowed():
    assert _cmd_denied("echo hi > data/notes.txt") is None


# --------------------------- MERT archive ---------------------------

def test_rm_archive_denied():
    assert _cmd_denied("rm -rf data/archive/mert_2026/") is not None


def test_redirect_into_archive_denied():
    assert _cmd_denied("echo x > data/archive/mert_2026/manifest.json") is not None


def test_read_archive_manifest_allowed():
    assert _cmd_denied("cat data/archive/mert_2026/manifest.json") is None


# --------------------------- audio library (monkeypatched root) ---------------------------

def test_delete_audio_denied(monkeypatch):
    monkeypatch.setattr(hook, "_audio_root", lambda: "e:/music")
    assert _cmd_denied(r"Remove-Item E:\MUSIC\artist\track.flac") is not None


def test_redirect_into_audio_denied(monkeypatch):
    monkeypatch.setattr(hook, "_audio_root", lambda: "e:/music")
    assert _cmd_denied("echo x > E:/music/note.txt") is not None


def test_read_audio_allowed(monkeypatch):
    monkeypatch.setattr(hook, "_audio_root", lambda: "e:/music")
    assert _cmd_denied(r"ffprobe E:\MUSIC\artist\track.flac") is None


def test_audio_root_off_when_unconfigured(monkeypatch):
    # no config -> audio guard silent, but db guard still active
    monkeypatch.setattr(hook, "_audio_root", lambda: None)
    assert _cmd_denied("rm E:/music/x.flac") is None
    assert _cmd_denied("rm data/metadata.db") is not None


# --------------------------- Edit/Write tool targets ---------------------------

def test_edit_tool_on_db_denied():
    assert hook.path_denied("data/metadata.db") is not None


def test_write_tool_on_archive_denied():
    assert hook.path_denied("data/archive/mert_2026/manifest.json") is not None


def test_edit_tool_on_source_allowed():
    assert hook.path_denied("src/playlist/pier_bridge_builder.py") is None


def test_edit_tool_on_db_backup_allowed():
    assert hook.path_denied("data/metadata.db.bak.20260706") is None


# --------------------------- end-to-end stdin -> stdout ---------------------------

def _run(payload):
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=15,
    )
    return proc.stdout.strip()


def test_e2e_deny_emits_permission_decision():
    out = _run({"tool_name": "Bash", "tool_input": {"command": "rm data/metadata.db"}})
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_e2e_allow_is_silent():
    out = _run(
        {"tool_name": "Bash", "tool_input": {"command": "sqlite3 data/metadata.db \"SELECT 1\""}}
    )
    assert out == ""


def test_e2e_copy_deny_points_to_safe_tool():
    out = _run({"tool_name": "PowerShell", "tool_input": {"command": r"Copy-Item data\metadata.db D:\x.db"}})
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "copy_db_safe" in parsed["hookSpecificOutput"]["permissionDecisionReason"]
