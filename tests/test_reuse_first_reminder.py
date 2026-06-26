"""Unit tests for the reuse-first PreToolUse advisory hook (build_message logic)."""

import importlib.util
import pathlib

_HOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "reuse_first_reminder.py"
)
_spec = importlib.util.spec_from_file_location("reuse_first_reminder", _HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _write(path, content):
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


def _edit(path, old, new):
    return {
        "tool_name": "Edit",
        "tool_input": {"file_path": path, "old_string": old, "new_string": new},
    }


def test_write_new_source_file_fires_general():
    data = _write("C:/repo/src/playlist/new_feature.py", "x = 1\n" * 100)
    result = hook.build_message(data)
    assert result is not None and result[0] == "general"


def test_trivial_edit_is_silent():
    data = _edit("C:/repo/src/playlist/pipeline.py", "x = 1", "x = 2")
    assert hook.build_message(data) is None


def test_edit_adding_a_def_fires_even_if_small():
    data = _edit(
        "C:/repo/src/playlist/pipeline.py",
        "y = 1",
        "y = 1\ndef helper():\n    return 1\n",
    )
    result = hook.build_message(data)
    assert result is not None and result[0] == "general"


def test_pyproject_edit_fires_dependency():
    data = _edit("C:/repo/pyproject.toml", "deps = []", 'deps = ["requests"]')
    result = hook.build_message(data)
    assert result is not None and result[0] == "dependency"


def test_web_package_json_fires_dependency():
    data = _edit("C:/repo/web/package.json", "{}", '{"x": 1}')
    assert hook.build_message(data)[0] == "dependency"


def test_markdown_file_is_silent():
    data = _write("C:/repo/docs/notes.md", "lots of text\n" * 100)
    assert hook.build_message(data) is None


def test_test_file_is_silent():
    data = _write("C:/repo/tests/test_thing.py", "def t():\n    assert True\n" * 50)
    assert hook.build_message(data) is None
