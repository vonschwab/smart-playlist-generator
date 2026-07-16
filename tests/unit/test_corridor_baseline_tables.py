import gzip
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_tables", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "capture_transforms.py")
tables = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tables)


def test_write_table_is_byte_reproducible(tmp_path):
    obj = {"b": [2, 1], "a": {"y": 1, "x": 2}}
    m1 = tables.write_table("t1", obj, out_dir=tmp_path)
    m2 = tables.write_table("t1", obj, out_dir=tmp_path)
    assert m1["sha256"] == m2["sha256"]          # same content -> same hash (gzip mtime pinned)
    with gzip.open(tmp_path / "t1.json.gz", "rt", encoding="utf-8") as f:
        assert json.load(f) == obj


def test_write_table_hash_changes_with_content(tmp_path):
    h1 = tables.write_table("t2", {"a": 1}, out_dir=tmp_path)["sha256"]
    h2 = tables.write_table("t2", {"a": 2}, out_dir=tmp_path)["sha256"]
    assert h1 != h2
