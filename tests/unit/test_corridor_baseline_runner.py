import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_runner", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "runner.py")
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_deep_set_creates_nested_path():
    d = {}
    runner.deep_set(d, "playlists.ds_pipeline.pier_bridge.transition_floor", 0.4)
    assert d == {"playlists": {"ds_pipeline": {"pier_bridge": {"transition_floor": 0.4}}}}


def test_deep_set_overwrites_leaf_preserving_siblings():
    d = {"a": {"b": 1, "c": 2}}
    runner.deep_set(d, "a.b", 9)
    assert d == {"a": {"b": 9, "c": 2}}


def test_deep_get_returns_default_on_missing():
    assert runner.deep_get({"a": {}}, "a.b.c", default="missing") == "missing"


def test_parse_ds_success_extracts_effective_and_metrics():
    log = ('noise\n{"pipeline": "ds", "mode": "dynamic", "metrics": {"min_transition": 0.5},'
           ' "effective": {"candidate_pool": {"similarity_floor": 0.3}}}\nmore noise\n')
    eff, met = runner.parse_ds_success(log)
    assert eff["candidate_pool"]["similarity_floor"] == 0.3
    assert met["min_transition"] == 0.5


def test_parse_ds_success_returns_none_when_absent():
    assert runner.parse_ds_success("no json here\n") == (None, None)
