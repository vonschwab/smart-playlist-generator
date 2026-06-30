"""generation_budget_s: config.yaml -> PierBridgeConfig plumbing.

Regression guard for the unwired-knob bug (2026-06-30): the dataclass default
was 60.0, the consumer read pb_cfg.generation_budget_s, and a comment claimed it
came from config.yaml — but NOTHING read it onto the typed config, so the beam's
soft deadline was permanently 60s regardless of config, cutting generations off
~30s before the 90s hard ceiling (segments bailed to fallback placement).
Wired through apply_pier_bridge_overrides alongside the variable_bridge_* keys.
"""
from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides


def _apply(overrides: dict) -> PierBridgeConfig:
    pb_cfg, _tuning, _sources, _weights = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides=overrides,
        pb_overrides=overrides.get("pier_bridge", {}),
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
        resolved_variant="raw",
    )
    return pb_cfg


def test_generation_budget_s_propagates_from_pier_bridge_overrides():
    cfg = _apply({"pier_bridge": {"generation_budget_s": 85}})
    assert cfg.generation_budget_s == 85.0


def test_generation_budget_s_accepts_float():
    cfg = _apply({"pier_bridge": {"generation_budget_s": 82.5}})
    assert cfg.generation_budget_s == 82.5


def test_generation_budget_s_absent_keeps_dataclass_default():
    cfg = _apply({"pier_bridge": {}})
    assert cfg.generation_budget_s == 60.0
