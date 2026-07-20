# tests/unit/test_mixarc_paths.py
"""MixArc home resolution: cli > env > repo-config > platformdirs."""
from pathlib import Path

from src.mixarc.paths import resolve_home


def test_cli_config_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXARC_HOME", str(tmp_path / "envhome"))
    cfg = tmp_path / "custom" / "config.yaml"
    home = resolve_home(cli_config=str(cfg))
    assert home.source == "cli"
    assert home.config_path == cfg
    assert home.anchor_dir == cfg.parent


def test_env_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXARC_HOME", str(tmp_path))
    home = resolve_home()
    assert home.source == "env"
    assert home.config_path == tmp_path / "config.yaml"
    assert home.anchor_dir == tmp_path


def test_repo_config_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("MIXARC_HOME", raising=False)
    import src.mixarc.paths as paths
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / "config.yaml").write_text("library: {}\n", encoding="utf-8")
    monkeypatch.setattr(paths, "_REPO_ROOT", fake_root)
    home = resolve_home()
    assert home.source == "repo"
    assert home.anchor_dir == fake_root


def test_platformdirs_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MIXARC_HOME", raising=False)
    import src.mixarc.paths as paths
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path / "no-config-here")
    home = resolve_home()
    assert home.source == "platformdirs"
    assert home.config_path.name == "config.yaml"
    assert "mixarc" in str(home.config_path).lower()


def test_resolve_database_path_anchor(tmp_path):
    from src.config_loader import resolve_database_path
    cfg = {"library": {"database_path": "data/metadata.db"}}
    assert resolve_database_path(cfg, anchor=tmp_path) == str((tmp_path / "data" / "metadata.db").resolve())


def test_resolve_database_path_default_anchor_unchanged():
    """No-anchor call must keep resolving against the repo root (golden-critical)."""
    from src.config_loader import _REPO_ROOT, resolve_database_path
    cfg = {"library": {"database_path": "data/metadata.db"}}
    assert resolve_database_path(cfg) == str((Path(_REPO_ROOT) / "data" / "metadata.db").resolve())
