"""Tests for the Claude-Code enrichment client, provider factory, and config plumbing."""
from __future__ import annotations

from src.config_loader import Config


def _write_config(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_config_ai_genre_defaults(tmp_path):
    cfg = Config(_write_config(tmp_path, "library:\n  database_path: data/metadata.db\n"))
    assert cfg.ai_genre_provider == "claude_code"
    assert cfg.ai_genre_claude_model == "haiku"


def test_config_ai_genre_explicit(tmp_path):
    cfg = Config(_write_config(
        tmp_path,
        "library:\n  database_path: data/metadata.db\nai_genre:\n  provider: openai\n  claude_model: sonnet\n",
    ))
    assert cfg.ai_genre_provider == "openai"
    assert cfg.ai_genre_claude_model == "sonnet"
