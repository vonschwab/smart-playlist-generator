"""Provider registry: new values accepted, default unchanged, skip semantics."""
import pytest

from src.ai_genre_enrichment.provider import KNOWN_PROVIDERS, get_enrichment_provider


def test_new_values_known():
    for v in ("claude_code", "openai", "anthropic_api", "zero_touch", "skip"):
        assert v in KNOWN_PROVIDERS


def test_default_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("PG_AI_PROVIDER", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("library: {}\n", encoding="utf-8")  # no ai_genre section — Dylan's shape
    assert get_enrichment_provider(str(cfg)) == "claude_code"


def test_env_override_accepts_skip(monkeypatch):
    monkeypatch.setenv("PG_AI_PROVIDER", "skip")
    assert get_enrichment_provider(None) == "skip"


def test_factory_refuses_client_for_non_llm_providers():
    from src.ai_genre_enrichment.provider import create_enrichment_client
    for v in ("skip", "zero_touch"):
        with pytest.raises(ValueError, match="no enrichment client"):
            create_enrichment_client(provider=v)


def test_stage_adjudicate_skips_for_skip_provider(tmp_path, monkeypatch):
    """stage_adjudicate short-circuits on provider=skip without constructing a client."""
    monkeypatch.setenv("PG_AI_PROVIDER", "skip")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("library: {}\n", encoding="utf-8")

    from scripts.analyze_library import stage_adjudicate
    from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient

    def _boom(*args, **kwargs):
        raise AssertionError("ClaudeCodeEnrichmentClient must not be constructed when provider=skip")

    monkeypatch.setattr(ClaudeCodeEnrichmentClient, "__init__", _boom)

    ctx = {"config_path": str(cfg)}
    result = stage_adjudicate(ctx)
    assert result == {"skipped": True, "reason": "adjudication skipped (ai_genre.provider=skip)"}
