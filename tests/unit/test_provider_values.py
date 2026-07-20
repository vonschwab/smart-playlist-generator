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


def test_stage_adjudicate_skips_for_zero_touch_provider(tmp_path, monkeypatch):
    """stage_adjudicate short-circuits on provider=zero_touch without constructing a client."""
    monkeypatch.setenv("PG_AI_PROVIDER", "zero_touch")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("library: {}\n", encoding="utf-8")

    from scripts.analyze_library import stage_adjudicate
    from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient

    def _boom(*args, **kwargs):
        raise AssertionError("ClaudeCodeEnrichmentClient must not be constructed when provider=zero_touch")

    monkeypatch.setattr(ClaudeCodeEnrichmentClient, "__init__", _boom)

    ctx = {"config_path": str(cfg)}
    result = stage_adjudicate(ctx)
    assert result == {
        "skipped": True,
        "reason": "zero-touch deterministic fusion handles genres — LLM adjudication skipped",
    }


def test_stage_apply_skips_for_skip_provider(tmp_path, monkeypatch):
    """stage_apply short-circuits on provider=skip without constructing a client (mirrors
    the stage_adjudicate skip test — task 7a added identical gates to both stages)."""
    monkeypatch.setenv("PG_AI_PROVIDER", "skip")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("library: {}\n", encoding="utf-8")

    from scripts.analyze_library import stage_apply
    from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient

    def _boom(*args, **kwargs):
        raise AssertionError("ClaudeCodeEnrichmentClient must not be constructed when provider=skip")

    monkeypatch.setattr(ClaudeCodeEnrichmentClient, "__init__", _boom)

    ctx = {"config_path": str(cfg)}
    result = stage_apply(ctx)
    assert result == {"skipped": True, "reason": "adjudication skipped (ai_genre.provider=skip)"}


def test_stage_apply_skips_for_zero_touch_provider(tmp_path, monkeypatch):
    """stage_apply short-circuits on provider=zero_touch without constructing a client."""
    monkeypatch.setenv("PG_AI_PROVIDER", "zero_touch")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("library: {}\n", encoding="utf-8")

    from scripts.analyze_library import stage_apply
    from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient

    def _boom(*args, **kwargs):
        raise AssertionError("ClaudeCodeEnrichmentClient must not be constructed when provider=zero_touch")

    monkeypatch.setattr(ClaudeCodeEnrichmentClient, "__init__", _boom)

    ctx = {"config_path": str(cfg)}
    result = stage_apply(ctx)
    assert result == {
        "skipped": True,
        "reason": "zero-touch deterministic fusion handles genres — LLM adjudication skipped",
    }


def test_anthropic_api_enforces_web_mode_guard(monkeypatch):
    """anthropic_api must reject web_mode != OFF just like claude_code does (Finding 1:
    the anthropic_api branch previously skipped this guard and silently accepted web
    search on a client that doesn't support it)."""
    from src.ai_genre_enrichment.provider import create_enrichment_client
    from src.ai_genre_enrichment.routing import WebMode

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with pytest.raises(ValueError, match="does not support web search"):
        create_enrichment_client(provider="anthropic_api", web_mode=WebMode.AUTO)
    with pytest.raises(ValueError, match="does not support web search"):
        create_enrichment_client(provider="anthropic_api", web_mode="auto")


def test_anthropic_api_web_mode_off_still_requires_key(monkeypatch):
    """Sanity check that the pre-existing ANTHROPIC_API_KEY gate still fires independently
    of the WebMode guard (order: missing key raises before the WebMode check ever runs)."""
    from src.ai_genre_enrichment.provider import create_enrichment_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        create_enrichment_client(provider="anthropic_api", web_mode="off")
