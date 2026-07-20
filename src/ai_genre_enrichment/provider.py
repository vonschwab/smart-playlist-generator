"""Provider selection for enrichment LLM clients.

Resolution order: explicit ``provider=`` arg -> PG_AI_PROVIDER env var (test
seam) -> config.yaml ``ai_genre.provider`` -> 'claude_code'.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .claude_client import ClaudeCodeEnrichmentClient
from .client import OpenAIEnrichmentClient
from .routing import WebMode

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
CLAUDE_DEFAULT_MODEL = "haiku"
KNOWN_PROVIDERS = ("claude_code", "openai", "anthropic_api", "zero_touch", "skip")


@lru_cache(maxsize=8)
def _config_ai_genre(config_path: str | None) -> tuple[str, str]:
    """(provider, claude_model) from config.yaml; safe defaults if unreadable.

    Reads only the ``ai_genre`` section directly from YAML to avoid triggering
    Config._validate_config (which requires ``library.database_path`` and is
    irrelevant for the enrichment factory).
    """
    try:
        import yaml

        path = config_path or "config.yaml"
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        section = raw.get("ai_genre") or {}
        provider = str(section.get("provider") or "claude_code")
        claude_model = str(section.get("claude_model") or CLAUDE_DEFAULT_MODEL)
        return provider, claude_model
    except (FileNotFoundError, OSError, KeyError, AttributeError):
        return "claude_code", CLAUDE_DEFAULT_MODEL


def get_enrichment_provider(config_path: str | None = None) -> str:
    provider = os.environ.get("PG_AI_PROVIDER") or _config_ai_genre(config_path)[0]
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown ai_genre.provider {provider!r}; expected one of {KNOWN_PROVIDERS}"
        )
    return provider


def resolve_enrichment_model(
    model: str | None = None, *, config_path: str | None = None
) -> str:
    """Explicit model wins; otherwise the active provider's default."""
    if model:
        return model
    if get_enrichment_provider(config_path) in ("claude_code", "anthropic_api"):
        return _config_ai_genre(config_path)[1]
    return OPENAI_DEFAULT_MODEL


def create_enrichment_client(
    *,
    model: str | None = None,
    dry_run: bool = False,
    web_mode: WebMode | str = WebMode.OFF,
    allowed_web_domains: list[str] | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    config_path: str | None = None,
) -> Any:
    if provider is None:
        provider = get_enrichment_provider(config_path)
    elif provider not in KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown ai_genre.provider {provider!r}; expected one of {KNOWN_PROVIDERS}"
        )
    if provider in ("skip", "zero_touch"):
        raise ValueError(
            f"provider '{provider}' has no enrichment client — "
            "stages must skip instead of constructing one"
        )
    if provider == "anthropic_api" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "anthropic_api provider requires ANTHROPIC_API_KEY in the environment"
        )
    if provider == "openai":
        return OpenAIEnrichmentClient(
            model=model or OPENAI_DEFAULT_MODEL,
            dry_run=dry_run,
            web_mode=web_mode,
            allowed_web_domains=allowed_web_domains,
            api_key=api_key,
            max_retries=max_retries,
        )
    # claude_code and anthropic_api share one construction path: both build a
    # ClaudeCodeEnrichmentClient (the Claude Code CLI authenticates via the
    # ANTHROPIC_API_KEY env var for anthropic_api — see class docstring re:
    # SDK/CLI auth), and both are equally bound by the WebMode guard below —
    # neither supports web search, so one guarded path keeps that in sync.
    if WebMode(web_mode) != WebMode.OFF:
        raise ValueError(
            "ai_genre.provider=claude_code does not support web search; "
            "this call requires web_mode='off' (Bandcamp locator is OpenAI-only and unwired)"
        )
    return ClaudeCodeEnrichmentClient(
        model=model or _config_ai_genre(config_path)[1],
        dry_run=dry_run,
        max_retries=max_retries,
    )
