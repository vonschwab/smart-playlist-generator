"""Structured contract for AI adjudication of deterministic source tags."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


TAG_ADJUDICATOR_INSTRUCTIONS = """
Classify source-provided release tags for a local music library genre authority layer.

Use the supplied tags and local payload only. Do not perform web search in this step.
The goal is to preserve valuable narrow genre/style signals when source-backed.
Niche, narrow, underground, regional, or scene specificity is not a reason to demote
a supported tag. Do not collapse tags such as fourth world, ambient jazz,
electroacoustic, or electronica into only electronic, experimental, jazz, rock,
indie, or other broad parent genres.

Separate genres/styles from descriptors, instruments, places, formats, and
mood/function tags. For example, saxophone is an instrument, Oakland is a place,
and meditation is usually a mood/function tag unless a source clearly presents it
as a recognized genre/style term.

Return strict JSON only with concise reasons. Do not provide chain-of-thought.
""".strip()


_TAG_ADJUDICATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tag_classifications", "warnings"],
    "properties": {
        "tag_classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["raw_tag", "normalized_tag", "classification", "confidence", "reason"],
                "properties": {
                    "raw_tag": {"type": "string"},
                    "normalized_tag": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": [
                            "genre_style",
                            "descriptor",
                            "instrument",
                            "place",
                            "format",
                            "mood_function",
                            "review_only",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def tag_adjudicator_response_format() -> dict[str, Any]:
    """Return a strict JSON schema for source-tag adjudication calls."""
    return {
        "type": "json_schema",
        "name": "ai_genre_tag_adjudicator",
        "schema": deepcopy(_TAG_ADJUDICATOR_SCHEMA),
        "strict": True,
    }


def adjudicate_tags(
    tags: list[tuple[str, str]],
    *,
    model: str | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Call AI to classify a batch of unknown tags.

    Args:
        tags: List of (raw_tag, normalized_tag) pairs to classify.
        model: Model override; None uses the active provider's default.
        dry_run: If True, return empty results without calling the backend.
        client: Injected enrichment client (tests); None builds one via the factory.

    Returns:
        Dict keyed by normalized_tag -> {"classification", "confidence", "reason"}.

    Raises:
        RuntimeError/ValueError: backend unavailable or call failed after retries.
        An explicitly requested adjudication that cannot run is an error, not a
        silent no-op.

    Note: No hard batch-size limit is enforced here. At scale, callers should
    chunk large batches to avoid token-limit failures (the analyze 'enrich'
    stage uses call_structured_batch for that).
    """
    if not tags or dry_run:
        return {}

    if client is None:
        from .provider import create_enrichment_client

        client = create_enrichment_client(model=model)

    tag_list = "\n".join(f"- raw: {raw!r}, normalized: {norm!r}" for raw, norm in tags)
    prompt = f"Classify the following source tags:\n\n{tag_list}"

    data = client.call_structured(
        prompt,
        tag_adjudicator_response_format(),
        instructions=TAG_ADJUDICATOR_INSTRUCTIONS,
    )

    results: dict[str, dict[str, Any]] = {}
    for item in data.get("tag_classifications", []):
        norm = item.get("normalized_tag", "").strip().casefold()
        if norm:
            results[norm] = {
                "classification": item["classification"],
                "confidence": item["confidence"],
                "reason": item.get("reason", ""),
            }

    usage = getattr(client, "last_token_usage", None) or {}
    logger.info(
        "AI adjudication: %d tags, %d input + %d output tokens",
        len(tags),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return results
