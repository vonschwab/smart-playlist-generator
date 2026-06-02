"""
Genre vocab normalization — Phase 1 of the genre subsystem redesign.

Takes the raw set of genre tokens in the corpus and identifies spelling/
hyphenation variants that should be merged to a single canonical form.
Optionally uses an LLM to confirm whether near-duplicate pairs are truly
the same genre (vs merely similar-looking strings, e.g. "funk rock" vs
"punk rock").

Entry point:
    decisions = normalize_vocab(tokens, client, cache_path=...)

Each CanonicalizationDecision records which original tokens map to which
canonical form.  Decisions are written to `genre_canonical_token.canonical_form`
by the CLI script (scripts/normalize_genre_vocab.py).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CanonicalizationDecision:
    """One resolved cluster of tokens → single canonical form."""
    tokens: list[str]        # all tokens in this cluster (≥2)
    canonical: str           # the chosen canonical form
    confidence: float        # 0–1
    is_same_genre: bool      # LLM confirmed these are the same genre
    reasoning: str           # LLM explanation or rule description
    source: str              # "rule" | "llm" | "dry_run"


# ---------------------------------------------------------------------------
# Vocab collection
# ---------------------------------------------------------------------------

def collect_raw_vocab(db_conn: sqlite3.Connection) -> list[str]:
    """
    Read all genre tokens present in the corpus.

    Sources:
      - metadata.db: track_genres.genre, album_genres.genre, artist_genres.genre
      - ai_genre_enrichment.db: enriched_genres.genre (if attached)

    Returns a deduplicated sorted list.
    """
    tokens: set[str] = set()

    for table, col in [
        ("track_genres", "genre"),
        ("album_genres", "genre"),
        ("artist_genres", "genre"),
    ]:
        try:
            rows = db_conn.execute(f"SELECT DISTINCT {col} FROM {table}").fetchall()
            tokens.update(r[0] for r in rows if r[0] and r[0].strip())
        except sqlite3.OperationalError as exc:
            logger.warning("Could not read %s.%s: %s", table, col, exc)

    # enriched genres from sidecar DB (attached as 'enrichment' or direct path)
    try:
        rows = db_conn.execute("SELECT DISTINCT genre FROM enriched_genres").fetchall()
        tokens.update(r[0] for r in rows if r[0] and r[0].strip())
    except sqlite3.OperationalError:
        pass  # sidecar not attached — OK

    return sorted(tokens)


# ---------------------------------------------------------------------------
# Structural clustering (rule-based, no LLM)
# ---------------------------------------------------------------------------

def _strip_punctuation(s: str) -> str:
    """Lowercase, remove hyphens, spaces, apostrophes for form comparison."""
    return re.sub(r"[-\s'.]", "", s.lower())


def cluster_by_form(tokens: list[str]) -> list[list[str]]:
    """
    Group tokens that are identical after stripping hyphens and spaces.

    e.g. ["trip hop", "trip-hop"] → [["trip hop", "trip-hop"]]

    These are guaranteed spelling variants — no LLM confirmation needed.
    Returns only clusters of size ≥ 2.
    """
    groups: dict[str, list[str]] = {}
    for tok in tokens:
        key = _strip_punctuation(tok)
        groups.setdefault(key, []).append(tok)

    return [sorted(members) for members in groups.values() if len(members) >= 2]


def cluster_by_similarity(
    tokens: list[str],
    *,
    ratio_threshold: float = 0.88,
    max_length_delta: int = 4,
) -> list[list[str]]:
    """
    Find pairs of tokens with high difflib ratio that are NOT already
    captured by cluster_by_form (i.e. they differ by more than punctuation).

    Returns only pairs (clusters of size 2) — larger merges are handled
    by the LLM step which can distinguish false positives like
    "funk rock" / "punk rock".

    Pairs that are already grouped by cluster_by_form are excluded.
    """
    form_clusters = cluster_by_form(tokens)
    already_merged: set[str] = set()
    for cluster in form_clusters:
        key = _strip_punctuation(cluster[0])
        for tok in cluster:
            already_merged.add(tok)

    candidates = [t for t in tokens if t not in already_merged]
    pairs: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if abs(len(a) - len(b)) > max_length_delta:
                continue
            key = frozenset({a, b})
            if key in seen:
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= ratio_threshold:
                pairs.append(sorted([a, b]))
                seen.add(key)

    return pairs


# ---------------------------------------------------------------------------
# LLM-assisted classification
# ---------------------------------------------------------------------------

_BATCH_PROMPT_TEMPLATE = """\
You are a music genre taxonomy expert. I will give you a list of token pairs.
For each pair, decide whether the two tokens refer to the SAME music genre
(possibly with different spellings/hyphenation) or to DIFFERENT genres.
Return a JSON object with exactly one field, "items", whose value is an array
with one object per pair, in the same order as the input.

Each object must have exactly these fields:
  "same_genre": true or false
  "canonical": the preferred spelling if same_genre is true, otherwise null
  "confidence": a number from 0.0 to 1.0
  "reasoning": one short sentence

Input pairs (0-indexed):
{pairs_json}

Example response shape:
{{"items": [{{"same_genre": true, "canonical": "example", "confidence": 0.9, "reasoning": "Same spelling variant."}}]}}
"""


def _fallback_decision(
    cluster: list[str],
    reasoning: str,
    *,
    source: str = "llm",
) -> CanonicalizationDecision:
    return CanonicalizationDecision(
        tokens=cluster,
        canonical=cluster[0],
        confidence=0.0,
        is_same_genre=False,
        reasoning=reasoning,
        source=source,
    )


def _coerce_cached_decision(
    cluster: list[str],
    cached: Any,
) -> CanonicalizationDecision | None:
    if not isinstance(cached, dict):
        return None
    try:
        decision = CanonicalizationDecision(**cached)
    except (TypeError, ValueError):
        return None
    if (
        decision.tokens != cluster
        or not isinstance(decision.canonical, str)
        or decision.canonical not in cluster
        or not isinstance(decision.confidence, (int, float))
        or isinstance(decision.confidence, bool)
        or not math.isfinite(float(decision.confidence))
        or not 0.0 <= float(decision.confidence) <= 1.0
        or not isinstance(decision.is_same_genre, bool)
        or not isinstance(decision.reasoning, str)
        or decision.source != "llm"
    ):
        return None
    return decision


def _coerce_response_decision(
    cluster: list[str],
    item: Any,
    *,
    source: str,
) -> CanonicalizationDecision | None:
    if not isinstance(item, dict):
        return None
    is_same = item.get("same_genre")
    canonical_raw = item.get("canonical")
    confidence_raw = item.get("confidence")
    reasoning = item.get("reasoning")
    if (
        not isinstance(is_same, bool)
        or (is_same and (
            not isinstance(canonical_raw, str)
            or not canonical_raw.strip()
        ))
        or (not is_same and canonical_raw is not None)
        or not isinstance(confidence_raw, (int, float))
        or isinstance(confidence_raw, bool)
        or not math.isfinite(float(confidence_raw))
        or not 0.0 <= float(confidence_raw) <= 1.0
        or not isinstance(reasoning, str)
    ):
        return None

    canonical = cluster[0]
    if is_same:
        canonical = canonical_raw
        if canonical not in cluster:
            matches = [
                token
                for token in cluster
                if _strip_punctuation(token) == _strip_punctuation(canonical)
            ]
            canonical = matches[0] if matches else cluster[0]
            logger.debug(
                "LLM canonical %r not in cluster %s, using %r",
                canonical_raw, cluster, canonical,
            )

    return CanonicalizationDecision(
        tokens=cluster,
        canonical=canonical,
        confidence=float(confidence_raw),
        is_same_genre=is_same,
        reasoning=reasoning,
        source=source,
    )


def classify_cluster_batch(
    clusters: list[list[str]],
    client: LLMClient,
    *,
    cache: dict[str, Any] | None = None,
    batch_size: int = 20,
) -> list[CanonicalizationDecision]:
    """
    Ask the LLM to classify each cluster as same-genre or different.

    Processes in batches of `batch_size` to limit prompt size.
    `cache` maps cluster_key → CanonicalizationDecision.__dict__ for
    previously answered clusters.
    """
    if cache is None:
        cache = {}

    decisions: list[CanonicalizationDecision] = []
    pending: list[list[str]] = []
    pending_keys: list[str] = []

    for cluster in clusters:
        key = "||".join(sorted(cluster))
        if key in cache:
            decision = _coerce_cached_decision(cluster, cache[key])
            if decision is not None:
                decisions.append(decision)
                continue
            logger.warning("Ignoring malformed cached LLM response for cluster %s", cluster)
            del cache[key]
        pending.append(cluster)
        pending_keys.append(key)

    logger.info(
        "classify_cluster_batch: %d clusters total, %d cached, %d to classify",
        len(clusters),
        len(decisions),
        len(pending),
    )

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        batch_keys = pending_keys[batch_start: batch_start + batch_size]
        pairs_json = json.dumps([[c[0], c[1]] for c in batch], indent=2)
        prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)

        try:
            response = client.complete_json(prompt)
        except RuntimeError as exc:
            logger.error("LLM batch classification failed: %s", exc)
            for cluster, key in zip(batch, batch_keys):
                decisions.append(_fallback_decision(cluster, "LLM call failed"))
            continue

        # Dry-run client returns {"_dry_run": True, ...} — no real items
        if isinstance(response, dict) and response.get("_dry_run"):
            for cluster, key in zip(batch, batch_keys):
                decisions.append(
                    _fallback_decision(
                        cluster,
                        "dry-run: LLM not called",
                        source="dry_run",
                    )
                )
            continue

        items = response if isinstance(response, list) else (
            response.get("items", []) if isinstance(response, dict) else []
        )
        if not isinstance(items, list):
            items = []

        for i, (cluster, key) in enumerate(zip(batch, batch_keys)):
            should_cache = False
            if i >= len(items):
                logger.warning("LLM returned fewer items than expected (got %d)", len(items))
                dec = _fallback_decision(cluster, "LLM response missing item")
            else:
                dec = _coerce_response_decision(
                    cluster,
                    items[i],
                    source="llm",
                )
                if dec is None:
                    logger.warning("LLM returned malformed item for cluster %s", cluster)
                    dec = _fallback_decision(cluster, "LLM response malformed item")
                else:
                    should_cache = True

            decisions.append(dec)
            if should_cache:
                cache[key] = dec.__dict__

    return decisions


# ---------------------------------------------------------------------------
# Rule-based decisions (no LLM call needed)
# ---------------------------------------------------------------------------

def _pick_canonical_form(tokens: list[str]) -> str:
    """
    Heuristic for picking the canonical form from a hyphenation cluster.
    Prefer hyphenated form over spaced form for compound genres
    (matches common music taxonomy convention: "post-punk" not "post punk").
    Exception: two-word genres where hyphenation is unusual (e.g. "trip hop").
    """
    # Prefer existing hyphenated forms for most compound genres
    hyphenated = [t for t in tokens if "-" in t]
    spaced = [t for t in tokens if " " in t and "-" not in t]
    solid = [t for t in tokens if " " not in t and "-" not in t]

    if len(solid) == 1 and solid[0] in ("britpop", "bossanova", "darkwave", "chillout"):
        # solid form is the common industry spelling
        return solid[0]
    if hyphenated:
        # Prefer shorter hyphenated form
        return min(hyphenated, key=len)
    if spaced:
        return min(spaced, key=len)
    return min(tokens, key=len)


def decisions_from_form_clusters(
    clusters: list[list[str]],
) -> list[CanonicalizationDecision]:
    """
    Build rule-based decisions for structural (hyphenation) clusters.
    No LLM call.
    """
    decisions = []
    for cluster in clusters:
        canonical = _pick_canonical_form(cluster)
        decisions.append(
            CanonicalizationDecision(
                tokens=cluster,
                canonical=canonical,
                confidence=1.0,
                is_same_genre=True,
                reasoning="Identical after stripping hyphens/spaces — rule-based merge.",
                source="rule",
            )
        )
    return decisions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_vocab(
    tokens: list[str],
    client: LLMClient,
    *,
    cache_path: Path | str | None = None,
) -> list[CanonicalizationDecision]:
    """
    Full normalization pipeline for a set of genre tokens.

    1. Find structural clusters (hyphenation variants) — resolved by rule.
    2. Find similarity clusters (fuzzy near-duplicates) — resolved by LLM.
    3. Return all CanonicalizationDecision objects.

    `cache_path` points to a JSON file for caching LLM responses.
    Cache is read at start and written at end.
    """
    # Load LLM cache
    llm_cache: dict[str, Any] = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            llm_cache = json.loads(cache_path.read_text())
            if not isinstance(llm_cache, dict):
                logger.warning("Ignoring malformed LLM cache from %s", cache_path)
                llm_cache = {}
            else:
                logger.info("Loaded %d cached LLM responses from %s", len(llm_cache), cache_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load LLM cache: %s", exc)

    # Step 1: structural (rule-based)
    form_clusters = cluster_by_form(tokens)
    rule_decisions = decisions_from_form_clusters(form_clusters)
    logger.info("Rule-based: %d structural clusters found", len(form_clusters))

    # Tokens already handled by rule-based step
    handled: set[str] = {tok for d in rule_decisions for tok in d.tokens}
    remaining = [t for t in tokens if t not in handled]

    # Step 2: similarity-based (LLM-confirmed)
    sim_clusters = cluster_by_similarity(remaining)
    logger.info("Similarity-based: %d candidate pairs for LLM review", len(sim_clusters))

    llm_decisions = classify_cluster_batch(sim_clusters, client, cache=llm_cache)
    # Filter to only decisions where LLM confirmed same genre
    same_genre_decisions = [d for d in llm_decisions if d.is_same_genre]

    # Persist updated cache
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(llm_cache, indent=2))
        except OSError as exc:
            logger.warning("Could not write LLM cache: %s", exc)

    all_decisions = rule_decisions + same_genre_decisions
    logger.info(
        "Total: %d decisions (%d rule, %d llm-confirmed same-genre)",
        len(all_decisions),
        len(rule_decisions),
        len(same_genre_decisions),
    )
    return all_decisions
