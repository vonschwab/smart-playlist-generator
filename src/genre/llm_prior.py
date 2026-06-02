"""
LLM-backed genre embedding prior — Phase 2 of the genre subsystem redesign.

For genres with low corpus support, the PMI-SVD embedding is unreliable
(sparse co-occurrence statistics).  This module builds a prior embedding
by asking the LLM to rate each genre's similarity to a fixed set of
"anchor genres" that are well-represented in the corpus.  The prior for
genre g is then the support-weighted average of anchor embeddings, placing
rare genres in the *same coordinate space* as corpus-trained embeddings.

Public API:
    build_llm_prior(vocab, corpus_emb, support, llm_client, cache_path, *, n_anchors, batch_size)
    -> (V, dim) float32, L2-normalized
"""

from __future__ import annotations

import json
import logging
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PRIOR_PROMPT_TEMPLATE = """\
You are a music genre taxonomy expert.

I will give you a list of QUERY genres and a fixed list of ANCHOR genres.
For each query genre, rate its musical similarity to EACH anchor (0-10 scale,
where 10 = effectively the same genre and 0 = completely unrelated).

Return a JSON object with exactly one field, "items", whose value is an array
with one object per query genre, in the same order as the input.
Each item must have exactly these fields:
  "genre": the query genre string (exactly as given)
  "scores": list of floats, one per anchor in the order given

Anchor genres (in order):
{anchors_json}

Query genres (in order):
{queries_json}

Example response shape:
{{"items": [{{"genre": "example", "scores": [1.0, 2.0]}}]}}
"""


def _make_prior_prompt(queries: list[str], anchors: list[str]) -> str:
    return _PRIOR_PROMPT_TEMPLATE.format(
        anchors_json=json.dumps(anchors, indent=2),
        queries_json=json.dumps(queries, indent=2),
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def select_anchors(vocab: list[str], support: np.ndarray, n_anchors: int) -> list[str]:
    """Return the top-n_anchors genre names by support count."""
    idx = np.argsort(support)[::-1][:n_anchors]
    return [vocab[i] for i in idx]


def build_llm_prior(
    vocab: list[str],
    corpus_emb: np.ndarray,
    support: np.ndarray,
    llm_client: LLMClient,
    cache_path: Path | str | None = None,
    *,
    n_anchors: int = 64,
    batch_size: int = 10,
) -> np.ndarray:
    """
    Build a prior embedding for every genre in vocab.

    For each genre, the LLM rates its similarity to `n_anchors` well-supported
    anchor genres.  The prior position is the similarity-weighted average of
    those anchor vectors in the corpus embedding space.

    Genres that are anchors themselves get their own corpus embedding as the
    prior (identity mapping, α→0 in blend anyway for high-support genres).

    Args:
        vocab:       V genre strings (same order as corpus_emb rows)
        corpus_emb:  (V, dim) float32 — trained PMI-SVD embedding
        support:     (V,) int — number of tracks with each genre
        llm_client:  LLMClient instance
        cache_path:  optional JSON cache file; read+write
        n_anchors:   how many anchor genres to use
        batch_size:  genres per LLM call

    Returns:
        (V, dim) float32, L2-normalized
    """
    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)
    V, dim = corpus_emb.shape

    # --- Anchor selection ---
    anchors = select_anchors(vocab, np.asarray(support), min(n_anchors, V))
    anchor_set = set(anchors)
    anchor_idx = {g: i for i, g in enumerate(vocab)}

    # --- Load cache ---
    cache: dict[str, list[float]] = {}  # genre -> scores list
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text())
            if (
                isinstance(loaded, dict)
                and loaded.get("version") == 1
                and loaded.get("anchors") == anchors
                and isinstance(loaded.get("scores"), dict)
            ):
                cache = loaded["scores"]
                logger.info("LLM prior: loaded %d cached entries from %s", len(cache), cache_path)
            else:
                logger.warning("LLM prior: invalidating cache with missing or changed anchors")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load LLM prior cache: %s", exc)

    # --- Determine which genres need LLM scoring ---
    pending = [
        g for g in vocab
        if g not in anchor_set and _coerce_scores(cache.get(g), len(anchors)) is None
    ]
    if pending:
        logger.info(
            "LLM prior: %d genres to score (%d cached, %d total)",
            len(pending), len(cache), V,
        )
        _run_batches(pending, anchors, llm_client, cache, batch_size=batch_size)
        if cache_path:
            _save_cache(cache, anchors, cache_path)

    # --- Build prior matrix ---
    # Get anchor row indices and their corpus vectors
    anchor_corpus_vecs = np.stack(
        [corpus_emb[anchor_idx[a]] for a in anchors], axis=0
    )  # (n_anchors, dim)

    prior = np.zeros((V, dim), dtype=np.float32)
    missing = 0
    for t, g in enumerate(vocab):
        if g in anchor_set:
            prior[t] = corpus_emb[t]
            continue

        scores_raw = _coerce_scores(cache.get(g), len(anchors))
        if scores_raw is None:
            # Fallback: use own corpus vector
            prior[t] = corpus_emb[t]
            missing += 1
            continue

        scores = np.array(scores_raw, dtype=np.float32)
        scores = np.clip(scores, 0.0, None)  # ensure non-negative
        total = scores.sum()
        if total < 1e-6:
            prior[t] = corpus_emb[t]
        else:
            prior[t] = (scores @ anchor_corpus_vecs) / total

    if missing:
        logger.warning("LLM prior: %d genres fell back to corpus embedding (missing scores)", missing)

    # L2-normalize
    norms = np.linalg.norm(prior, axis=1, keepdims=True)
    prior = prior / np.maximum(norms, 1e-12)

    logger.info("build_llm_prior: done — shape %s", prior.shape)
    return prior


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_batches(
    genres: list[str],
    anchors: list[str],
    llm_client: LLMClient,
    cache: dict[str, list[float]],
    batch_size: int,
) -> None:
    n_anchors = len(anchors)
    for start in range(0, len(genres), batch_size):
        batch = genres[start: start + batch_size]
        prompt = _make_prior_prompt(batch, anchors)

        try:
            response = llm_client.complete_json(prompt)
        except RuntimeError as exc:
            logger.error("LLM prior batch failed: %s", exc)
            continue

        # Dry-run response
        if isinstance(response, dict) and response.get("_dry_run"):
            continue

        items = response if isinstance(response, list) else (
            response.get("items", []) if isinstance(response, dict) else []
        )
        if not isinstance(items, list):
            logger.warning("LLM prior: unexpected response type %s", type(items))
            continue

        for i, g in enumerate(batch):
            if i >= len(items):
                logger.warning("LLM prior: short response (wanted %d, got %d)", len(batch), len(items))
                continue

            item = items[i]
            if not isinstance(item, dict):
                logger.warning("LLM prior: malformed item for genre %r", g)
                continue
            if item.get("genre") != g:
                logger.warning(
                    "LLM prior: expected genre %r, got %r",
                    g, item.get("genre"),
                )
                continue
            scores = _coerce_scores(item.get("scores"), n_anchors)
            if scores is None:
                logger.warning(
                    "LLM prior: genre %r did not return %d numeric scores",
                    g, n_anchors,
                )
                continue

            cache[g] = scores

        logger.debug(
            "LLM prior: scored batch [%d–%d] / %d",
            start, min(start + batch_size, len(genres)) - 1, len(genres),
        )


def _coerce_scores(scores: Any, n_anchors: int) -> list[float] | None:
    if not isinstance(scores, list) or len(scores) != n_anchors:
        return None
    if not all(isinstance(score, Real) and not isinstance(score, bool) for score in scores):
        return None
    result = [float(score) for score in scores]
    if not all(np.isfinite(score) and 0.0 <= score <= 10.0 for score in result):
        return None
    return result


def _save_cache(cache: dict[str, list[float]], anchors: list[str], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "anchors": anchors, "scores": cache}
        path.write_text(json.dumps(payload, indent=2))
        logger.info("LLM prior: cache saved (%d entries) → %s", len(cache), path)
    except OSError as exc:
        logger.warning("Could not save LLM prior cache: %s", exc)
