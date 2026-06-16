"""Phase-2 scoring for the album adjudicator (metrics.md protocol).

Pure set metrics comparing a proposed genre set to gold, plus the file-tag
preservation floor metric and distribution aggregation (min/p10/p50/p90 — never
just means; the worst release defines trust).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .tag_classification import normalize_source_tag


def match_keys(terms: Iterable[str], canonicalize_fn: Callable[[str], Any]) -> set[str]:
    """Map terms to comparison keys: canonical name where the taxonomy resolves it,
    else a normalized fallback. Lets gold and proposed match on canonical-equivalence
    (soul-jazz == soul jazz) while still matching gap terms (ethio-jazz) by string.
    """
    keys: set[str] = set()
    for term in terms:
        result = canonicalize_fn(term)
        name = getattr(result, "canonical", None)
        if getattr(result, "resolution", None) in ("canonical", "alias") and name:
            keys.add(name)
        else:
            keys.add(normalize_source_tag(str(term)))
    return keys


def set_metrics(proposed: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    p, g = set(proposed), set(gold)
    correct = p & g
    n_p, n_g, n_c = len(p), len(g), len(correct)
    precision = n_c / n_p if n_p else 0.0
    recall = n_c / n_g if n_g else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    noise_rate = (n_p - n_c) / n_p if n_p else 0.0
    return {
        "n_proposed": n_p, "n_gold": n_g, "n_correct": n_c,
        "precision": precision, "recall": recall, "f1": f1, "noise_rate": noise_rate,
    }


def preservation(proposed: Iterable[str], must_preserve: Iterable[str]) -> float:
    must = set(must_preserve)
    if not must:
        return 1.0
    return len(set(proposed) & must) / len(must)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }
