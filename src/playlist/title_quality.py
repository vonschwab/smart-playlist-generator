"""Title-artifact flag detection (pure functions, no I/O).

Used to flag tracks whose titles indicate non-canonical recordings:
live, demo, medley, remix, instrumental, remaster, alternate take, etc.

This module ONLY detects flags. Soft penalties and hard exclusions are
applied in the beam/scoring layer based on these flags plus config knobs.
"""
from __future__ import annotations

import re
from typing import Set

# Each entry maps a canonical flag to a list of case-insensitive patterns.
# Patterns use word boundaries to avoid false positives ("demolish" != "demo").
_FLAG_PATTERNS: dict[str, list[str]] = {
    "live": [
        r"\blive\s+(?:at|in|from)\b",
        r"\(live\b",
        r"-\s*live\b",
        r"\[live\b",
    ],
    "demo": [
        r"\bdemo\b",
    ],
    "medley": [
        r"\bmedley\b",
    ],
    "remix": [
        r"\bremix\b",
        r"\bmix\)\s*$",
        r"-\s*[\w\s]+\s+remix\b",
    ],
    "instrumental": [
        r"\binstrumental\b",
    ],
    "remaster": [
        r"\bremaster(?:ed)?\b",
    ],
    "version": [
        r"\bversion\b",
        r"\balternate\s+version\b",
        r"\balt\.\s+version\b",
    ],
    "take": [
        r"\btake\s+\d+\b",
        r"\(take\s+\d+",
    ],
    "mono": [
        r"\bmono\s+(?:version|mix|master|recording)\b",
        r"\(mono\b",
        r"\[mono\b",
    ],
    "stereo": [
        r"\bstereo\s+(?:version|mix|master|recording)\b",
        r"\(stereo\b",
        r"\[stereo\b",
        r"\bstereo\s+\d",
    ],
    "edit": [
        r"\bradio\s+edit\b",
        r"\bsingle\s+edit\b",
        r"\(edit\)",
        r"-\s*edit\b",
    ],
    "outtake": [
        r"\bouttake\b",
    ],
    "alternate": [
        r"\balternate\s+take\b",
        r"\balt\.\s+take\b",
    ],
}


def detect_title_artifacts(title: str | None) -> Set[str]:
    """Return the set of artifact flags present in `title`.

    Detection is case-insensitive with word-boundary matching.
    Returns an empty set for None or empty strings.
    """
    if not title:
        return set()
    text = str(title)
    flags: set[str] = set()
    for flag, patterns in _FLAG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                flags.add(flag)
                break
    return flags


def compute_title_artifact_penalty(
    *,
    title: str | None,
    weights: dict[str, float],
) -> float:
    """Sum of weights for each flag detected in the title.

    `weights` maps flag name (e.g., 'demo') to per-flag penalty magnitude.
    Flags detected but not present in `weights` contribute nothing.
    Returns 0.0 for empty/None titles or empty weights.
    """
    if not title or not weights:
        return 0.0
    flags = detect_title_artifacts(title)
    if not flags:
        return 0.0
    total = 0.0
    for flag in flags:
        w = weights.get(flag)
        if w is None:
            continue
        try:
            total += float(w)
        except Exception:
            continue
    return float(max(0.0, total))
