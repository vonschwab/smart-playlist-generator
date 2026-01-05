from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from src.features.artifacts import ArtifactBundle
from src.string_utils import normalize_artist_key as _normalize_artist_key_punct
from src.string_utils import normalize_match_string, normalize_artist_name
from src.title_dedupe import normalize_title_for_dedupe
from src.artist_utils import extract_primary_artist  # Deprecated, kept for backward compat


def normalize_primary_artist_key(value: str) -> str:
    """
    Normalize an "artist identity" key robust to common collaboration patterns.

    Intent: collapse variants like:
      - "Mount Eerie" vs "Mount Eerie with Julie Doiron & Fred Squire"
      - "Charli XCX" vs "Charli XCX feat. MØ"
      - "Artist A x Artist B" (treat as collab)
      - "Bill Evans Trio" vs "Bill Evans" (ensemble normalization)

    Uses extract_primary_artist() which handles ensemble suffixes (Trio, Quartet, etc.)
    and collaboration markers properly.
    """
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Treat "x" / "×" collaborations similarly to "feat" to avoid bypassing 1-per-artist constraints.
    text = text.replace("×", " x ")
    text = re.sub(r"\s+x\s+", " feat ", text, flags=re.IGNORECASE)
    # Use extract_primary_artist for ensemble normalization
    return extract_primary_artist(text, lowercase=True)


def normalize_title_key(value: str) -> str:
    """Normalize a title key for duplicate detection (stable across versions)."""
    if not value:
        return ""
    return normalize_title_for_dedupe(str(value), mode="loose")


@dataclass(frozen=True)
class TrackIdentityKeys:
    artist_key: str
    title_key: str

    @property
    def track_key(self) -> Tuple[str, str]:
        return (self.artist_key, self.title_key)


def identity_keys_for_index(bundle: ArtifactBundle, idx: int) -> TrackIdentityKeys:
    """
    Compute robust (artist_key, title_key) for an artifact bundle index.

    Falls back to per-track sentinels when metadata is missing to avoid accidental collisions.
    """
    tid = ""
    try:
        tid = str(bundle.track_ids[int(idx)])
    except Exception:
        tid = str(idx)

    artist_raw: Optional[str] = None
    try:
        if bundle.track_artists is not None:
            artist_raw = str(bundle.track_artists[int(idx)] or "") or None
    except Exception:
        artist_raw = None
    if not artist_raw:
        try:
            artist_raw = str(bundle.artist_keys[int(idx)] or "") or None
        except Exception:
            artist_raw = None

    title_raw: Optional[str] = None
    try:
        if bundle.track_titles is not None:
            title_raw = str(bundle.track_titles[int(idx)] or "") or None
    except Exception:
        title_raw = None

    artist_key = normalize_primary_artist_key(artist_raw or "")
    if not artist_key:
        artist_key = _normalize_artist_key_punct(artist_raw or "")
    if not artist_key:
        artist_key = f"unknown_artist:{tid}"

    title_key = normalize_title_key(title_raw or "")
    if not title_key:
        title_key = f"unknown_title:{tid}"

    return TrackIdentityKeys(artist_key=artist_key, title_key=title_key)

