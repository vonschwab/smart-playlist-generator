"""
Title Deduplication Module
==========================
Fuzzy title matching to prevent the same song appearing multiple times
in a playlist across different releases (remasters, compilations, live versions).

Features:
- Scoped by artist: only considers duplicates within the same artist
- Two normalization modes: "strict" (conservative) and "loose" (aggressive)
- Configurable fuzzy threshold with short-title safeguards
- Uses difflib.SequenceMatcher consistent with existing track_matcher.py
"""
import re
import logging
from typing import Dict, Set, List, Optional, Tuple
from difflib import SequenceMatcher

from .string_utils import normalize_match_string

logger = logging.getLogger(__name__)

# Version keywords that indicate a track variant (not part of the core title)
# These are only stripped in "loose" mode
VERSION_KEYWORDS = {
    'live', 'demo', 'remaster', 'remastered', 'edit', 'version', 'ver',
    'mono', 'stereo', 'acoustic', 'instrumental', 'mix', 'remix', 'rmx',
    're-record', 'rerecord', 'session', 'alternate', 'alt', 'alternative',
    'radio', 'clean', 'explicit', 'bonus', 'anniversary', 'deluxe',
    'extended', 'single', 'album', 'original', 'reprise', 'interlude',
    'intro', 'outro', 'stripped', 'unplugged', 'orchestral', 'symphonic',
    'take', 'outtake', 'rough', 'early', 'late', 'final', 'master',
    '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
    '2020', '2021', '2022', '2023', '2024', '2025',  # Year tags
}

# Pattern to match parenthetical/bracketed content containing version keywords
_VERSION_BRACKET_PATTERN = re.compile(
    r'\s*[\(\[]([^\)\]]+)[\)\]]',
    re.IGNORECASE
)

# Pattern to match dash suffixes like " - Live", " - Remaster", etc.
_DASH_SUFFIX_PATTERN = re.compile(
    r'\s+-\s+(' + '|'.join(re.escape(kw) for kw in VERSION_KEYWORDS) + r')(\s|$)',
    re.IGNORECASE
)


def _contains_version_keyword(text: str) -> bool:
    """Check if text contains any version-related keywords."""
    text_lower = text.lower()
    # Check for exact word matches
    words = set(re.findall(r'\b\w+\b', text_lower))
    return bool(words & VERSION_KEYWORDS)


def normalize_title_for_dedupe(title: str, mode: str = "loose") -> str:
    """
    Normalize title for duplicate detection.

    Args:
        title: Original track title
        mode: "strict" (conservative) or "loose" (aggressive)

    Returns:
        Normalized title string

    Mode differences:
    - strict: casefold, trim, collapse whitespace, unify punctuation
    - loose: additionally strip version-related parenthetical/bracket content
             and dash suffixes, plus feat/ft sections
    """
    if not title:
        return ""

    # Basic normalization (both modes)
    normalized = title.lower().strip()

    # Replace common punctuation variants
    normalized = normalized.replace("'", "'")
    normalized = normalized.replace("'", "'")
    normalized = normalized.replace(""", '"')
    normalized = normalized.replace(""", '"')
    normalized = normalized.replace("…", "...")

    if mode == "loose":
        # Strip feat/ft sections
        normalized = re.sub(
            r'\s+(feat|ft|featuring|with)[\.\s]+.*$',
            '',
            normalized,
            flags=re.IGNORECASE
        )

        # Strip dash suffixes containing version keywords
        normalized = _DASH_SUFFIX_PATTERN.sub('', normalized)

        # Strip parenthetical/bracketed content IF it contains version keywords
        def strip_if_version(match):
            content = match.group(1)
            if _contains_version_keyword(content):
                return ''
            return match.group(0)  # Keep if no version keyword

        normalized = _VERSION_BRACKET_PATTERN.sub(strip_if_version, normalized)

    # Remove special characters (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', ' ', normalized)

    # Collapse multiple spaces
    normalized = ' '.join(normalized.split())

    return normalized.strip()


def normalize_artist_key(artist: str) -> str:
    """
    Normalize artist name for grouping in dedupe.
    Uses the same normalization as track_matcher for consistency.
    """
    return normalize_match_string(artist, is_artist=True)


def title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two normalized titles.
    Uses SequenceMatcher for consistency with track_matcher.py.

    Args:
        title1: First normalized title
        title2: Second normalized title

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, title1, title2).ratio()


class TitleDedupeTracker:
    """
    Tracks seen titles per artist to detect duplicates during playlist construction.

    Usage:
        tracker = TitleDedupeTracker(threshold=92, mode="loose")

        # Check if a candidate is a duplicate
        if tracker.is_duplicate("Artist", "Song Title (Remastered 2011)"):
            skip_this_track()
        else:
            tracker.add("Artist", "Song Title (Remastered 2011)")
    """

    def __init__(
        self,
        threshold: int = 92,
        mode: str = "loose",
        short_title_min_len: int = 6,
        enabled: bool = True,
    ):
        """
        Initialize tracker.

        Args:
            threshold: Fuzzy match threshold (0-100). 92 is a good default.
            mode: Normalization mode ("strict" or "loose")
            short_title_min_len: Titles shorter than this require exact match
            enabled: Set to False to disable dedupe entirely
        """
        self.threshold = threshold / 100.0  # Convert to 0-1 range
        self.mode = mode
        self.short_title_min_len = short_title_min_len
        self.enabled = enabled

        # Store: artist_key -> list of (original_title, normalized_title)
        self._seen: Dict[str, List[Tuple[str, str]]] = {}
        self._duplicate_count = 0

    def _get_normalized(self, artist: str, title: str) -> Tuple[str, str]:
        """Get normalized artist key and title."""
        artist_key = normalize_artist_key(artist)
        title_normalized = normalize_title_for_dedupe(title, mode=self.mode)
        return artist_key, title_normalized

    def is_duplicate(
        self,
        artist: str,
        title: str,
        debug: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this (artist, title) is a duplicate of something already seen.

        Args:
            artist: Artist name
            title: Track title
            debug: If True, log match details

        Returns:
            Tuple of (is_duplicate, matched_title_or_none)
        """
        if not self.enabled:
            return False, None

        artist_key, title_normalized = self._get_normalized(artist, title)

        if not artist_key or not title_normalized:
            return False, None

        # Look up seen titles for this artist
        seen_titles = self._seen.get(artist_key, [])

        # Short titles require exact match
        require_exact = len(title_normalized) < self.short_title_min_len

        for orig_title, norm_title in seen_titles:
            if require_exact:
                if title_normalized == norm_title:
                    if debug:
                        logger.debug(
                            f"Title dedupe: exact match for short title "
                            f"'{title}' -> '{orig_title}' (artist: {artist})"
                        )
                    return True, orig_title
            else:
                score = title_similarity(title_normalized, norm_title)
                if score >= self.threshold:
                    if debug:
                        logger.debug(
                            f"Title dedupe: fuzzy match {score:.2%} "
                            f"'{title}' ({title_normalized}) vs "
                            f"'{orig_title}' ({norm_title}) (artist: {artist})"
                        )
                    return True, orig_title

        return False, None

    def add(self, artist: str, title: str, debug: bool = False) -> None:
        """
        Add an (artist, title) pair to the seen set.

        Args:
            artist: Artist name
            title: Track title
            debug: If True, log the addition
        """
        if not self.enabled:
            return

        artist_key, title_normalized = self._get_normalized(artist, title)
        if debug:
            logger.debug("TitleDedupe ADD: artist=%s title='%s' -> key=%s norm='%s'",
                        artist, title, artist_key, title_normalized)

        if not artist_key or not title_normalized:
            return

        if artist_key not in self._seen:
            self._seen[artist_key] = []

        self._seen[artist_key].append((title, title_normalized))

    def check_and_add(
        self,
        artist: str,
        title: str,
        debug: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if duplicate, and if not, add to seen set.
        Convenience method combining is_duplicate() and add().

        Args:
            artist: Artist name
            title: Track title
            debug: If True, log match details

        Returns:
            Tuple of (is_duplicate, matched_title_or_none)
        """
        is_dup, matched = self.is_duplicate(artist, title, debug=debug)
        if is_dup:
            self._duplicate_count += 1
        else:
            self.add(artist, title)
        return is_dup, matched

    @property
    def duplicate_count(self) -> int:
        """Number of duplicates detected so far."""
        return self._duplicate_count

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the tracker state."""
        total_titles = sum(len(titles) for titles in self._seen.values())
        return {
            'artists_tracked': len(self._seen),
            'titles_tracked': total_titles,
            'duplicates_detected': self._duplicate_count,
        }

    def reset(self) -> None:
        """Clear all tracked titles."""
        self._seen.clear()
        self._duplicate_count = 0


def calculate_version_preference_score(title: str) -> int:
    """
    Calculate a preference score for a track version.
    Higher score = more preferred (canonical/album version).

    Useful for choosing between duplicate tracks.

    Args:
        title: Track title

    Returns:
        Preference score (higher = more canonical)
    """
    title_lower = title.lower()
    score = 100  # Base score

    # Penalize version indicators
    penalties = {
        'live': -30,
        'demo': -25,
        'remix': -20,
        'acoustic': -15,
        'instrumental': -15,
        'radio edit': -10,
        'single version': -10,
        'remaster': -5,  # Slight penalty, but often better quality
        'remastered': -5,
        'bonus': -10,
        'alternate': -15,
        'extended': -10,
    }

    for keyword, penalty in penalties.items():
        if keyword in title_lower:
            score += penalty

    # Bonus for explicit "album version"
    if 'album version' in title_lower:
        score += 10

    return score
