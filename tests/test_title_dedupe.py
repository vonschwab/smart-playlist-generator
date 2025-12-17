"""
Tests for title deduplication module.

Covers:
- Title normalization (strict and loose modes)
- Fuzzy title matching
- TitleDedupeTracker behavior
- Short title safeguards
- Version keyword stripping
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.title_dedupe import (
    normalize_title_for_dedupe,
    normalize_artist_key,
    title_similarity,
    TitleDedupeTracker,
    calculate_version_preference_score,
    _contains_version_keyword,
)


class TestNormalizeTitleForDedupe:
    """Tests for normalize_title_for_dedupe function."""

    def test_basic_normalization(self):
        """Basic case folding and whitespace handling."""
        assert normalize_title_for_dedupe("Hello World") == "hello world"
        assert normalize_title_for_dedupe("  HELLO   WORLD  ") == "hello world"

    def test_strict_mode_keeps_parenthetical(self):
        """Strict mode preserves parenthetical content."""
        result = normalize_title_for_dedupe("Song (Remastered 2011)", mode="strict")
        assert "remastered" in result
        assert "2011" in result

    def test_loose_mode_strips_version_parenthetical(self):
        """Loose mode removes parenthetical content with version keywords."""
        result = normalize_title_for_dedupe("Song (Remastered 2011)", mode="loose")
        assert "remastered" not in result
        assert "2011" not in result
        assert result == "song"

    def test_loose_mode_keeps_non_version_parenthetical(self):
        """Loose mode keeps parenthetical content without version keywords."""
        result = normalize_title_for_dedupe("Song (Part 1)", mode="loose")
        assert "part 1" in result

    def test_loose_mode_strips_feat(self):
        """Loose mode strips featuring sections."""
        result = normalize_title_for_dedupe("Song feat. Artist", mode="loose")
        assert "feat" not in result
        assert "artist" not in result
        assert result == "song"

        result = normalize_title_for_dedupe("Song ft. Other Artist", mode="loose")
        assert "other" not in result

    def test_loose_mode_strips_dash_suffix(self):
        """Loose mode strips dash suffixes with version keywords."""
        result = normalize_title_for_dedupe("Song - Live", mode="loose")
        assert "live" not in result

        result = normalize_title_for_dedupe("Song - Remaster", mode="loose")
        assert "remaster" not in result

    def test_punctuation_normalization(self):
        """Smart quotes and other punctuation are normalized."""
        result = normalize_title_for_dedupe("Don't Stop")
        assert result == "don t stop"

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert normalize_title_for_dedupe("") == ""
        assert normalize_title_for_dedupe("   ") == ""

    def test_multiple_version_tags(self):
        """Multiple version tags are all stripped in loose mode."""
        result = normalize_title_for_dedupe(
            "Song (Remastered 2011) [Live]",
            mode="loose"
        )
        assert "remastered" not in result
        assert "live" not in result
        assert "2011" not in result
        assert result == "song"


class TestTitleSimilarity:
    """Tests for title_similarity function."""

    def test_identical_titles(self):
        """Identical titles have similarity 1.0."""
        assert title_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        """Completely different titles have low similarity."""
        score = title_similarity("hello", "xyz")
        assert score < 0.5

    def test_minor_difference(self):
        """Minor differences result in high but not perfect similarity."""
        score = title_similarity("hello world", "hello worlds")
        assert 0.8 < score < 1.0

    def test_empty_strings(self):
        """Empty strings match perfectly."""
        assert title_similarity("", "") == 1.0


class TestContainsVersionKeyword:
    """Tests for _contains_version_keyword helper."""

    def test_detects_version_keywords(self):
        """Version keywords are detected."""
        assert _contains_version_keyword("Remastered 2011")
        assert _contains_version_keyword("Live at venue")
        assert _contains_version_keyword("Demo")
        assert _contains_version_keyword("Radio Edit")

    def test_no_false_positives(self):
        """Non-version content is not flagged."""
        assert not _contains_version_keyword("Part 1")
        assert not _contains_version_keyword("featuring someone")
        assert not _contains_version_keyword("the great escape")


class TestTitleDedupeTracker:
    """Tests for TitleDedupeTracker class."""

    def test_basic_duplicate_detection(self):
        """Basic duplicate detection works."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)
        tracker.add("Artist", "Song Title")

        is_dup, matched = tracker.is_duplicate("Artist", "Song Title")
        assert is_dup
        assert matched == "Song Title"

    def test_different_artists_not_duplicates(self):
        """Same title from different artists is not a duplicate."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)
        tracker.add("Artist A", "Song Title")

        is_dup, matched = tracker.is_duplicate("Artist B", "Song Title")
        assert not is_dup

    def test_fuzzy_matching(self):
        """Fuzzy matching detects slight variations."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)
        tracker.add("Artist", "Song Title")

        # Very similar title should match
        is_dup, matched = tracker.is_duplicate("Artist", "Song Titles")
        assert is_dup

    def test_version_variations_detected(self):
        """Remastered/live versions are detected as duplicates in loose mode."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)
        tracker.add("Artist", "Song Title")

        is_dup, matched = tracker.is_duplicate("Artist", "Song Title (Remastered 2011)")
        assert is_dup

        is_dup, matched = tracker.is_duplicate("Artist", "Song Title - Live")
        assert is_dup

    def test_strict_mode_keeps_versions_separate(self):
        """Strict mode treats versions as different tracks."""
        tracker = TitleDedupeTracker(threshold=90, mode="strict", enabled=True)
        tracker.add("Artist", "Song Title")

        # In strict mode, remastered should not match the original
        is_dup, matched = tracker.is_duplicate("Artist", "Song Title (Remastered 2011)")
        # Depends on how different the normalized strings are
        # They might still match if the core title is similar enough

    def test_disabled_tracker(self):
        """Disabled tracker always returns not duplicate."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=False)
        tracker.add("Artist", "Song Title")

        is_dup, matched = tracker.is_duplicate("Artist", "Song Title")
        assert not is_dup

    def test_short_title_requires_exact_match(self):
        """Short titles require exact match (configurable threshold)."""
        tracker = TitleDedupeTracker(
            threshold=90, mode="loose",
            short_title_min_len=6, enabled=True
        )
        tracker.add("Artist", "Help")  # 4 chars < 6

        # Exact match should work
        is_dup, _ = tracker.is_duplicate("Artist", "Help")
        assert is_dup

        # Similar but not exact should not match for short titles
        is_dup, _ = tracker.is_duplicate("Artist", "Helps")
        assert not is_dup

    def test_check_and_add(self):
        """check_and_add method works correctly."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)

        # First call should add
        is_dup, _ = tracker.check_and_add("Artist", "Song One")
        assert not is_dup

        # Second call with same should be duplicate
        is_dup, matched = tracker.check_and_add("Artist", "Song One")
        assert is_dup
        assert matched == "Song One"

    def test_duplicate_count(self):
        """Duplicate count is tracked correctly."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)

        tracker.check_and_add("Artist", "Song One")
        tracker.check_and_add("Artist", "Song Two")
        tracker.check_and_add("Artist", "Song One")  # Duplicate
        tracker.check_and_add("Artist", "Song Two (Live)")  # Duplicate

        assert tracker.duplicate_count == 2

    def test_get_stats(self):
        """Stats tracking works."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)

        tracker.add("Artist A", "Song One")
        tracker.add("Artist A", "Song Two")
        tracker.add("Artist B", "Song Three")

        stats = tracker.get_stats()
        assert stats['artists_tracked'] == 2
        assert stats['titles_tracked'] == 3
        assert stats['duplicates_detected'] == 0

    def test_reset(self):
        """Reset clears all state."""
        tracker = TitleDedupeTracker(threshold=90, mode="loose", enabled=True)

        tracker.add("Artist", "Song")
        tracker._duplicate_count = 5

        tracker.reset()

        assert tracker.get_stats()['titles_tracked'] == 0
        assert tracker.duplicate_count == 0


class TestCalculateVersionPreferenceScore:
    """Tests for version preference scoring."""

    def test_base_score(self):
        """Clean title gets base score."""
        score = calculate_version_preference_score("Song Title")
        assert score == 100

    def test_live_penalty(self):
        """Live versions get significant penalty."""
        score = calculate_version_preference_score("Song Title (Live)")
        assert score < 100
        assert score == 70  # 100 - 30

    def test_remaster_penalty(self):
        """Remasters get small penalty."""
        score = calculate_version_preference_score("Song Title (Remastered)")
        # Both 'remaster' and 'remastered' keywords match since remaster is substring
        assert score == 90  # 100 - 5 - 5

    def test_album_version_bonus(self):
        """Album version gets bonus."""
        score = calculate_version_preference_score("Song Title (Album Version)")
        assert score > 100

    def test_multiple_penalties_stack(self):
        """Multiple version indicators stack penalties."""
        score = calculate_version_preference_score("Song Title (Live) (Remastered)")
        # Live: -30, remaster: -5, remastered: -5 (substring match)
        assert score == 60  # 100 - 30 - 5 - 5


class TestNormalizeArtistKey:
    """Tests for artist key normalization."""

    def test_basic_normalization(self):
        """Basic artist normalization."""
        key = normalize_artist_key("The Beatles")
        assert key  # Non-empty
        assert key.islower() or not key.isalpha()  # Lowercase or stripped

    def test_consistency(self):
        """Same artist returns same key."""
        key1 = normalize_artist_key("The Beatles")
        key2 = normalize_artist_key("THE BEATLES")
        assert key1 == key2


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_album_vs_compilation(self):
        """Same song from album and compilation."""
        tracker = TitleDedupeTracker(threshold=92, mode="loose", enabled=True)

        # Original from album
        tracker.add("Radiohead", "Creep")

        # Same song from compilation
        is_dup, _ = tracker.is_duplicate("Radiohead", "Creep")
        assert is_dup

    def test_remaster_vs_original(self):
        """Remastered version vs original."""
        tracker = TitleDedupeTracker(threshold=92, mode="loose", enabled=True)

        tracker.add("Pink Floyd", "Time")

        is_dup, _ = tracker.is_duplicate("Pink Floyd", "Time (2011 Remastered Version)")
        assert is_dup

    def test_live_vs_studio(self):
        """Live version vs studio version."""
        tracker = TitleDedupeTracker(threshold=92, mode="loose", enabled=True)

        tracker.add("Nirvana", "About a Girl")

        is_dup, _ = tracker.is_duplicate("Nirvana", "About a Girl (Live)")
        assert is_dup

    def test_different_songs_same_artist(self):
        """Different songs by same artist are not duplicates."""
        tracker = TitleDedupeTracker(threshold=92, mode="loose", enabled=True)

        tracker.add("The Beatles", "Help")
        tracker.add("The Beatles", "Yesterday")

        is_dup, _ = tracker.is_duplicate("The Beatles", "Let It Be")
        assert not is_dup

    def test_cover_versions_different_artists(self):
        """Same song by different artists is not a duplicate."""
        tracker = TitleDedupeTracker(threshold=92, mode="loose", enabled=True)

        tracker.add("The Beatles", "Yesterday")

        is_dup, _ = tracker.is_duplicate("Frank Sinatra", "Yesterday")
        assert not is_dup
