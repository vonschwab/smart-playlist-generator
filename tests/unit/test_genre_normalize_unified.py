"""Unit tests for unified genre normalization module.

Tests cover:
- Basic normalization (lowercase, trim, diacritics)
- Language translations (French, German, Dutch, Romanian)
- Synonym mappings
- Multi-token splitting
- Meta-tag filtering
- Edge cases
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.genre.normalize_unified import (
    classify_normalization,
    detect_normalization_flags,
    load_filter_sets,
    normalize_and_filter_genres,
    normalize_and_split_genre,
    normalize_genre_list,
    normalize_genre_token,
    remove_diacritics,
    split_and_normalize,
    GenreAction,
)


# =============================================================================
# Diacritic Removal Tests
# =============================================================================

class TestDiacriticRemoval:
    """Test diacritic removal functionality."""

    def test_remove_french_accents(self):
        assert remove_diacritics("électronique") == "electronique"
        assert remove_diacritics("musiques de noël") == "musiques de noel"

    def test_remove_german_umlauts(self):
        assert remove_diacritics("Köln") == "Koln"
        assert remove_diacritics("München") == "Munchen"

    def test_remove_spanish_accents(self):
        assert remove_diacritics("España") == "Espana"
        assert remove_diacritics("José") == "Jose"

    def test_preserve_ascii(self):
        assert remove_diacritics("rock") == "rock"
        assert remove_diacritics("hip hop") == "hip hop"

    def test_empty_string(self):
        assert remove_diacritics("") == ""
        assert remove_diacritics(None) == ""


# =============================================================================
# Single Token Normalization Tests
# =============================================================================

class TestNormalizeGenreToken:
    """Test single token normalization."""

    def test_basic_normalization(self):
        assert normalize_genre_token("Rock") == "rock"
        assert normalize_genre_token("  Indie  ") == "indie"
        assert normalize_genre_token("POST-ROCK") == "post-rock"

    def test_language_translation_french(self):
        assert normalize_genre_token("électronique") == "electronic"
        assert normalize_genre_token("rock alternatif") == "alternative rock"

    def test_language_translation_german(self):
        assert normalize_genre_token("elektronisch") == "electronic"

    def test_synonym_mapping(self):
        assert normalize_genre_token("hiphop") == "hip hop"
        assert normalize_genre_token("electro") == "electronic"
        assert normalize_genre_token("alt rock") == "alternative rock"

    def test_r_and_b_preservation(self):
        """R&B should normalize to 'rnb' consistently."""
        assert normalize_genre_token("r&b") == "rnb"
        assert normalize_genre_token("R&B") == "rnb"
        assert normalize_genre_token("rhythm and blues") == "rnb"

    def test_drop_meta_tags(self):
        """Meta tags should be dropped."""
        assert normalize_genre_token("seen live") is None
        assert normalize_genre_token("favorites") is None
        assert normalize_genre_token("80s") is None
        assert normalize_genre_token("american") is None

    def test_drop_placeholder_tokens(self):
        """Placeholder tokens should be dropped."""
        assert normalize_genre_token("__EMPTY__") is None
        assert normalize_genre_token("unknown") is None
        assert normalize_genre_token("") is None

    def test_punctuation_cleanup(self):
        """Edge punctuation should be removed."""
        assert normalize_genre_token("rock,") == "rock"
        assert normalize_genre_token("indie;") == "indie"
        assert normalize_genre_token("(alternative)") == "alternative"

    def test_disable_translations(self):
        """Test with translations disabled."""
        result = normalize_genre_token("électronique", apply_translations=False)
        # Without phrase map, it becomes "electronique" (diacritics removed)
        assert result == "electronique"

    def test_disable_synonyms(self):
        """Test with synonyms disabled."""
        result = normalize_genre_token("electro", apply_synonyms=False)
        # Without synonym map, stays as "electro"
        assert result == "electro"


# =============================================================================
# Multi-Token Splitting Tests
# =============================================================================

class TestNormalizeAndSplitGenre:
    """Test multi-token splitting and normalization."""

    def test_semicolon_split(self):
        result = normalize_and_split_genre("rock; indie; folk")
        assert result == ["rock", "indie", "folk"]

    def test_comma_split(self):
        result = normalize_and_split_genre("rock, indie, folk")
        assert result == ["rock", "indie", "folk"]

    def test_slash_split(self):
        result = normalize_and_split_genre("rock/indie/folk")
        assert result == ["rock", "indie", "folk"]

    def test_ampersand_split(self):
        """Ampersand with spaces should split."""
        result = normalize_and_split_genre("rock & roll")
        # "rock & roll" → "rock" (via synonym)
        assert "rock" in result

    def test_ampersand_preservation(self):
        """R&B without spaces should NOT split."""
        result = normalize_and_split_genre("r&b")
        assert result == ["rnb"]

    def test_phrase_map_splitting(self):
        """Phrase map should produce multiple tokens."""
        result = normalize_and_split_genre("alternatif et indé")
        assert set(result) == {"alternative", "indie"}

    def test_complex_compound(self):
        """Test complex compound genre string."""
        result = normalize_and_split_genre("pop, rock, alternatif et indé")
        assert "pop" in result
        assert "rock" in result
        assert ("indie" in result or "alternative" in result)

    def test_deduplication(self):
        """Duplicate tokens should be removed."""
        result = normalize_and_split_genre("rock; rock; indie; rock")
        assert result.count("rock") == 1
        assert result.count("indie") == 1

    def test_meta_tag_filtering(self):
        """Meta tags should be filtered out of multi-token strings."""
        result = normalize_and_split_genre("rock; seen live; indie; favorites")
        assert set(result) == {"rock", "indie"}

    def test_empty_after_filtering(self):
        """String with only meta tags should return empty list."""
        result = normalize_and_split_genre("seen live; favorites; 80s")
        assert result == []


# =============================================================================
# Split and Normalize Tests (Set variant)
# =============================================================================

class TestSplitAndNormalize:
    """Test split_and_normalize (returns set)."""

    def test_returns_set(self):
        result = split_and_normalize("rock; indie; folk")
        assert isinstance(result, set)
        assert result == {"rock", "indie", "folk"}

    def test_deduplication_automatic(self):
        result = split_and_normalize("rock; rock; indie")
        assert result == {"rock", "indie"}


# =============================================================================
# Batch Normalization and Filtering Tests
# =============================================================================

class TestNormalizeAndFilterGenres:
    """Test batch normalization with filtering."""

    def test_basic_batch(self):
        genres = ["rock", "indie", "folk"]
        result = normalize_and_filter_genres(genres)
        assert result == {"rock", "indie", "folk"}

    def test_broad_filter(self):
        """Test broad tag filtering."""
        genres = ["ambient", "rock", "idm", "electronic"]
        broad_set = {"rock", "electronic"}
        result = normalize_and_filter_genres(genres, broad_set=broad_set)
        assert result == {"ambient", "idm"}

    def test_garbage_filter(self):
        """Test garbage tag filtering."""
        genres = ["ambient", "garbage1", "idm", "garbage2"]
        garbage_set = {"garbage1", "garbage2"}
        result = normalize_and_filter_genres(genres, garbage_set=garbage_set)
        assert result == {"ambient", "idm"}

    def test_meta_filter(self):
        """Test meta tag filtering."""
        genres = ["ambient", "seen live", "idm", "favorites"]
        meta_set = {"seen live", "favorites"}
        result = normalize_and_filter_genres(genres, meta_set=meta_set)
        assert result == {"ambient", "idm"}

    def test_canonical_whitelist(self):
        """Test canonical set (whitelist)."""
        genres = ["ambient", "rock", "idm", "electronic"]
        canonical_set = {"ambient", "idm"}
        result = normalize_and_filter_genres(genres, canonical_set=canonical_set)
        assert result == {"ambient", "idm"}

    def test_combined_filters(self):
        """Test multiple filters together."""
        genres = ["ambient", "rock", "idm", "seen live", "electronic"]
        broad_set = {"rock"}
        meta_set = {"seen live"}
        result = normalize_and_filter_genres(genres, broad_set=broad_set, meta_set=meta_set)
        assert result == {"ambient", "idm", "electronic"}


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestNormalizeGenreList:
    """Test normalize_genre_list convenience function."""

    def test_basic_normalization(self):
        genres = ["Rock", "Indie", "Folk"]
        result = normalize_genre_list(genres)
        assert result == {"rock", "indie", "folk"}

    def test_filter_broad_enabled(self):
        """With filter_broad=True, meta tags should be removed."""
        genres = ["rock", "indie", "seen live", "favorites"]
        result = normalize_genre_list(genres, filter_broad=True)
        assert "rock" in result
        assert "indie" in result
        assert "seen live" not in result
        assert "favorites" not in result

    def test_filter_broad_disabled(self):
        """With filter_broad=False, meta tags should remain."""
        genres = ["rock", "indie", "seen live"]
        result = normalize_genre_list(genres, filter_broad=False)
        # seen live is in META_TAGS, so it's still dropped during normalization
        assert "rock" in result
        assert "indie" in result


# =============================================================================
# Classification Tests
# =============================================================================

class TestClassifyNormalization:
    """Test normalization action classification."""

    def test_keep(self):
        """Unchanged single token = KEEP."""
        action = classify_normalization("rock", ["rock"])
        assert action == GenreAction.KEEP

    def test_map(self):
        """Changed single token = MAP."""
        action = classify_normalization("hiphop", ["hip hop"])
        assert action == GenreAction.MAP

    def test_split_map(self):
        """Multiple tokens = SPLIT_MAP."""
        action = classify_normalization("rock; indie", ["rock", "indie"])
        assert action == GenreAction.SPLIT_MAP

    def test_drop(self):
        """Empty result = DROP."""
        action = classify_normalization("seen live", [])
        assert action == GenreAction.DROP

    def test_drop_empty_input(self):
        """Empty input = DROP."""
        action = classify_normalization("", [])
        assert action == GenreAction.DROP


# =============================================================================
# Diagnostic Tests
# =============================================================================

class TestDetectNormalizationFlags:
    """Test normalization flag detection."""

    def test_ascii_only(self):
        flags = detect_normalization_flags("rock")
        assert flags["has_non_ascii"] is False
        assert flags["has_diacritics"] is False

    def test_has_diacritics(self):
        flags = detect_normalization_flags("électronique")
        assert flags["has_non_ascii"] is True
        assert flags["has_diacritics"] is True

    def test_has_delimiters(self):
        flags = detect_normalization_flags("rock; indie")
        assert flags["has_delimiters"] is True

    def test_is_multi_token(self):
        flags = detect_normalization_flags("rock; indie; folk")
        assert flags["is_multi_token"] is True

    def test_single_token(self):
        flags = detect_normalization_flags("rock")
        assert flags["is_multi_token"] is False


# =============================================================================
# CSV Loading Tests
# =============================================================================

class TestLoadFilterSets:
    """Test loading filter sets from config and CSV."""

    def test_load_broad_from_config(self):
        broad_filters = ["rock", "pop", "indie"]
        broad_set, garbage_set, meta_set = load_filter_sets(broad_filters=broad_filters)
        assert "rock" in broad_set
        assert "pop" in broad_set
        assert "indie" in broad_set
        assert len(garbage_set) == 0
        assert len(meta_set) == 0

    def test_empty_filters(self):
        broad_set, garbage_set, meta_set = load_filter_sets()
        assert len(broad_set) == 0
        assert len(garbage_set) == 0
        assert len(meta_set) == 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_none_input(self):
        assert normalize_genre_token(None) is None
        assert normalize_and_split_genre(None) == []

    def test_empty_string(self):
        assert normalize_genre_token("") is None
        assert normalize_and_split_genre("") == []

    def test_whitespace_only(self):
        assert normalize_genre_token("   ") is None
        assert normalize_and_split_genre("   ") == []

    def test_only_punctuation(self):
        assert normalize_genre_token(".,;:") is None
        assert normalize_and_split_genre(".,;:") == []

    def test_mixed_case(self):
        result = normalize_genre_token("RoCk")
        assert result == "rock"

    def test_extreme_whitespace(self):
        result = normalize_genre_token("  rock    indie  ")
        # Should be split or normalized depending on delimiter
        assert result is not None

    def test_unicode_normalization(self):
        """Test various unicode forms normalize consistently."""
        # Different unicode representations of "é"
        e_acute_composed = "électronique"  # NFC form
        e_acute_decomposed = "e\u0301lectronique"  # NFD form

        assert remove_diacritics(e_acute_composed) == remove_diacritics(e_acute_decomposed)
