"""
Unit tests for artist identity resolver.

Tests the behavior of resolve_artist_identity_keys() for:
- Single artists
- Ensemble variants
- Collaboration strings
- Edge cases
"""
import pytest
from src.playlist.artist_identity_resolver import (
    resolve_artist_identity_keys,
    ArtistIdentityConfig,
    _normalize_component,
    _strip_ensemble_designator,
    format_identity_keys_for_logging,
)


class TestNormalizeComponent:
    """Test component normalization helper."""

    def test_basic_normalization(self):
        assert _normalize_component("Bill Evans") == "bill evans"
        assert _normalize_component("  AHMAD  JAMAL  ") == "ahmad jamal"

    def test_strips_leading_the(self):
        assert _normalize_component("The Beatles") == "beatles"
        assert _normalize_component("the who") == "who"

    def test_preserves_internal_the(self):
        assert _normalize_component("Sly and the Family Stone") == "sly and the family stone"

    def test_empty_input(self):
        assert _normalize_component("") == ""
        assert _normalize_component("   ") == ""


class TestStripEnsembleDesignator:
    """Test ensemble term stripping."""

    def test_strips_trailing_trio(self):
        terms = ["trio", "quartet", "quintet"]
        assert _strip_ensemble_designator("bill evans trio", terms) == "bill evans"
        assert _strip_ensemble_designator("ahmad jamal trio", terms) == "ahmad jamal"

    def test_strips_trailing_quartet(self):
        terms = ["trio", "quartet", "quintet"]
        assert _strip_ensemble_designator("dave brubeck quartet", terms) == "dave brubeck"

    def test_strips_trailing_big_band(self):
        # Multi-word terms must be first in list
        terms = ["big band", "orchestra", "trio"]
        assert _strip_ensemble_designator("count basie big band", terms) == "count basie"

    def test_does_not_strip_mid_name(self):
        terms = ["ensemble"]
        # "ensemble" appears mid-name, not at end
        assert _strip_ensemble_designator("art ensemble of chicago", terms) == "art ensemble of chicago"

    def test_only_strips_one_term(self):
        terms = ["trio", "quartet"]
        # Only strips the first match found
        assert _strip_ensemble_designator("bill evans trio", terms) == "bill evans"

    def test_no_terms_list(self):
        assert _strip_ensemble_designator("bill evans trio", []) == "bill evans trio"


class TestResolveArtistIdentityKeys:
    """Test main identity key resolution function."""

    def test_feature_disabled_fallback(self):
        """When disabled, should fall back to basic normalization."""
        cfg = ArtistIdentityConfig(enabled=False)
        # When disabled, uses normalize_artist_key which has different behavior
        # Just verify it returns a non-empty set
        keys = resolve_artist_identity_keys("Bill Evans Trio", cfg)
        assert len(keys) == 1
        assert keys != {"bill evans"}  # Should NOT collapse when disabled

    def test_single_artist_no_ensemble(self):
        """Basic artist with no ensemble term."""
        cfg = ArtistIdentityConfig(enabled=True)
        assert resolve_artist_identity_keys("Bill Evans", cfg) == {"bill evans"}
        assert resolve_artist_identity_keys("Ahmad Jamal", cfg) == {"ahmad jamal"}

    def test_ensemble_variants_collapsed(self):
        """Ensemble suffixes should be stripped."""
        cfg = ArtistIdentityConfig(enabled=True)

        assert resolve_artist_identity_keys("Bill Evans Trio", cfg) == {"bill evans"}
        assert resolve_artist_identity_keys("Bill Evans Quartet", cfg) == {"bill evans"}
        assert resolve_artist_identity_keys("Bill Evans Quintet", cfg) == {"bill evans"}
        assert resolve_artist_identity_keys("Ahmad Jamal Quintet", cfg) == {"ahmad jamal"}
        assert resolve_artist_identity_keys("Dave Brubeck Quartet", cfg) == {"dave brubeck"}

    def test_big_band_multi_word_term(self):
        """Multi-word ensemble terms."""
        cfg = ArtistIdentityConfig(enabled=True)
        assert resolve_artist_identity_keys("Count Basie Big Band", cfg) == {"count basie"}

    def test_orchestra_stripped(self):
        """Orchestra suffix."""
        cfg = ArtistIdentityConfig(enabled=True)
        assert resolve_artist_identity_keys("Duke Ellington Orchestra", cfg) == {"duke ellington"}

    def test_collaboration_ampersand(self):
        """Collaborations with '&' should split."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Bob Brookmeyer & Bill Evans", cfg)
        assert keys == {"bob brookmeyer", "bill evans"}

    def test_collaboration_comma(self):
        """Collaborations with comma should split."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Duke Ellington, John Coltrane", cfg)
        assert keys == {"duke ellington", "john coltrane"}

    def test_collaboration_feat(self):
        """Collaborations with 'feat.' should split."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Charli XCX feat. MØ", cfg)
        assert "charli xcx" in keys
        assert "mø" in keys or "mo" in keys  # Unicode normalization might vary

    def test_collaboration_x(self):
        """Collaborations with ' x ' should split."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Artist A x Artist B", cfg)
        assert keys == {"artist a", "artist b"}

    def test_collab_with_ensemble_terms(self):
        """Collaboration where both participants have ensemble terms."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Bill Evans Trio & Ahmad Jamal Quintet", cfg)
        assert keys == {"bill evans", "ahmad jamal"}

    def test_preserves_mid_name_ensemble(self):
        """Should NOT strip 'ensemble' from middle of name."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Art Ensemble of Chicago", cfg)
        # "ensemble" is not trailing, so should be preserved
        assert "art ensemble of chicago" in keys

    def test_empty_string_fallback(self):
        """Empty input should return set with empty string."""
        cfg = ArtistIdentityConfig(enabled=True)
        assert resolve_artist_identity_keys("", cfg) == {""}
        assert resolve_artist_identity_keys(None, cfg) == {""}

    def test_whitespace_only_fallback(self):
        """Whitespace-only input."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("   ", cfg)
        assert keys == {""}

    def test_case_insensitive_splitting(self):
        """Delimiters should work case-insensitively."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("Artist A FEAT. Artist B", cfg)
        assert len(keys) == 2

    def test_multiple_delimiters(self):
        """String with multiple different delimiters."""
        cfg = ArtistIdentityConfig(enabled=True)
        keys = resolve_artist_identity_keys("A, B & C", cfg)
        assert len(keys) == 3

    def test_strip_disabled(self):
        """When strip_trailing_ensemble_terms=False, should not strip."""
        cfg = ArtistIdentityConfig(
            enabled=True,
            strip_trailing_ensemble_terms=False,
        )
        keys = resolve_artist_identity_keys("Bill Evans Trio", cfg)
        assert keys == {"bill evans trio"}

    def test_custom_delimiters(self):
        """Custom delimiter list."""
        cfg = ArtistIdentityConfig(
            enabled=True,
            split_delimiters=["|"],
        )
        keys = resolve_artist_identity_keys("Artist A | Artist B", cfg)
        assert len(keys) == 2


class TestFormatIdentityKeysForLogging:
    """Test logging formatter."""

    def test_empty_set(self):
        assert format_identity_keys_for_logging(set()) == "{}"

    def test_single_key(self):
        assert format_identity_keys_for_logging({"bill evans"}) == "{bill evans}"

    def test_two_keys(self):
        result = format_identity_keys_for_logging({"bill evans", "ahmad jamal"})
        # Should show both (order may vary due to set)
        assert "bill evans" in result or "ahmad jamal" in result
        assert "{" in result and "}" in result

    def test_truncation(self):
        keys = {"a", "b", "c", "d", "e"}
        result = format_identity_keys_for_logging(keys, max_keys=3)
        assert "... +" in result  # Should show truncation indicator


class TestMinGapScenario:
    """Test the exact scenario from the problem description."""

    def test_bill_evans_variants_collapse(self):
        """All Bill Evans variants should map to same identity."""
        cfg = ArtistIdentityConfig(enabled=True)

        variants = [
            "Bill Evans",
            "Bill Evans Trio",
            "Bill Evans Quintet",
        ]

        identity_sets = [resolve_artist_identity_keys(v, cfg) for v in variants]

        # All should resolve to same identity
        assert all(keys == {"bill evans"} for keys in identity_sets)

    def test_ahmad_jamal_variants_collapse(self):
        """Ahmad Jamal variants should collapse."""
        cfg = ArtistIdentityConfig(enabled=True)

        variants = [
            "Ahmad Jamal",
            "Ahmad Jamal Quintet",
        ]

        identity_sets = [resolve_artist_identity_keys(v, cfg) for v in variants]
        assert all(keys == {"ahmad jamal"} for keys in identity_sets)

    def test_collab_updates_both_identities(self):
        """Collaboration string should produce multiple identities."""
        cfg = ArtistIdentityConfig(enabled=True)

        keys = resolve_artist_identity_keys("Bob Brookmeyer & Bill Evans", cfg)

        # Should produce TWO identity keys
        assert len(keys) == 2
        assert "bob brookmeyer" in keys
        assert "bill evans" in keys

        # This means min_gap enforcement should update BOTH keys' last_seen positions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
