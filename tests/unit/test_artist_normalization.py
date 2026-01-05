"""Unit tests for artist normalization.

Tests cover:
- normalize_artist_key (existing, for database keys)
- normalize_artist_name (new canonical implementation)
"""

from src.string_utils import normalize_artist_key, normalize_artist_name


# =============================================================================
# Legacy Tests for normalize_artist_key
# =============================================================================

def test_normalize_artist_key_diacritics():
    assert normalize_artist_key("Luiz Bonfá") == "luiz bonfa"
    assert normalize_artist_key("Luiz Bonfa") == "luiz bonfa"
    assert normalize_artist_key("João Gilberto") == "joao gilberto"
    assert normalize_artist_key("Antônio Carlos Jobim") == "antonio carlos jobim"


def test_normalize_artist_key_typography():
    assert normalize_artist_key("Guns N' Roses") == "guns n roses"
    assert normalize_artist_key("Antonio—Carlos Jobim") == "antonio carlos jobim"
    assert normalize_artist_key("D'Angelo") == "d angelo"


# =============================================================================
# Tests for normalize_artist_name (Canonical Implementation)
# =============================================================================

class TestBasicNormalization:
    """Test basic normalization features."""

    def test_lowercase_default(self):
        assert normalize_artist_name("John Coltrane") == "john coltrane"
        assert normalize_artist_name("RADIOHEAD") == "radiohead"

    def test_preserve_case(self):
        assert normalize_artist_name("John Coltrane", lowercase=False) == "John Coltrane"

    def test_whitespace_collapse(self):
        assert normalize_artist_name("Bill   Evans") == "bill evans"

    def test_empty_input(self):
        assert normalize_artist_name("") == ""
        assert normalize_artist_name(None) == ""


class TestEnsembleHandling:
    """Test ensemble suffix stripping."""

    def test_trio_suffix(self):
        assert normalize_artist_name("Bill Evans Trio") == "bill evans"
        assert normalize_artist_name("The Bill Evans Trio") == "bill evans"

    def test_quartet_suffix(self):
        assert normalize_artist_name("Modern Jazz Quartet") == "modern jazz"

    def test_all_ensemble_suffixes(self):
        assert normalize_artist_name("Artist Quintet") == "artist"
        assert normalize_artist_name("Artist Sextet") == "artist"
        assert normalize_artist_name("Artist Septet") == "artist"
        assert normalize_artist_name("Artist Octet") == "artist"

    def test_preserve_mid_name_ensemble(self):
        """Don't strip ensemble terms in middle of name."""
        result = normalize_artist_name("Art Ensemble of Chicago")
        assert result == "art ensemble of chicago"

    def test_disable_ensemble_stripping(self):
        assert normalize_artist_name("Bill Evans Trio", strip_ensemble=False) == "bill evans trio"


class TestCollaborationHandling:
    """Test collaboration marker handling."""

    def test_featuring_variants(self):
        assert normalize_artist_name("John Coltrane feat. Cannonball Adderley") == "john coltrane"
        assert normalize_artist_name("Artist ft. Someone") == "artist"

    def test_with_marker(self):
        assert normalize_artist_name("Mount Eerie with Julie Doiron") == "mount eerie"

    def test_ampersand_split(self):
        result = normalize_artist_name("Pink Siifu & Fly Anakin")
        assert result == "pink siifu"

    def test_preserve_band_names(self):
        """Don't split band names with '& The'."""
        assert normalize_artist_name("Echo & The Bunnymen") == "echo & the bunnymen"
        assert normalize_artist_name("Sly & The Family Stone") == "sly & the family stone"

    def test_disable_collaboration_stripping(self):
        result = normalize_artist_name("Artist feat. Someone", strip_collaborations=False)
        assert "feat" in result


class TestUnicodeHandling:
    """Test Unicode normalization and diacritic removal."""

    def test_remove_diacritics(self):
        assert normalize_artist_name("Sigur Rós") == "sigur ros"
        assert normalize_artist_name("Björk") == "bjork"

    def test_disable_unicode_normalization(self):
        result = normalize_artist_name("Sigur Rós", normalize_unicode=False, lowercase=False)
        assert "ó" in result


class TestCombinedFeatures:
    """Test combinations of normalization features."""

    def test_collaboration_plus_ensemble(self):
        assert normalize_artist_name("Bill Evans Trio feat. Cannonball Adderley") == "bill evans"
