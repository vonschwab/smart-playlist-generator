"""Unit tests for genre mode functionality."""
import random
import pytest


class TestGenreSeedDeterminism:
    """Test deterministic seed selection logic."""

    def test_random_with_seed_is_deterministic(self):
        """Verify Random(seed).sample() produces deterministic results."""
        import random

        tracks = [{'id': i, 'name': f'Track {i}'} for i in range(20)]

        # Use same seed twice
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        sample1 = rng1.sample(tracks, 4)
        sample2 = rng2.sample(tracks, 4)

        # Should get identical samples
        assert [t['id'] for t in sample1] == [t['id'] for t in sample2]

    def test_different_seeds_produce_different_samples(self):
        """Verify different seeds produce different results."""
        import random

        tracks = [{'id': i, 'name': f'Track {i}'} for i in range(20)]

        rng1 = random.Random(42)
        rng2 = random.Random(43)

        sample1 = rng1.sample(tracks, 4)
        sample2 = rng2.sample(tracks, 4)

        # Should get different samples (very high probability)
        assert [t['id'] for t in sample1] != [t['id'] for t in sample2]


class TestGenreNormalization:
    """Test genre normalization in suggestions."""

    def test_normalization_lowercases_and_strips(self):
        """Genre normalization should lowercase and strip whitespace."""
        from src.genre_normalization import normalize_genre_token

        assert normalize_genre_token("New Age") == "new age"
        assert normalize_genre_token(" PROGRESSIVE ROCK ") == "progressive rock"
        assert normalize_genre_token("Hip-Hop") == "hip-hop"
