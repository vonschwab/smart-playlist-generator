"""
Regression test for dial grid tuning.

Verifies that tuning dials (sonic_weight, genre_weight, min_genre_similarity, etc.)
actually affect playlist generation and metrics.

This test failed before the fix (all dials produced identical playlists).
"""
import pytest
from pathlib import Path

from src.playlist.pipeline import generate_playlist_ds
from src.features.artifacts import load_artifact_bundle


# Skip if artifact not available
artifact_path = Path("experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz")
ARTIFACT_AVAILABLE = artifact_path.exists()


@pytest.mark.skipif(not ARTIFACT_AVAILABLE, reason="Artifact not available")
class TestDialGridTuning:
    """Test that tuning dials produce different results."""

    @pytest.fixture(scope="class")
    def bundle(self):
        """Load artifact bundle once for all tests."""
        return load_artifact_bundle(artifact_path)

    @pytest.fixture(scope="class")
    def seed_id(self, bundle):
        """Pick a seed track."""
        if len(bundle.track_ids) == 0:
            pytest.skip("No tracks in artifact")
        return str(bundle.track_ids[0])

    def test_sonic_weight_changes_result(self, seed_id):
        """Test that sonic_weight affects playlist generation."""
        # Run with sonic_weight=0.55 (genre-heavy)
        result_a = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.55,
            genre_weight=0.45,
        )

        # Run with sonic_weight=0.75 (sonic-heavy)
        result_b = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.75,
            genre_weight=0.25,
        )

        # They should differ (or at least one parameter should be logged differently)
        assert result_a.track_ids != result_b.track_ids, \
            f"sonic_weight=0.55 and 0.75 produced identical playlists: {result_a.track_ids}"

    def test_genre_weight_changes_result(self, seed_id):
        """Test that genre_weight affects playlist generation (complement to sonic_weight test)."""
        # Run with sonic-heavy (low genre weight)
        result_a = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.8,
            genre_weight=0.2,
        )

        # Run with genre-heavy (low sonic weight)
        result_b = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.3,
            genre_weight=0.7,
        )

        # They should differ
        assert result_a.track_ids != result_b.track_ids, \
            f"sonic_weight=0.8 and 0.3 (inverse genre weights) produced identical playlists"

    def test_transition_strictness_changes_result(self, seed_id):
        """Test that transition strictness override is applied.

        Note: The transition_floor may only affect repair, not initial construction,
        so this test just verifies the override is applied to config, not that it
        necessarily changes the playlist.
        """
        result_a = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            overrides={"construct": {}},
        )

        result_b = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            overrides={"construct": {"transition_floor": 0.6, "hard_floor": True}},
        )

        # Verify override was applied to config
        params_a = result_a.params_requested.get("construct", {})
        params_b = result_b.params_requested.get("construct", {})

        assert params_a.get("transition_floor") != params_b.get("transition_floor"), \
            "Override not applied to config"
        assert params_b.get("transition_floor") == 0.6, \
            f"Expected transition_floor=0.6, got {params_b.get('transition_floor')}"

    def test_min_genre_similarity_affects_pool(self, seed_id):
        """Test that min_genre_similarity filters candidates (NEW: now implemented)."""
        # Lenient genre gating
        result_lenient = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.60,
            genre_weight=0.40,
            min_genre_similarity=0.15,
        )

        # Strict genre gating
        result_strict = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.60,
            genre_weight=0.40,
            min_genre_similarity=0.50,
        )

        # Should differ in pool size or exclusion counts
        lenient_pool = result_lenient.stats.get("candidate_pool", {})
        strict_pool = result_strict.stats.get("candidate_pool", {})

        # Strict should exclude more candidates by genre
        assert strict_pool.get("below_genre_similarity", 0) >= lenient_pool.get("below_genre_similarity", 0), \
            "Strict genre gate should exclude at least as many candidates as lenient"

        # Playlists should differ
        assert result_lenient.track_ids != result_strict.track_ids, \
            "min_genre_similarity=0.15 and 0.50 produced identical playlists"

    def test_genre_method_affects_selection(self, seed_id):
        """Test that genre_method produces different results (NEW: now implemented)."""
        # Cosine similarity
        result_cosine = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.60,
            genre_weight=0.40,
            min_genre_similarity=0.25,
            genre_method="cosine",
        )

        # Weighted Jaccard
        result_jaccard = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.60,
            genre_weight=0.40,
            min_genre_similarity=0.25,
            genre_method="weighted_jaccard",
        )

        # Should produce different results
        assert result_cosine.track_ids != result_jaccard.track_ids, \
            "genre_method=cosine and weighted_jaccard produced identical playlists"

    def test_transition_strictness_binds(self, seed_id):
        """Test that transition_strictness now affects rejection counts and playlists."""
        # Baseline (low floor)
        result_baseline = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            overrides={"construct": {}},  # Use default floor=0.3
        )

        # Strictish (high floor=0.85)
        result_strictish = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            overrides={"construct": {"transition_floor": 0.85, "hard_floor": True}},
        )

        # Check rejection counts - strict should reject more transitions
        baseline_rejected = result_baseline.stats.get("playlist", {}).get("below_floor_count", 0)
        strictish_rejected = result_strictish.stats.get("playlist", {}).get("below_floor_count", 0)

        # Playlists might differ, or at least rejection counts should differ
        assert result_baseline.track_ids != result_strictish.track_ids or \
               baseline_rejected != strictish_rejected, \
            "transition_strictness baseline vs strictish produced identical results"

    def test_extreme_dials_produce_extreme_metrics(self, seed_id):
        """Test that extreme dial values produce noticeable differences in metrics."""
        # Genre-heavy (low sonic weight), strict genre gate
        result_genre_heavy = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.40,
            genre_weight=0.60,
            min_genre_similarity=0.40,
        )

        # Sonic-heavy (high sonic weight, lenient genre)
        result_sonic_heavy = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=25,
            mode="dynamic",
            random_seed=0,
            sonic_weight=0.85,
            genre_weight=0.15,
            min_genre_similarity=0.10,
        )

        # They must differ
        assert result_genre_heavy.track_ids != result_sonic_heavy.track_ids, \
            "Extreme dial differences still produced identical results"

        # Check pool stats - genre-heavy should have more genre exclusions
        gh_pool = result_genre_heavy.stats.get("candidate_pool", {})
        sh_pool = result_sonic_heavy.stats.get("candidate_pool", {})

        assert gh_pool.get("below_genre_similarity", 0) > sh_pool.get("below_genre_similarity", 0), \
            "Genre-heavy config should exclude more tracks by genre than sonic-heavy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
