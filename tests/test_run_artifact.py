"""
Tests for run artifact schema and writer.
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.eval.run_artifact import (
    EdgeRecord,
    ExclusionCounters,
    RunArtifact,
    RunArtifactWriter,
    SettingsSnapshot,
    SummaryMetrics,
    TrackRecord,
    append_to_consolidated_csv,
    compute_summary_metrics,
    generate_run_id,
)


@pytest.fixture
def sample_settings():
    """Create a sample SettingsSnapshot."""
    return SettingsSnapshot(
        run_id="20241215_120000_dynamic_abc12345",
        timestamp="20241215_120000",
        mode="dynamic",
        pipeline="ds",
        seed_track_id="abc12345678901234567890123456789",
        seed_artist="test artist",
        seed_title="test title",
        playlist_length=30,
        random_seed=42,
        sonic_weight=0.67,
        genre_weight=0.33,
        genre_method="ensemble",
        min_genre_similarity=0.2,
        genre_gate_mode="hard",
        transition_floor=0.30,
        transition_gamma=1.0,
        hard_floor=True,
        center_transitions=False,
        alpha=0.55,
        beta=0.45,
        gamma=0.04,
        alpha_schedule="arc",
        similarity_floor=0.30,
        max_pool_size=1200,
        max_artist_fraction=0.125,
        min_gap=6,
        sonic_variant="raw",
    )


@pytest.fixture
def sample_tracks():
    """Create sample TrackRecords."""
    return [
        TrackRecord(
            position=0,
            track_id="track_001",
            artist_key="artist a",
            artist_name="Artist A",
            title="Song One",
            duration_ms=180000,
            seed_sim=1.0,
            genres=["rock", "alternative"],
        ),
        TrackRecord(
            position=1,
            track_id="track_002",
            artist_key="artist b",
            artist_name="Artist B",
            title="Song Two",
            duration_ms=200000,
            seed_sim=0.85,
            genres=["rock"],
        ),
        TrackRecord(
            position=2,
            track_id="track_003",
            artist_key="artist c",
            artist_name="Artist C",
            title="Song Three",
            duration_ms=220000,
            seed_sim=0.75,
            genres=["indie rock"],
        ),
    ]


@pytest.fixture
def sample_edges():
    """Create sample EdgeRecords."""
    return [
        EdgeRecord(
            position=0,
            prev_track_id="track_001",
            next_track_id="track_002",
            prev_artist="artist a",
            next_artist="artist b",
            sonic_sim=0.82,
            genre_sim=0.75,
            hybrid_sim=0.78,
            transition_sim=0.72,
            transition_raw=0.70,
            transition_centered=0.68,
            below_floor=False,
            same_artist=False,
        ),
        EdgeRecord(
            position=1,
            prev_track_id="track_002",
            next_track_id="track_003",
            prev_artist="artist b",
            next_artist="artist c",
            sonic_sim=0.65,
            genre_sim=0.60,
            hybrid_sim=0.62,
            transition_sim=0.55,
            transition_raw=0.52,
            transition_centered=0.50,
            below_floor=False,
            same_artist=False,
        ),
    ]


@pytest.fixture
def sample_exclusions():
    """Create sample ExclusionCounters."""
    return ExclusionCounters(
        below_similarity_floor=500,
        genre_gate_rejected=50,
        artist_cap_rejected=100,
        adjacency_rejected=10,
        min_gap_rejected=20,
        transition_floor_rejected=5,
        total_candidates_considered=10000,
    )


@pytest.fixture
def sample_artifact(sample_settings, sample_tracks, sample_edges, sample_exclusions):
    """Create a complete RunArtifact."""
    metrics = compute_summary_metrics(sample_tracks, sample_edges, 0.2)
    return RunArtifact(
        settings=sample_settings,
        tracks=sample_tracks,
        edges=sample_edges,
        exclusions=sample_exclusions,
        metrics=metrics,
    )


class TestSettingsSnapshot:
    def test_to_dict(self, sample_settings):
        """Test settings serialization to dict."""
        d = sample_settings.to_dict()
        assert d["mode"] == "dynamic"
        assert d["sonic_weight"] == 0.67
        assert d["genre_weight"] == 0.33
        assert d["min_genre_similarity"] == 0.2
        assert d["genre_gate_mode"] == "hard"
        assert d["alpha"] == 0.55
        assert d["beta"] == 0.45

    def test_all_fields_serialized(self, sample_settings):
        """Ensure all fields are in the dict."""
        d = sample_settings.to_dict()
        expected_fields = [
            "run_id", "timestamp", "mode", "pipeline", "seed_track_id",
            "seed_artist", "seed_title", "playlist_length", "random_seed",
            "sonic_weight", "genre_weight", "genre_method", "min_genre_similarity",
            "genre_gate_mode", "transition_floor", "transition_gamma",
            "hard_floor", "center_transitions", "alpha", "beta", "gamma",
            "alpha_schedule", "similarity_floor", "max_pool_size",
            "max_artist_fraction", "min_gap", "sonic_variant",
        ]
        for field in expected_fields:
            assert field in d, f"Missing field: {field}"


class TestTrackRecord:
    def test_to_dict(self, sample_tracks):
        """Test track serialization."""
        track = sample_tracks[0]
        d = track.to_dict()
        assert d["position"] == 0
        assert d["track_id"] == "track_001"
        assert d["seed_sim"] == 1.0
        assert d["genres"] == "rock,alternative"


class TestEdgeRecord:
    def test_to_dict(self, sample_edges):
        """Test edge serialization."""
        edge = sample_edges[0]
        d = edge.to_dict()
        assert d["position"] == 0
        assert d["sonic_sim"] == 0.82
        assert d["genre_sim"] == 0.75
        assert d["below_floor"] is False


class TestComputeSummaryMetrics:
    def test_basic_metrics(self, sample_tracks, sample_edges):
        """Test summary metric computation."""
        metrics = compute_summary_metrics(sample_tracks, sample_edges, 0.2)

        # Check edge metrics are computed
        assert metrics.edge_hybrid_mean > 0
        assert metrics.edge_sonic_mean > 0
        assert metrics.edge_genre_mean > 0

        # Check diversity metrics
        assert metrics.unique_artists == 3
        assert metrics.max_artist_percentage == pytest.approx(1 / 3)

        # No constraint violations in sample
        assert metrics.below_floor_count == 0

    def test_empty_edges(self, sample_tracks):
        """Test with no edges."""
        metrics = compute_summary_metrics(sample_tracks, [], 0.2)
        assert metrics.edge_hybrid_mean == 0.0
        # With no edges, diversity is not computed from tracks
        # (current implementation only computes from tracks if edges exist)
        assert metrics.unique_artists == 0


class TestRunArtifact:
    def test_to_dict(self, sample_artifact):
        """Test full artifact serialization."""
        d = sample_artifact.to_dict()
        assert "settings" in d
        assert "tracks" in d
        assert "edges" in d
        assert "exclusions" in d
        assert "metrics" in d

        assert len(d["tracks"]) == 3
        assert len(d["edges"]) == 2

    def test_json_serializable(self, sample_artifact):
        """Test that artifact can be JSON serialized."""
        d = sample_artifact.to_dict()
        json_str = json.dumps(d, indent=2)
        assert len(json_str) > 0

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["settings"]["mode"] == "dynamic"


class TestRunArtifactWriter:
    def test_write_disabled(self, sample_artifact):
        """Test writer when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RunArtifactWriter(Path(tmpdir), enabled=False)
            result = writer.write(sample_artifact)
            assert result is None

    def test_write_json(self, sample_artifact):
        """Test JSON file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RunArtifactWriter(Path(tmpdir), enabled=True)
            paths = writer.write(sample_artifact)

            assert paths is not None
            assert "json" in paths
            assert paths["json"].exists()

            # Verify JSON content
            with open(paths["json"]) as f:
                data = json.load(f)
            assert data["settings"]["mode"] == "dynamic"
            assert len(data["tracks"]) == 3

    def test_write_csvs(self, sample_artifact):
        """Test CSV file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RunArtifactWriter(Path(tmpdir), enabled=True)
            paths = writer.write(sample_artifact)

            assert "tracks_csv" in paths
            assert "edges_csv" in paths
            assert "summary_csv" in paths

            # Verify tracks CSV has header + 3 rows
            with open(paths["tracks_csv"]) as f:
                lines = f.readlines()
            assert len(lines) == 4  # header + 3 tracks

    def test_creates_output_dir(self, sample_artifact):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"
            writer = RunArtifactWriter(nested_dir, enabled=True)
            paths = writer.write(sample_artifact)

            assert nested_dir.exists()
            assert paths["json"].exists()


class TestAppendToConsolidatedCSV:
    def test_creates_file_with_header(self, sample_artifact):
        """Test creating new consolidated CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "consolidated.csv"
            append_to_consolidated_csv(csv_path, sample_artifact)

            assert csv_path.exists()
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 row

    def test_appends_without_header(self, sample_artifact):
        """Test appending to existing consolidated CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "consolidated.csv"

            # Write twice
            append_to_consolidated_csv(csv_path, sample_artifact)
            append_to_consolidated_csv(csv_path, sample_artifact)

            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows


class TestGenerateRunId:
    def test_basic_format(self):
        """Test run ID generation format."""
        run_id = generate_run_id("abc123456789", "dynamic", "20241215_120000")
        assert "20241215_120000" in run_id
        assert "dynamic" in run_id
        assert "abc12345" in run_id  # truncated seed

    def test_short_seed(self):
        """Test with short seed ID."""
        run_id = generate_run_id("abc", "narrow")
        assert "narrow" in run_id
        assert "abc" in run_id


class TestSchemaStability:
    """Tests to catch schema changes that might break compatibility."""

    def test_settings_field_count(self, sample_settings):
        """Track expected field count for schema stability."""
        d = sample_settings.to_dict()
        # Update this if you intentionally add/remove fields
        assert len(d) == 27

    def test_track_field_count(self, sample_tracks):
        """Track expected field count for tracks."""
        d = sample_tracks[0].to_dict()
        assert len(d) == 8

    def test_edge_field_count(self, sample_edges):
        """Track expected field count for edges."""
        d = sample_edges[0].to_dict()
        assert len(d) == 13

    def test_exclusion_field_count(self, sample_exclusions):
        """Track expected field count for exclusions."""
        d = sample_exclusions.to_dict()
        assert len(d) == 7

    def test_metrics_field_count(self, sample_tracks, sample_edges):
        """Track expected field count for metrics."""
        metrics = compute_summary_metrics(sample_tracks, sample_edges, 0.2)
        d = metrics.to_dict()
        assert len(d) == 26
