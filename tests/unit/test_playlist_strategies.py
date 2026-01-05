"""Unit tests for playlist generation strategies.

Tests strategy pattern infrastructure extracted from playlist_generator.py (Phase 4.1).

Coverage:
- PlaylistRequest dataclass
- PlaylistResult dataclass
- PlaylistGenerationStrategy base class
- Strategy pattern implementation
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.strategies.base_strategy import (
    PlaylistGenerationStrategy,
    PlaylistRequest,
    PlaylistResult,
)
from src.playlist.playlist_factory import PlaylistFactory


# =============================================================================
# Mock Strategy for Testing
# =============================================================================

class MockStrategy(PlaylistGenerationStrategy):
    """Mock strategy for testing base functionality."""

    def __init__(self, config=None, mode="mock"):
        super().__init__(config or {})
        self.mode = mode
        self.execute_called = False

    def can_handle(self, request: PlaylistRequest) -> bool:
        """Handle requests with matching mode."""
        return request.mode == self.mode

    def execute(self, request: PlaylistRequest) -> PlaylistResult:
        """Execute mock playlist generation."""
        self.execute_called = True

        if request.num_tracks <= 0:
            return self._create_failure_result(request, "Invalid track count")

        # Generate mock playlist
        track_ids = [f"track_{i:03d}" for i in range(request.num_tracks)]

        return PlaylistResult(
            track_ids=track_ids,
            name=f"Mock Playlist ({request.mode})",
            mode=request.mode,
            metrics={"diversity": 0.85},
        )


# =============================================================================
# PlaylistRequest Tests
# =============================================================================

class TestPlaylistRequest:
    """Test PlaylistRequest dataclass."""

    def test_create_basic_request(self):
        """Test creating basic playlist request."""
        request = PlaylistRequest(
            mode="dynamic",
            num_tracks=30,
            config={"some": "config"},
        )

        assert request.mode == "dynamic"
        assert request.num_tracks == 30
        assert request.config == {"some": "config"}
        assert request.artist is None
        assert request.genre is None

    def test_create_artist_request(self):
        """Test creating artist-based request."""
        request = PlaylistRequest(
            mode="artist",
            num_tracks=30,
            config={},
            artist="Bill Evans",
        )

        assert request.mode == "artist"
        assert request.artist == "Bill Evans"

    def test_create_genre_request(self):
        """Test creating genre-based request."""
        request = PlaylistRequest(
            mode="genre",
            num_tracks=30,
            config={},
            genre="jazz",
        )

        assert request.mode == "genre"
        assert request.genre == "jazz"

    def test_request_with_overrides(self):
        """Test request with configuration overrides."""
        request = PlaylistRequest(
            mode="dynamic",
            num_tracks=30,
            config={},
            overrides={"transition_floor": 0.30},
        )

        assert request.overrides == {"transition_floor": 0.30}


# =============================================================================
# PlaylistResult Tests
# =============================================================================

class TestPlaylistResult:
    """Test PlaylistResult dataclass."""

    def test_create_basic_result(self):
        """Test creating basic playlist result."""
        result = PlaylistResult(
            track_ids=["track_001", "track_002", "track_003"],
            name="My Playlist",
            mode="dynamic",
        )

        assert len(result.track_ids) == 3
        assert result.name == "My Playlist"
        assert result.mode == "dynamic"
        assert result.success is True
        assert result.failure_reason is None

    def test_result_with_metrics(self):
        """Test result with quality metrics."""
        result = PlaylistResult(
            track_ids=["track_001"],
            name="Test",
            mode="dynamic",
            metrics={
                "diversity": 0.85,
                "coherence": 0.75,
            },
        )

        assert result.metrics["diversity"] == 0.85
        assert result.metrics["coherence"] == 0.75

    def test_result_with_diagnostics(self):
        """Test result with diagnostics."""
        result = PlaylistResult(
            track_ids=["track_001"],
            name="Test",
            mode="dynamic",
            diagnostics={
                "pool_size": 1000,
                "candidates_filtered": 200,
            },
        )

        assert result.diagnostics["pool_size"] == 1000

    def test_failure_result(self):
        """Test creating failure result."""
        result = PlaylistResult(
            track_ids=[],
            name="",
            mode="dynamic",
            success=False,
            failure_reason="Insufficient tracks",
        )

        assert result.success is False
        assert result.failure_reason == "Insufficient tracks"
        assert len(result.track_ids) == 0


# =============================================================================
# PlaylistGenerationStrategy Tests
# =============================================================================

class TestPlaylistGenerationStrategy:
    """Test PlaylistGenerationStrategy base class."""

    def test_strategy_can_handle(self):
        """Test strategy can_handle method."""
        strategy = MockStrategy(mode="mock")

        request = PlaylistRequest(
            mode="mock",
            num_tracks=30,
            config={},
        )

        assert strategy.can_handle(request) is True

    def test_strategy_cannot_handle(self):
        """Test strategy rejects wrong mode."""
        strategy = MockStrategy(mode="mock")

        request = PlaylistRequest(
            mode="different",
            num_tracks=30,
            config={},
        )

        assert strategy.can_handle(request) is False

    def test_strategy_execute(self):
        """Test strategy execute method."""
        strategy = MockStrategy(mode="mock")

        request = PlaylistRequest(
            mode="mock",
            num_tracks=30,
            config={},
        )

        result = strategy.execute(request)

        assert strategy.execute_called is True
        assert result.success is True
        assert len(result.track_ids) == 30
        assert result.mode == "mock"

    def test_strategy_create_failure_result(self):
        """Test strategy failure result creation."""
        strategy = MockStrategy(mode="mock")

        request = PlaylistRequest(
            mode="mock",
            num_tracks=0,  # Invalid
            config={},
        )

        result = strategy.execute(request)

        assert result.success is False
        assert result.failure_reason == "Invalid track count"
        assert len(result.track_ids) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestStrategyIntegration:
    """Test strategy pattern integration."""

    def test_multiple_strategies(self):
        """Test multiple strategies with different modes."""
        strategy_a = MockStrategy(mode="mode_a")
        strategy_b = MockStrategy(mode="mode_b")

        request_a = PlaylistRequest(mode="mode_a", num_tracks=10, config={})
        request_b = PlaylistRequest(mode="mode_b", num_tracks=20, config={})

        # Strategy A handles request A
        assert strategy_a.can_handle(request_a) is True
        assert strategy_a.can_handle(request_b) is False

        # Strategy B handles request B
        assert strategy_b.can_handle(request_a) is False
        assert strategy_b.can_handle(request_b) is True

        # Execute both
        result_a = strategy_a.execute(request_a)
        result_b = strategy_b.execute(request_b)

        assert len(result_a.track_ids) == 10
        assert len(result_b.track_ids) == 20

    def test_strategy_selection(self):
        """Test selecting correct strategy for request."""
        strategies = [
            MockStrategy(mode="artist"),
            MockStrategy(mode="genre"),
            MockStrategy(mode="batch"),
        ]

        request = PlaylistRequest(mode="genre", num_tracks=30, config={})

        # Find matching strategy
        selected = None
        for strategy in strategies:
            if strategy.can_handle(request):
                selected = strategy
                break

        assert selected is not None
        assert selected.mode == "genre"

        # Execute
        result = selected.execute(request)
        assert result.mode == "genre"


# =============================================================================
# PlaylistFactory Tests
# =============================================================================

class TestPlaylistFactory:
    """Test PlaylistFactory."""

    def test_create_factory(self):
        """Test creating playlist factory."""
        factory = PlaylistFactory(config={})

        assert factory.config == {}
        assert len(factory.strategies) == 0

    def test_register_strategy(self):
        """Test registering strategies."""
        factory = PlaylistFactory(config={})

        strategy = MockStrategy(mode="artist")
        factory.register_strategy(strategy)

        assert len(factory.strategies) == 1

    def test_create_with_registered_strategy(self):
        """Test creating playlist with registered strategy."""
        factory = PlaylistFactory(config={})
        factory.register_strategy(MockStrategy(mode="artist"))

        request = PlaylistRequest(
            mode="artist",
            num_tracks=30,
            config={},
            artist="Bill Evans",
        )

        result = factory.create(request)

        assert result.success is True
        assert len(result.track_ids) == 30
        assert result.mode == "artist"

    def test_create_with_no_matching_strategy(self):
        """Test creating playlist when no strategy matches."""
        factory = PlaylistFactory(config={})
        factory.register_strategy(MockStrategy(mode="artist"))

        request = PlaylistRequest(
            mode="genre",  # No strategy for this mode
            num_tracks=30,
            config={},
        )

        with pytest.raises(ValueError, match="No strategy can handle"):
            factory.create(request)

    def test_create_with_multiple_strategies(self):
        """Test factory with multiple registered strategies."""
        factory = PlaylistFactory(config={})
        factory.register_strategy(MockStrategy(mode="artist"))
        factory.register_strategy(MockStrategy(mode="genre"))
        factory.register_strategy(MockStrategy(mode="batch"))

        # Test artist mode
        request_artist = PlaylistRequest(mode="artist", num_tracks=30, config={})
        result_artist = factory.create(request_artist)
        assert result_artist.mode == "artist"

        # Test genre mode
        request_genre = PlaylistRequest(mode="genre", num_tracks=25, config={})
        result_genre = factory.create(request_genre)
        assert result_genre.mode == "genre"

        # Test batch mode
        request_batch = PlaylistRequest(mode="batch", num_tracks=20, config={})
        result_batch = factory.create(request_batch)
        assert result_batch.mode == "batch"

    def test_get_supported_modes(self):
        """Test getting supported modes."""
        factory = PlaylistFactory(config={})
        factory.register_strategy(MockStrategy(mode="artist"))
        factory.register_strategy(MockStrategy(mode="genre"))

        modes = factory.get_supported_modes()

        assert "artist" in modes
        assert "genre" in modes
        assert len(modes) == 2
