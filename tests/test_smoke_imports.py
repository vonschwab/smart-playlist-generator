"""Smoke tests for module imports.

These tests verify that key modules can be imported without errors.
This catches missing dependencies, syntax errors, and circular imports.
"""

import pytest


class TestCoreImports:
    """Test that core modules are importable."""

    def test_config_loader(self):
        from src.config_loader import Config
        assert Config is not None

    def test_local_library_client(self):
        from src.local_library_client import LocalLibraryClient
        assert LocalLibraryClient is not None

    def test_playlist_generator(self):
        from src.playlist_generator import PlaylistGenerator
        assert PlaylistGenerator is not None

    def test_m3u_exporter(self):
        from src.m3u_exporter import M3UExporter
        assert M3UExporter is not None

    def test_logging_config(self):
        from src.logging_config import setup_logging
        assert setup_logging is not None


class TestPlaylistModuleImports:
    """Test that playlist submodule is importable."""

    def test_pipeline(self):
        from src.playlist.pipeline import DSPipelineResult
        assert DSPipelineResult is not None

    def test_constructor(self):
        from src.playlist.constructor import construct_playlist
        assert construct_playlist is not None

    def test_config(self):
        from src.playlist.config import DSPipelineConfig
        assert DSPipelineConfig is not None

    def test_filtering(self):
        from src.playlist.filtering import apply_filters
        assert apply_filters is not None

    def test_ordering(self):
        from src.playlist import ordering
        assert ordering is not None


class TestFeatureModuleImports:
    """Test that features submodule is importable."""

    def test_artifacts(self):
        from src.features.artifacts import load_artifact_bundle
        assert load_artifact_bundle is not None

    def test_beat3tower_types(self):
        from src.features.beat3tower_types import Beat3TowerFeatures
        assert Beat3TowerFeatures is not None

    def test_beat3tower_normalizer(self):
        from src.features.beat3tower_normalizer import Beat3TowerNormalizer
        assert Beat3TowerNormalizer is not None


class TestSimilarityModuleImports:
    """Test that similarity submodule is importable."""

    def test_sonic_variant(self):
        from src.similarity.sonic_variant import resolve_sonic_variant
        assert resolve_sonic_variant is not None

    def test_hybrid(self):
        from src.similarity.hybrid import HybridEmbeddingModel
        assert HybridEmbeddingModel is not None


class TestGenreModuleImports:
    """Test that genre submodule is importable."""

    def test_genre_normalization(self):
        from src.genre_normalization import normalize_genre_list
        assert normalize_genre_list is not None

    def test_genre_normalize(self):
        from src.genre.normalize import normalize_and_split_genre
        assert normalize_and_split_genre is not None


class TestUtilityImports:
    """Test that utility modules are importable."""

    def test_title_dedupe(self):
        from src.title_dedupe import TitleDeduplicator
        assert TitleDeduplicator is not None

    def test_string_utils(self):
        from src.string_utils import normalize_string
        assert normalize_string is not None

    def test_artist_utils(self):
        from src.artist_utils import split_collaborators
        assert split_collaborators is not None
