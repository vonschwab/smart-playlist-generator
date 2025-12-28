"""
Playlist Generator - Core logic for creating AI-powered playlists
"""
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter, defaultdict
import random
import logging
import time
import os
import numpy as np
from .artist_utils import extract_primary_artist
from .similarity_calculator import SimilarityCalculator
from .string_utils import normalize_genre, normalize_song_title
from .string_utils import normalize_match_string
from .title_dedupe import TitleDedupeTracker
from src.features.artifacts import load_artifact_bundle
from src.similarity.hybrid import transition_similarity_end_to_start
from src.similarity.sonic_variant import compute_sonic_variant_norm, get_variant_from_env, resolve_sonic_variant
from src.playlist.ds_pipeline_runner import DsRunResult, generate_playlist_ds as run_ds_pipeline
# Phase 2: Import utilities from refactored module
from src.playlist import utils
# Phase 3: Import filtering from refactored module
from src.playlist import filtering

# Phase 4: Import history_analyzer from refactored module
from src.playlist import history_analyzer

# Phase 5: Import scoring from refactored module
from src.playlist import scoring

# Phase 6: Import diversity from refactored module
from src.playlist import diversity

# Phase 7: Import ordering from refactored module
from src.playlist import ordering

# Phase 8: Import candidate_generator from refactored module
from src.playlist import candidate_generator

# Phase 9: Import reporter from refactored module
from src.playlist import reporter

logger = logging.getLogger(__name__)


# Phase 2: Delegate to refactored utils module (backward compatibility)
def sanitize_for_logging(text: str) -> str:
    """Sanitize text for Windows console logging by replacing unencodable characters"""
    return utils.sanitize_for_logging(text)


def safe_get_artist(track: Dict[str, Any], lowercase: bool = True) -> str:
    """
    Safely get artist name from a track dictionary with None-safe fallback

    Args:
        track: Track dictionary
        lowercase: Whether to convert to lowercase

    Returns:
        Artist name (empty string if None or missing)
    """
    return utils.safe_get_artist(track, lowercase=lowercase)


def _convert_seconds_to_ms(seconds: Optional[int]) -> int:
    """Helper to convert seconds to milliseconds with None safety."""
    return utils.convert_seconds_to_ms(seconds)



class PlaylistGenerator:
    """Generates playlists based on listening history and similarity"""

    def __init__(self, library_client, config, lastfm_client=None, track_matcher=None, metadata_client=None):
        self.library = library_client
        self.config = config
        self.lastfm = lastfm_client
        self.matcher = track_matcher
        self.metadata = metadata_client  # Metadata database for genre lookups
        self.pipeline_override: Optional[str] = None
        self._logged_ds_artifact_warning = False
        self._last_ds_report: Optional[Dict[str, Any]] = None
        # Expose similarity calculator if provided by the library client; fall back to new instance
        self.similarity_calc = getattr(library_client, 'similarity_calc', None)
        if self.similarity_calc is None:
            db_path = None
            try:
                db_path = self.config.get('library', {}).get('database_path', 'data/metadata.db')
            except Exception as e:
                logger.debug(f"Falling back to default metadata path after config read failure: {e}")
                db_path = 'data/metadata.db'
            self.similarity_calc = SimilarityCalculator(db_path=db_path, config=self.config)
        self.genre_similarity_cache = {}  # Legacy cache placeholder
        self._warn_if_ds_artifact_missing()
        # Capture resolved variant once; env can override unless CLI sets explicit
        sonic_cfg = self.config.get('playlists', 'sonic', default={}) or {}
        self.sonic_variant = resolve_sonic_variant(config_variant=sonic_cfg.get("sim_variant"))

    def _get_pipeline_choice(self, pipeline_override: Optional[str] = None) -> str:
        return "ds"

    def _warn_if_ds_artifact_missing(self) -> None:
        """Log once if DS pipeline is configured but artifacts are missing or unreadable."""
        if self._logged_ds_artifact_warning:
            return
        ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
        artifact_path = ds_cfg.get('artifact_path')
        if not artifact_path:
            return
        from pathlib import Path
        try:
            path = Path(artifact_path)
            if not path.exists():
                logger.error("DS pipeline enabled but artifact missing at %s; DS runs will fail until provided.", artifact_path)
                self._logged_ds_artifact_warning = True
                return
            # Lightweight readability check
            try:
                load_artifact_bundle(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("DS pipeline artifact unreadable (%s); DS runs will fail until fixed.", exc)
            self._logged_ds_artifact_warning = True
        except Exception:
            # Avoid breaking init; warn once
            logger.error("DS pipeline artifact check failed; DS runs will fail until fixed.")
            self._logged_ds_artifact_warning = True

    def _compute_edge_scores_from_artifact(
        self,
        tracks: List[Dict[str, Any]],
        artifact_path: Optional[str],
        transition_floor: Optional[float] = None,
        transition_gamma: Optional[float] = None,
        embedding_random_seed: Optional[int] = None,
        center_transitions: bool = False,
        verbose: bool = False,
        sonic_variant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute per-edge scores (T/S/G) for the final playlist order using artifact matrices.
        """
        # Phase 9: Delegate to reporter module
        return reporter.compute_edge_scores_from_artifact(
            tracks=tracks,
            artifact_path=artifact_path,
            transition_floor=transition_floor,
            transition_gamma=transition_gamma,
            embedding_random_seed=embedding_random_seed,
            center_transitions=center_transitions,
            verbose=verbose,
            sonic_variant=sonic_variant,
            config_sonic_variant=self.sonic_variant,
            last_ds_report=getattr(self, "_last_ds_report", None),
        )

    def _maybe_generate_ds_playlist(
        self,
        seed_track_id: Optional[str],
        target_length: int,
        pipeline_override: Optional[str] = None,
        mode_override: Optional[str] = None,
        seed_artist: Optional[str] = None,
        allowed_track_ids: Optional[List[str]] = None,
        excluded_track_ids: Optional[Set[str]] = None,
        anchor_seed_ids: Optional[List[str]] = None,
        anchor_seed_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run DS pipeline and return ordered track dicts; raise on failure.
        """
        if not seed_track_id:
            raise ValueError("DS pipeline requires a seed track id; none provided.")

        ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
        artifact_path = ds_cfg.get('artifact_path')
        if not artifact_path:
            raise ValueError("DS pipeline artifact_path is not configured.")
        sonic_cfg = self.config.get("playlists", "sonic", default={}) or {}
        sonic_variant_cfg = resolve_sonic_variant(
            explicit_variant=self.sonic_variant,
            config_variant=ds_cfg.get("sonic_variant") or sonic_cfg.get("sim_variant"),
        )
        mode = mode_override or ds_cfg.get('mode', 'dynamic')
        random_seed = ds_cfg.get('random_seed', 0)
        enable_logging = ds_cfg.get('enable_logging', False)

        # Build overrides from new config.yaml tuning structure
        # Priority: explicit overrides > new tuning sections > legacy overrides field
        legacy_overrides = ds_cfg.get('overrides') or {}
        overrides = {
            'scoring': ds_cfg.get('scoring', {}),
            'constraints': ds_cfg.get('constraints', {}),
            'candidate_pool': ds_cfg.get('candidate_pool', {}),
            'repair': ds_cfg.get('repair', {}),
            'tower_weights': ds_cfg.get('tower_weights'),
            'transition_weights': ds_cfg.get('transition_weights'),
            'tower_pca_dims': ds_cfg.get('tower_pca_dims'),
            'embedding': ds_cfg.get('embedding', {}),
        }
        # Merge legacy overrides (if any) on top
        for key in ['candidate', 'construct', 'repair']:
            if key in legacy_overrides:
                overrides[key] = legacy_overrides[key]

        center_transitions = bool(ds_cfg.get("center_transitions", False))
        # Also check constraints section for center_transitions
        if overrides.get('constraints', {}).get('center_transitions'):
            center_transitions = True
        if center_transitions:
            constraints_updates = {**overrides.get("constraints", {}), "center_transitions": True}
            overrides = {**overrides, "constraints": constraints_updates}

        seed_to_use = str(seed_track_id)
        anchor_seed_ids_resolved: Optional[List[str]] = None
        try:
            bundle = load_artifact_bundle(artifact_path)
            if seed_track_id not in bundle.track_id_to_index and seed_artist:
                artist_norm = seed_artist.strip().lower()
                # try track_artists match
                if bundle.track_artists is not None:
                    for idx, artist in enumerate(bundle.track_artists):
                        if str(artist).strip().lower() == artist_norm:
                            seed_to_use = str(bundle.track_ids[idx])
                            logger.info(
                                "DS pipeline seed not found; falling back to artist match %s -> %s",
                                seed_track_id,
                                seed_to_use,
                            )
                            break

            # Resolve anchor seed tracks to bundle track_ids by matching title+artist
            # anchor_seed_ids are Plex rating_keys which don't match bundle track_ids (MD5 hashes)
            if anchor_seed_tracks and len(anchor_seed_tracks) > 1 and bundle.track_artists is not None and bundle.track_titles is not None:
                anchor_seed_ids_resolved = []
                for seed_track in anchor_seed_tracks:
                    seed_title = str(seed_track.get('title', '')).strip().lower()
                    seed_artist_name = str(seed_track.get('artist', '')).strip().lower()
                    if not seed_title or not seed_artist_name:
                        continue
                    # Find matching track in bundle
                    for idx in range(len(bundle.track_artists)):
                        bundle_artist = str(bundle.track_artists[idx]).strip().lower()
                        bundle_title = str(bundle.track_titles[idx]).strip().lower()
                        if bundle_artist == seed_artist_name and bundle_title == seed_title:
                            anchor_seed_ids_resolved.append(str(bundle.track_ids[idx]))
                            break
                if anchor_seed_ids_resolved:
                    logger.info(
                        "Resolved %d/%d anchor seeds by title+artist match",
                        len(anchor_seed_ids_resolved),
                        len(anchor_seed_tracks),
                    )
                else:
                    logger.debug("No anchor seeds resolved by title+artist match")
        except FileNotFoundError:
            raise FileNotFoundError(f"DS pipeline artifact missing at {artifact_path}")
        except Exception as exc:
            raise RuntimeError(f"DS pipeline artifact load failed ({exc})")

        self._last_ds_report = None

        # Read genre similarity configuration
        playlists_cfg = self.config.config.get('playlists', {})
        genre_cfg = playlists_cfg.get('genre_similarity', {})
        genre_enabled = genre_cfg.get('enabled', True)
        min_genre_sim = genre_cfg.get('min_genre_similarity', 0.30) if genre_enabled else None
        if genre_enabled and mode == "narrow":
            min_genre_sim = genre_cfg.get('min_genre_similarity_narrow', min_genre_sim)
        genre_method = genre_cfg.get('method', 'ensemble') if genre_enabled else None
        # Default to 50/50 when genre is enabled
        sonic_weight = genre_cfg.get('sonic_weight', 0.50) if genre_enabled else None
        genre_weight = genre_cfg.get('weight', 0.50) if genre_enabled else None

        if mode == "sonic_only":
            genre_enabled = False
            min_genre_sim = None
            genre_method = None
            genre_weight = 0.0
            sonic_weight = 1.0

        logger.info(
            "Invoking DS pipeline seed=%s mode=%s target_length=%d allowed_ids=%d genre_gate=%s",
            seed_to_use,
            mode,
            target_length,
            len(allowed_track_ids or []),
            f"enabled (min_sim={min_genre_sim})" if genre_enabled else "disabled",
        )
        logger.info(
            "Genre params: sonic_weight=%s genre_weight=%s min_genre_sim=%s method=%s",
            sonic_weight,
            genre_weight,
            min_genre_sim,
            genre_method,
        )
        if anchor_seed_ids_resolved:
            logger.info("Passing %d anchor_seed_ids to pipeline: %s", len(anchor_seed_ids_resolved), anchor_seed_ids_resolved[:5])
        ds_result: DsRunResult = run_ds_pipeline(
            artifact_path=artifact_path,
            seed_track_id=seed_to_use,
            anchor_seed_ids=anchor_seed_ids_resolved,  # Use resolved bundle track_ids, not Plex rating_keys
            mode=mode,
            length=target_length,
            random_seed=random_seed,
            overrides=overrides,
            enable_logging=enable_logging,
            allowed_track_ids=allowed_track_ids,
            excluded_track_ids=excluded_track_ids,
            sonic_variant=sonic_variant_cfg,
            # Genre similarity parameters
            sonic_weight=sonic_weight,
            genre_weight=genre_weight,
            min_genre_similarity=min_genre_sim,
            genre_method=genre_method,
        )

        tracks: List[Dict[str, Any]] = []
        ds_stats = getattr(ds_result, "playlist_stats", {}) or {}
        playlist_stats_only = ds_stats.get("playlist") or {}
        if hasattr(self.library, "get_tracks_by_ids"):
            fetched = getattr(self.library, "get_tracks_by_ids")(ds_result.track_ids)  # type: ignore[attr-defined]
            lookup = {str(t.get("rating_key")): t for t in fetched or []}
            tracks = [lookup[str(tid)] for tid in ds_result.track_ids if str(tid) in lookup]
        else:
            for tid in ds_result.track_ids:
                track = self.library.get_track_by_key(str(tid))
                if track:
                    tracks.append(track)

        if not tracks:
            raise RuntimeError("DS pipeline returned no usable tracks from library lookup.")

        metrics = ds_result.metrics or {}
        actual_len = len(ds_result.track_ids)
        self._last_ds_report = {
            "metrics": metrics,
            "requested_len": target_length,
            "actual_len": actual_len,
            "track_ids_ordered": list(ds_result.track_ids),
            "playlist_stats": ds_stats,
            "artifact_path": artifact_path,
            "transition_floor": playlist_stats_only.get("transition_floor"),
            "transition_gamma": playlist_stats_only.get("transition_gamma"),
            "transition_centered": bool(playlist_stats_only.get("transition_centered")),
            "random_seed": random_seed,
            "sonic_variant": sonic_variant_cfg or os.getenv("SONIC_SIM_VARIANT") or "raw",
        }
        # Log candidate pool stats for diagnostics
        pool_stats = ds_stats.get("candidate_pool", {})
        if pool_stats:
            logger.info(
                "Candidate pool SUMMARY: size=%d distinct_artists=%d eligible_artists=%d",
                pool_stats.get("pool_size", 0),
                pool_stats.get("distinct_artists", 0),
                pool_stats.get("eligible_artists", 0),
            )
            logger.info(
                "Candidate pool EXCLUSIONS: total_candidates=%d below_floor=%d below_genre=%d artist_cap_excluded=%d eligible=%d",
                pool_stats.get("total_candidates_considered", 0),
                pool_stats.get("below_similarity_floor", 0),
                pool_stats.get("below_genre_similarity", 0),
                pool_stats.get("artist_cap_excluded", 0),
                pool_stats.get("eligible_count", 0),
            )

        logger.info(
            "DS pipeline success pipeline=ds seed=%s mode=%s requested_len=%d actual_len=%d distinct_artists=%s max_artist=%s min_transition=%s mean_transition=%s below_floor=%s",
            seed_to_use,
            mode,
            target_length,
            actual_len,
            len((metrics.get('artist_counts') or {}).keys()),
            max((metrics.get('artist_counts') or {}).values()) if metrics.get("artist_counts") else None,
            metrics.get("min_transition"),
            metrics.get("mean_transition"),
            metrics.get("below_floor"),
        )
        # Attach playlist stats for edge logging
        self._last_ds_report["playlist_stats"] = getattr(ds_result, "playlist_stats", {}) if self._last_ds_report else {}
        return tracks

    def _ensure_similarity_calculator(self) -> None:
        """Ensure similarity calculator exists (fallback to local init)."""
        if getattr(self, 'similarity_calc', None):
            return
        self.similarity_calc = getattr(self.library, 'similarity_calc', None)
        if self.similarity_calc is None:
            try:
                db_path = self.config.get('library', {}).get('database_path', 'data/metadata.db')
            except Exception as e:
                logger.debug(f"Falling back to default metadata path after config read failure: {e}")
                db_path = 'data/metadata.db'
            self.similarity_calc = SimilarityCalculator(db_path=db_path, config=self.config)

    def _similarity_config(self, limit_per_seed: Optional[int]) -> Dict[str, Any]:
        """Derive similarity generation settings (caps, pool sizes, per-seed limits)."""
        target_playlist_size = self.config.get('playlists', 'tracks_per_playlist', default=30)
        candidate_per_seed = limit_per_seed if limit_per_seed is not None else self.config.get('playlists', 'similar_per_seed', default=30)
        buffer_size = max(2, int(target_playlist_size * 1.0))  # 100% buffer
        pool_target = target_playlist_size + buffer_size
        artist_cap = max(6, getattr(self.config, 'max_tracks_per_artist', 6))
        return {
            'target_playlist_size': target_playlist_size,
            'candidate_per_seed': candidate_per_seed,
            'pool_target': pool_target,
            'artist_cap': artist_cap,
        }

    def _build_candidate_config(self, limit_per_seed: Optional[int], use_genre_discovery: bool) -> candidate_generator.CandidateConfig:
        """
        Build CandidateConfig from playlist config.

        Phase 8: Helper for candidate_generator module delegation
        """
        sim_cfg = self._similarity_config(limit_per_seed)
        return candidate_generator.CandidateConfig(
            limit_per_seed=sim_cfg['candidate_per_seed'],
            pool_target=sim_cfg['pool_target'],
            artist_cap=sim_cfg['artist_cap'],
            use_genre_discovery=use_genre_discovery,
            sonic_ratio=self.config.dynamic_sonic_ratio,
            genre_ratio=self.config.dynamic_genre_ratio,
            min_track_duration_seconds=self.config.min_track_duration_seconds,
            max_track_duration_seconds=self.config.max_track_duration_seconds,
            title_dedupe_enabled=self.config.title_dedupe_enabled,
            title_dedupe_threshold=self.config.title_dedupe_threshold,
            title_dedupe_mode=self.config.title_dedupe_mode,
            title_dedupe_short_title_min_len=self.config.title_dedupe_short_title_min_len,
        )

    def _filter_long_tracks(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove tracks above maximum duration (seconds value from config).

        Phase 3: Delegates to filtering.filter_by_duration()
        """
        max_duration_seconds = self.config.get('playlists', 'max_track_duration_seconds', default=720)
        return filtering.filter_by_duration(
            tracks=candidates,
            max_duration_seconds=max_duration_seconds,
        )

    def _build_seed_title_set(self, seeds: List[Dict[str, Any]]) -> set:
        """
        Normalize seed titles once for duplicate-title filtering.

        Phase 8: Delegates to candidate_generator.build_seed_title_set()
        """
        return candidate_generator.build_seed_title_set(seeds=seeds)

    def _cap_candidates_by_artist(
        self, candidates: List[Dict[str, Any]], artist_cap: int, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Apply artist cap and truncate to limit based on sonic score ordering.

        Phase 5: Delegates to scoring.cap_candidates_by_artist()
        """
        return scoring.cap_candidates_by_artist(
            candidates=candidates,
            artist_cap=artist_cap,
            limit=limit,
            score_key="similarity_score",
        )

    def _score_genre_and_hybrid(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float]]]:
        """
        Compute genre and hybrid scores; partition into pass/fail.

        Phase 5: Delegates to scoring.score_genre_and_hybrid()
        """
        return scoring.score_genre_and_hybrid(
            candidates=candidates,
            similarity_calculator=self.similarity_calc,
            genre_method=self.similarity_calc.genre_method,
        )

    def _finalize_pool(
        self, candidates: List[Dict[str, Any]], artist_cap: int, pool_target: int
    ) -> List[Dict[str, Any]]:
        """
        Sort by hybrid score, enforce artist cap, and trim to pool_target.

        Phase 5: Delegates to scoring.finalize_pool()
        """
        return scoring.finalize_pool(
            candidates=candidates,
            artist_cap=artist_cap,
            pool_target=pool_target,
            score_key="hybrid_score",
        )

    def _collect_sonic_candidates(
        self,
        seeds: List[Dict[str, Any]],
        seed_titles: set,
        candidate_per_seed: int,
        title_dedupe_tracker: Optional[TitleDedupeTracker] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Collect sonic-only candidates across seeds with basic filtering.

        Args:
            seeds: List of seed tracks
            seed_titles: Set of normalized seed titles for quick exclusion
            candidate_per_seed: Number of candidates to fetch per seed
            title_dedupe_tracker: Optional tracker for fuzzy title deduplication

        Returns:
            (candidates, filtered_counts)
        """
        min_track_duration_ms = self.config.min_track_duration_seconds * 1000
        all_candidates: Dict[str, Dict[str, Any]] = {}
        filtered_short = 0
        filtered_long = 0
        filtered_dupe_title = 0
        filtered_fuzzy_dupe = 0

        # Pre-populate tracker with seed titles if enabled
        if title_dedupe_tracker and title_dedupe_tracker.enabled:
            for seed in seeds:
                seed_artist = seed.get('artist', '')
                seed_title = seed.get('title', '')
                if seed_artist and seed_title:
                    title_dedupe_tracker.add(seed_artist, seed_title)

        for seed in seeds:
            seed_id = seed.get('rating_key')
            if not seed_id:
                continue

            similar = self.library.get_similar_tracks_sonic_only(seed_id, limit=candidate_per_seed, min_similarity=0.1)
            before_long = len(similar)
            similar = self._filter_long_tracks(similar)
            filtered_long += max(0, before_long - len(similar))

            weight = seed.get('play_count', 1)

            for track in similar:
                track_key = track.get('rating_key')
                track_artist = track.get('artist', '')
                track_title = track.get('title', '')

                # Skip tracks with same title as seed tracks (e.g., remasters, live versions)
                track_title_normalized = normalize_song_title(track_title)
                if track_title_normalized in seed_titles:
                    filtered_dupe_title += 1
                    continue

                # Fuzzy title deduplication (check against already-accepted candidates)
                if title_dedupe_tracker and title_dedupe_tracker.enabled:
                    is_dup, matched = title_dedupe_tracker.is_duplicate(
                        track_artist, track_title, debug=logger.isEnabledFor(logging.DEBUG)
                    )
                    if is_dup:
                        filtered_fuzzy_dupe += 1
                        logger.debug(
                            f"Title dedupe: skipping '{track_artist} - {track_title}' "
                            f"(fuzzy match to '{matched}')"
                        )
                        continue

                # Filter out short tracks (interludes, skits, etc.)
                track_duration = track.get('duration') or 0
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                    filtered_short += 1
                    continue

                existing = all_candidates.get(track_key)
                if existing and existing.get('similarity_score', 0) >= track.get('similarity_score', 0):
                    continue

                # Add to tracker for future duplicate detection
                if title_dedupe_tracker and title_dedupe_tracker.enabled:
                    title_dedupe_tracker.add(track_artist, track_title)

                track['seed_artist'] = seed.get('artist')
                track['seed_title'] = seed.get('title')
                track['seed_rating_key'] = seed_id
                track['weight'] = weight
                track['source'] = 'sonic'
                all_candidates[track_key] = track

        return list(all_candidates.values()), {
            'short': filtered_short,
            'long': filtered_long,
            'dupe_title': filtered_dupe_title,
            'fuzzy_dupe': filtered_fuzzy_dupe,
        }
    
    def analyze_listening_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze listening history to identify top artists/tracks

        Args:
            history: List of played tracks

        Returns:
            List of seed tracks (most frequently played)

        Phase 4: Delegates to history_analyzer.analyze_listening_history()
        """
        seed_count = self.config.get('playlists', 'seed_count', default=5)

        config = history_analyzer.HistoryAnalysisConfig(
            seed_count=seed_count,
            artist_count=0,  # Don't analyze top artists in this method
            include_collaborations=False,
        )

        result = history_analyzer.analyze_listening_history(
            history=history,
            config=config,
        )

        return result.seed_tracks

    def _select_diverse_seeds(self, play_counts: Counter, track_metadata: Dict,
                              count: int) -> List[Dict[str, Any]]:
        """
        Select seed tracks ensuring artist diversity

        Args:
            play_counts: Counter of plays by rating key
            track_metadata: Metadata for each track
            count: Number of seeds to select

        Returns:
            List of diverse seed tracks

        Phase 4: Delegates to history_analyzer.select_diverse_seeds()
        """
        return history_analyzer.select_diverse_seeds(
            play_counts=play_counts,
            track_metadata=track_metadata,
            count=count,
        )
    
    def generate_similar_tracks(self, seeds: List[Dict[str, Any]], dynamic: bool = False, limit_per_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate list of similar tracks based on seeds.

        Pipeline:
          1) Sonic-only discovery per seed (no genre filtering)
          2) Merge/dedup and apply per-artist cap
          3) Filter by genre threshold via hybrid similarity
          4) Rank by hybrid score and keep a buffered pool (target + 10%)

        Args:
            seeds: List of seed tracks
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)
            limit_per_seed: Override for similar_per_seed (useful for extending playlists)

        Returns:
            List of similar tracks (deduplicated and filtered for duplicate titles)

        Phase 8: Delegates to candidate_generator.generate_candidates()
        """
        # Ensure similarity calculator is available (fallback if not injected)
        self._ensure_similarity_calculator()

        # Build configuration
        config = self._build_candidate_config(limit_per_seed, dynamic)

        # Generate candidates
        result = candidate_generator.generate_candidates(
            seeds=seeds,
            library_client=self.library,
            similarity_calculator=self.similarity_calc,
            metadata_client=self.metadata,
            config=config,
        )

        return result.candidates

    def _generate_similar_tracks_dynamic(self, seeds: List[Dict[str, Any]], limit_per_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate tracks using dynamic mode: mix sonic similarity with genre-based discovery
        60% from sonic analysis, 40% from genre matching

        Args:
            seeds: List of seed tracks (must have 'genres' field)
            limit_per_seed: Override for similar_per_seed (useful for extending playlists)

        Returns:
            Mixed list of sonically similar and genre-matched tracks
        """
        similar_per_seed = limit_per_seed if limit_per_seed is not None else self.config.get('playlists', 'similar_per_seed', default=10)
        min_track_duration_ms = self.config.min_track_duration_seconds * 1000

        # Calculate how many tracks from each source (from config)
        sonic_per_seed = int(similar_per_seed * self.config.dynamic_sonic_ratio)
        genre_per_seed = int(similar_per_seed * self.config.dynamic_genre_ratio)

        logger.info(f"  Target: {sonic_per_seed} sonic + {genre_per_seed} genre-based tracks per seed")

        # Part 1: Get sonic similarity tracks (existing logic)
        sonic_tracks = []
        seen_keys = set()
        filtered_short_count = 0
        filtered_long_count = 0
        filtered_fuzzy_dupe_count = 0

        # Build set of normalized seed titles to filter out
        seed_titles = {normalize_song_title(seed.get('title', '')) for seed in seeds}
        seed_titles.discard('')

        # Build set of seed artists to exclude from similarity results
        seed_artists = {seed.get('artist', '').lower() for seed in seeds}
        seed_artists.discard('')

        # Create title dedupe tracker if enabled
        title_dedupe_tracker = TitleDedupeTracker(
            threshold=self.config.title_dedupe_threshold,
            mode=self.config.title_dedupe_mode,
            short_title_min_len=self.config.title_dedupe_short_title_min_len,
            enabled=self.config.title_dedupe_enabled,
        )

        # Pre-populate tracker with seed titles
        if title_dedupe_tracker.enabled:
            for seed in seeds:
                seed_artist = seed.get('artist', '')
                seed_title = seed.get('title', '')
                if seed_artist and seed_title:
                    title_dedupe_tracker.add(seed_artist, seed_title)

        for seed in seeds:
            key = seed.get('rating_key')
            if not key:
                continue

            similar = self.library.get_similar_tracks(key, limit=sonic_per_seed)
            before_long = len(similar)
            similar = self._filter_long_tracks(similar)
            filtered_long_count += max(0, before_long - len(similar))
            weight = seed.get('play_count', 1)

            for track in similar:
                track_key = track.get('rating_key')

                if track_key in seen_keys:
                    continue

                # Skip tracks by seed artists
                track_artist = track.get('artist', '').lower()
                if track_artist in seed_artists:
                    continue

                # Skip tracks with same title as seeds
                track_title = track.get('title', '')
                track_title_normalized = normalize_song_title(track_title)
                if track_title_normalized in seed_titles:
                    continue

                # Fuzzy title deduplication
                if title_dedupe_tracker.enabled:
                    is_dup, matched = title_dedupe_tracker.is_duplicate(
                        track.get('artist', ''), track_title,
                        debug=logger.isEnabledFor(logging.DEBUG)
                    )
                    if is_dup:
                        filtered_fuzzy_dupe_count += 1
                        logger.debug(
                            f"Title dedupe (dynamic/sonic): skipping '{track.get('artist')} - {track_title}' "
                            f"(fuzzy match to '{matched}')"
                        )
                        continue

                # Filter short tracks
                track_duration = track.get('duration') or 0
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                    filtered_short_count += 1
                    continue

                # Add to tracker for future duplicate detection
                if title_dedupe_tracker.enabled:
                    title_dedupe_tracker.add(track.get('artist', ''), track_title)

                seen_keys.add(track_key)
                track['seed_artist'] = seed.get('artist')
                track['seed_title'] = seed.get('title')
                track['weight'] = weight
                track['source'] = 'sonic'

                sonic_tracks.append(track)

        logger.info(f"  Sonic similarity: {len(sonic_tracks)} tracks (filtered {filtered_fuzzy_dupe_count} fuzzy dupes)")

        # Part 2: Get genre-based tracks using metadata database
        genre_tracks = []

        # Extract all genres from seeds
        all_genres = set()
        for seed in seeds:
            seed_genres = seed.get('genres', [])
            if seed_genres:
                # Take top 3 genres from each seed
                all_genres.update(seed_genres[:3])

        if not all_genres:
            logger.info("  No genres available for genre-based discovery, using sonic only")
            return sonic_tracks

        seed_genre_list = sorted(list(all_genres))
        logger.info(f"  Seed genres: {', '.join(seed_genre_list)}")

        # Use metadata database if available, otherwise fall back to local metadata
        if self.metadata:
            cursor = self.metadata.conn.cursor()
            normalized_seed_genres = [normalize_genre(g) for g in all_genres]
            placeholders = ",".join("?" * len(normalized_seed_genres))
            cursor.execute(f"""
                SELECT artist, GROUP_CONCAT(genre) AS genres, COUNT(*) AS match_count
                FROM artist_genres
                WHERE genre IN ({placeholders})
                GROUP BY artist
                ORDER BY match_count DESC
            """, tuple(normalized_seed_genres))

            artist_genre_scores = []
            for row in cursor.fetchall():
                artist_name = row['artist']
                # Skip seed artists
                if any(safe_get_artist(seed) == artist_name.lower() for seed in seeds):
                    continue

                matching_genres = row['genres'].split(',') if row['genres'] else []
                artist_genre_scores.append({
                    'artist': artist_name,
                    'matching_genres': matching_genres,
                    'match_count': row['match_count'],
                    'all_tags': matching_genres
                })

            logger.info(f"  Found {len(artist_genre_scores)} artists with matching genres")

            # Log top matching artists
            logger.info(f"  Top genre matches:")
            for artist_score in artist_genre_scores[:10]:
                genres_str = ', '.join(artist_score['matching_genres'])
                logger.info(f"    - {sanitize_for_logging(artist_score['artist'])}: {artist_score['match_count']} matches ({genres_str})")

            # Select tracks from top-matching artists (1 track per artist for diversity)
            target_genre_tracks = genre_per_seed * len(seeds)
            selected_artists = set()

            for artist_score in artist_genre_scores:
                if len(genre_tracks) >= target_genre_tracks:
                    break

                artist_name = artist_score['artist']

                # Get tracks from this artist (from metadata DB to get rating keys)
                matched_tracks = self.metadata.get_tracks_by_artist(artist_name, limit=5)

                # Find first track that's not already used
                for matched in matched_tracks:
                    track_key = matched.get('track_id')

                    # Skip if already in sonic tracks or genre tracks
                    if track_key in seen_keys:
                        continue

                    # Skip seeds
                    matched_title = matched.get('title', '')
                    track_title_normalized = normalize_song_title(matched_title)
                    if track_title_normalized in seed_titles:
                        continue

                    # Fuzzy title deduplication
                    if title_dedupe_tracker.enabled:
                        is_dup, dup_matched = title_dedupe_tracker.is_duplicate(
                            artist_name, matched_title,
                            debug=logger.isEnabledFor(logging.DEBUG)
                        )
                        if is_dup:
                            filtered_fuzzy_dupe_count += 1
                            logger.debug(
                                f"Title dedupe (dynamic/genre-meta): skipping '{artist_name} - {matched_title}' "
                                f"(fuzzy match to '{dup_matched}')"
                            )
                            continue

                    # Get full track data from library (includes duration)
                    full_track_data = self.library.get_track_by_key(track_key)
                    if not full_track_data:
                        continue

                    # Hard filter: Check duration constraints
                    track_duration_ms = full_track_data.get('duration', 0)
                    min_duration_ms = int(self.config.get('playlists', 'min_track_duration_seconds', default=46) * 1000)
                    max_duration_ms = int(self.config.get('playlists', 'max_track_duration_seconds', default=720) * 1000)

                    # Skip if duration outside acceptable range
                    if track_duration_ms > 0:
                        if track_duration_ms < min_duration_ms:
                            logger.debug(f"Skipping {full_track_data.get('title')} - too short ({track_duration_ms}ms < {min_duration_ms}ms)")
                            continue
                        if track_duration_ms > max_duration_ms:
                            logger.debug(f"Skipping {full_track_data.get('title')} - too long ({track_duration_ms}ms > {max_duration_ms}ms)")
                            continue

                    # Found a good track - add it with full library data
                    track = {
                        'rating_key': track_key,
                        'title': full_track_data.get('title'),
                        'artist': full_track_data.get('artist'),
                        'album': full_track_data.get('album'),
                        'duration': full_track_data.get('duration', 0),
                        'uri': f"/library/metadata/{track_key}",
                        'source': 'genre',
                        'weight': matched.get('weight', 1.0),
                        'matched_genres': artist_score['matching_genres']  # Store for reporting
                    }

                    # Add to tracker for future duplicate detection
                    if title_dedupe_tracker.enabled:
                        title_dedupe_tracker.add(full_track_data.get('artist', ''), full_track_data.get('title', ''))

                    seen_keys.add(track_key)
                    genre_tracks.append(track)
                    selected_artists.add(artist_name)

                    # Only take 1 track per artist
                    break

            logger.info(f"  Genre-based discovery: {len(genre_tracks)} tracks from {len(selected_artists)} artists")

        else:
            # Fallback to local library metadata (original implementation)
            logger.info("  Metadata database not available, falling back to local library genre metadata")

            # Get all tracks from library (cached)
            all_library_tracks = self.library.get_all_tracks()

            # Filter tracks by genre and diversify by artist
            artist_track_counts = {}

            for track in all_library_tracks:
                track_key = track.get('rating_key')

                # Skip if already in sonic tracks
                if track_key in seen_keys:
                    continue

                # Skip seeds
                track_title = track.get('title', '')
                track_title_normalized = normalize_song_title(track_title)
                if track_title_normalized in seed_titles:
                    continue

                # Fuzzy title deduplication
                track_artist = track.get('artist', '')
                if title_dedupe_tracker.enabled:
                    is_dup, dup_matched = title_dedupe_tracker.is_duplicate(
                        track_artist, track_title,
                        debug=logger.isEnabledFor(logging.DEBUG)
                    )
                    if is_dup:
                        filtered_fuzzy_dupe_count += 1
                        logger.debug(
                            f"Title dedupe (dynamic/genre-fallback): skipping '{track_artist} - {track_title}' "
                            f"(fuzzy match to '{dup_matched}')"
                        )
                        continue

                # Filter tracks by duration (hard limits)
                track_duration = track.get('duration') or 0
                if track_duration > 0:
                    if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                        continue
                    if max_track_duration_ms > 0 and track_duration > max_track_duration_ms:
                        filtered_long_count += 1
                        continue

                # Check if track's genre matches any seed genres
                track_genre = track.get('genre', '')
                if not track_genre:
                    continue

                # Match if any seed genre is in the track's genre (with normalization)
                track_genre_normalized = normalize_genre(track_genre)
                genre_match = False
                for seed_genre in all_genres:
                    seed_genre_normalized = normalize_genre(seed_genre)
                    if seed_genre_normalized in track_genre_normalized:
                        genre_match = True
                        break

                if not genre_match:
                    continue

                # Limit tracks per artist to increase diversity
                artist = track.get('artist', 'Unknown')
                artist_track_counts[artist] = artist_track_counts.get(artist, 0) + 1

                # Max 2 tracks per artist in genre pool
                if artist_track_counts[artist] > 2:
                    continue

                # Add to tracker for future duplicate detection
                if title_dedupe_tracker.enabled:
                    title_dedupe_tracker.add(track_artist, track_title)

                seen_keys.add(track_key)
                track['source'] = 'genre'
                track['weight'] = 1
                genre_tracks.append(track)

                # Stop once we have enough genre tracks
                target_genre_tracks = genre_per_seed * len(seeds)
                if len(genre_tracks) >= target_genre_tracks:
                    break

            logger.info(f"  Genre-based discovery: {len(genre_tracks)} tracks from {len(artist_track_counts)} artists")

        # Combine sonic and genre tracks
        all_tracks = sonic_tracks + genre_tracks

        if filtered_short_count > 0:
            logger.info(f"  Filtered out {filtered_short_count} short tracks (< {self.config.min_track_duration_seconds}s)")
        if filtered_long_count > 0:
            logger.info(f"  Filtered out {filtered_long_count} long tracks (> {self.config.max_track_duration_seconds}s)")
        if filtered_fuzzy_dupe_count > 0:
            logger.info(f"  Filtered out {filtered_fuzzy_dupe_count} fuzzy duplicate titles")

        logger.info(f"  Total: {len(all_tracks)} tracks ({len(sonic_tracks)} sonic + {len(genre_tracks)} genre)")

        return all_tracks

    def filter_tracks(self, tracks: List[Dict[str, Any]],
                     history: List[Dict[str, Any]],
                     exempt_tracks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Filter out recently played tracks to keep playlist fresh

        Args:
            tracks: List of candidate tracks
            history: Play history to exclude
            exempt_tracks: Optional list of tracks exempt from filtering (e.g., seed tracks)

        Returns:
            Filtered list of tracks

        Phase 3: Delegates to filtering.filter_by_recently_played()
        """
        # Check if filtering is enabled
        if not self.config.recently_played_filter_enabled:
            logger.info("Recently played filtering is disabled")
            return tracks

        # Get filter configuration
        lookback_days = self.config.recently_played_lookback_days
        min_playcount = self.config.recently_played_min_playcount

        # Delegate to filtering module
        result = filtering.filter_by_recently_played(
            tracks=tracks,
            play_history=history,
            lookback_days=lookback_days,
            min_playcount=min_playcount,
            exempt_tracks=exempt_tracks,
        )

        return result.filtered_tracks

    def _filter_by_scrobbles(
        self,
        tracks: List[Dict[str, Any]],
        scrobbles: List[Dict[str, Any]],
        lookback_days: int,
        sample_limit: int = 5,
        exempt_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates using Last.FM scrobbles without DB matching.
        Uses normalized artist/title keys (ignoring mbid for consistency).

        Phase 3: Delegates to filtering.filter_by_scrobbles()
        """
        result = filtering.filter_by_scrobbles(
            tracks=tracks,
            scrobbles=scrobbles,
            lookback_days=lookback_days,
            exempt_tracks=exempt_tracks,
            sample_limit=sample_limit,
        )
        return result.filtered_tracks

    def _ensure_seed_tracks_present(
        self,
        seed_tracks: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        target_length: int,
    ) -> List[Dict[str, Any]]:
        """
        Ensure all seed tracks are kept in the final playlist without reordering the DS
        ordering of candidates. Missing seeds are inserted at positions that avoid
        immediate same-artist adjacency when possible. Length is capped at target_length,
        preserving all seeds when there is room.

        Phase 3: Delegates to filtering.ensure_seed_tracks_present()
        """
        return filtering.ensure_seed_tracks_present(
            seed_tracks=seed_tracks,
            candidates=candidates,
            target_length=target_length,
        )

    def _log_recency_edge_diff(self, before_tracks: List[Dict[str, Any]], after_tracks: List[Dict[str, Any]]) -> None:
        """Diagnostic: log adjacency changes introduced by recency filtering (no behavior change)."""
        # Phase 9: Delegate to reporter module
        reporter.log_recency_edge_diff(
            before_tracks=before_tracks,
            after_tracks=after_tracks,
        )

    def _compute_excluded_from_scrobbles(
        self, candidates: List[Dict[str, Any]], scrobbles: List[Dict[str, Any]], lookback_days: int, seed_id: Optional[str]
    ) -> Set[str]:
        """
        Return rating_keys to exclude based on scrobbles, preserving seed_id if present.
        Uses artist::title matching (ignoring mbid) for consistency between Last.fm and library.

        Phase 3: Delegates to filtering.filter_by_scrobbles() and converts result to excluded IDs
        """
        if not scrobbles or lookback_days <= 0:
            return set()

        # Find the seed track to exempt if seed_id is provided
        exempt_tracks = None
        if seed_id:
            seed_track = next((t for t in candidates if str(t.get("rating_key")) == str(seed_id)), None)
            if seed_track:
                exempt_tracks = [seed_track]

        # Use filtering module to determine which tracks would be filtered
        result = filtering.filter_by_scrobbles(
            tracks=candidates,
            scrobbles=scrobbles,
            lookback_days=lookback_days,
            exempt_tracks=exempt_tracks,
            sample_limit=0,  # Disable logging samples here
        )

        # Convert filtered tracks to excluded IDs (tracks that were removed)
        filtered_ids = {str(t.get("rating_key")) for t in result.filtered_tracks if t.get("rating_key")}
        all_ids = {str(t.get("rating_key")) for t in candidates if t.get("rating_key")}
        excluded = all_ids - filtered_ids

        if seed_id and seed_id in excluded:
            logger.warning("Recency exclusions contained seed_id %s; preserving seed.", seed_id)

        return excluded
    
    def diversify_tracks(self, tracks: List[Dict[str, Any]],
                        max_per_artist: int = None) -> List[Dict[str, Any]]:
        """
        Ensure playlist has diversity (don't let one artist dominate)

        Args:
            tracks: List of tracks
            max_per_artist: Maximum tracks per artist (uses config if None)

        Returns:
            Diversified track list

        Phase 6: Delegates to diversity.diversify_by_artist_cap()
        """
        if max_per_artist is None:
            max_per_artist = self.config.max_tracks_per_artist

        return diversity.diversify_by_artist_cap(
            tracks=tracks,
            max_per_artist=max_per_artist,
        )
    
    def _is_collaboration_of(self, collaboration_name: str, base_artist: str) -> bool:
        """
        Check if a collaboration name includes the base artist

        Args:
            collaboration_name: Full artist name (potentially a collaboration)
            base_artist: Base artist name to check for

        Returns:
            True if collaboration_name is a collaboration including base_artist

        Phase 4: Delegates to history_analyzer.is_collaboration_of()
        """
        return history_analyzer.is_collaboration_of(
            collaboration_name=collaboration_name,
            base_artist=base_artist,
        )

    def _analyze_top_artists_from_history(self, history: List[Dict[str, Any]],
                                          artist_count: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze listening history to get top played artists and their tracks

        Args:
            history: List of played tracks
            artist_count: Number of top artists to return

        Returns:
            Dict mapping artist name to list of their played tracks

        Phase 4: Delegates to history_analyzer.analyze_top_artists()
        """
        return history_analyzer.analyze_top_artists(
            history=history,
            artist_count=artist_count,
            include_collaborations=True,  # Original behavior includes collaborations
        )

    def _get_seed_tracks_for_artist(self, artist: str, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get seed tracks for an artist: 1 top played + random from top 20

        Args:
            artist: Artist name
            tracks: List of tracks by this artist

        Returns:
            List of seed tracks (count from config.seed_count)

        Phase 4: Delegates to history_analyzer.get_seed_tracks_for_artist()
        """
        seed_count = self.config.get('playlists', 'seed_count', default=5)

        return history_analyzer.get_seed_tracks_for_artist(
            artist=artist,
            tracks=tracks,
            seed_count=seed_count,
        )

    def create_playlist_batch(
        self,
        count: int,
        dynamic: bool = False,
        pipeline_override: Optional[str] = None,
        ds_mode_override: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Create multiple playlists with single seed artist per playlist

        Args:
            count: Number of playlists to create
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info(f"="*70)
        logger.info(f"Creating {count} playlists (1 seed artist per playlist)")
        if dynamic:
            logger.info("Dynamic mode enabled: 60% sonic similarity + 40% genre-based discovery")
        logger.info(f"="*70)

        # Get listening history - prefer Last.FM if available
        if self.lastfm and self.matcher:
            history = self._get_lastfm_history()
        else:
            history = self._get_local_history()

        if not history:
            logger.warning("No listening history found")
            return []

        # We need exactly 'count' artists (one per playlist)
        artists_needed = count

        logger.info(f"\nNeed {artists_needed} artists (one per playlist)")
        logger.info(f"Each artist will provide 4 seed tracks\n")

        # Get top played artists and their tracks
        top_artists_tracks = self._analyze_top_artists_from_history(history, artists_needed)

        if len(top_artists_tracks) < artists_needed:
            logger.warning(f"Only found {len(top_artists_tracks)} artists, need {artists_needed}")
            return []

        # Get seed tracks for each artist (2 most played + 2 random from top 10)
        logger.info(f"\n{'='*70}")
        logger.info("Selecting seed tracks for each artist:")
        logger.info(f"{'='*70}\n")

        artist_seeds = {}
        for artist, tracks in top_artists_tracks.items():
            logger.info(f"  Artist: {artist}")
            seeds = self._get_seed_tracks_for_artist(artist, tracks)
            if seeds:
                artist_seeds[artist] = seeds

        # Fetch genre tags for artists (Last.FM genres removed; rely on DB/file genres)
        logger.info(f"\n{'='*70}")
        logger.info("Genre tags: using database/file sources only (Last.FM disabled for genres)")
        logger.info(f"{'='*70}\n")

        # Create playlists from single artists
        playlists = self._create_playlists_from_single_artists(
            artist_seeds,
            history,
            dynamic=dynamic,
            pipeline_override=pipeline_override,
            ds_mode_override=ds_mode_override,
        )

        return playlists

    def create_playlist_for_artist(
        self,
        artist_name: str,
        track_count: int = 30,
        track_title: Optional[str] = None,
        dynamic: bool = False,
        verbose: bool = False,
        ds_mode_override: Optional[str] = None,
        artist_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single playlist for a specific artist without requiring listening history

        Args:
            artist_name: Name of the artist to create playlist for
            track_count: Target number of tracks in playlist

        Returns:
            Playlist dictionary with tracks and metadata, or None if unable to create
        """
        logger.info(f"Creating playlist for artist: {artist_name}")

        # Get all tracks by this artist from local library
        all_library_tracks = self.library.get_all_tracks()

        # Filter to just this artist (case-insensitive, exact match)
        artist_tracks = [t for t in all_library_tracks
                        if safe_get_artist(t) == artist_name.lower()]

        # If we don't have enough tracks (or zero), search for collaborations
        if len(artist_tracks) < 4:
            logger.info(f"Artist has only {len(artist_tracks)} exact match tracks, searching for collaborations...")

            # Search for collaboration tracks
            collaboration_tracks = [t for t in all_library_tracks
                                  if self._is_collaboration_of(t.get('artist', ''), artist_name)]

            if collaboration_tracks:
                # Group by collaboration artist name for logging
                collab_artists = {}
                for track in collaboration_tracks:
                    collab_artist = track.get('artist', '')
                    if collab_artist not in collab_artists:
                        collab_artists[collab_artist] = 0
                    collab_artists[collab_artist] += 1

                for collab_artist, count in collab_artists.items():
                    logger.info(f"  Found collaboration: {collab_artist} ({count} tracks)")

                # Combine exact matches and collaborations
                artist_tracks.extend(collaboration_tracks)
                logger.info(f"Found {len(artist_tracks)} total tracks ({len(artist_tracks) - len(collaboration_tracks)} solo, {len(collaboration_tracks)} collaborations)")
            else:
                logger.warning(f"Artist has only {len(artist_tracks)} tracks and no collaborations found, need at least 4")
                return None

        if len(artist_tracks) < 4:
            logger.warning(f"Artist has only {len(artist_tracks)} total tracks (including collaborations), need at least 4")
            return None

        logger.info(f"Using {len(artist_tracks)} tracks for {artist_name}")

        # Add play count (0 for all, since we don't have history)
        for track in artist_tracks:
            track['play_count'] = 0

        # If a specific track title is provided, select canonical match and keep seeds focused
        seed_tracks: List[Dict[str, Any]] = []
        if track_title:
            selected = self._select_canonical_track(artist_tracks, track_title)
            if selected:
                seed_tracks = [selected]
                logger.info("Using specified seed track: %s - %s", selected.get("artist"), selected.get("title"))
            else:
                logger.warning("Requested seed track '%s' not found for artist %s; falling back to random seeds.", track_title, artist_name)

        if not seed_tracks:
            # Get 4 seed tracks (2 most played + 2 random from top 10)
            # Since we don't have play counts, just pick 4 random tracks
            import random
            seed_tracks = random.sample(artist_tracks, min(4, len(artist_tracks)))

        seed_titles = [track.get('title') for track in seed_tracks]
        logger.info(f"Seeds ({len(seed_tracks)}): {', '.join(seed_titles)}")

        # Genre tags are sourced from database/file tags only (Last.FM disabled for genres)

        # Prepare recency data (scrobbles preferred; matched history retained for filtering)
        scrobbles: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        if self.lastfm:
            try:
                scrobbles = self._get_lastfm_scrobbles_raw(use_cache=True)
            except Exception as exc:
                logger.warning("Last.FM scrobble fetch failed; skipping scrobble recency filter (%s)", exc, exc_info=True)

        # DS scope selection
        allowed_track_ids: Optional[List[str]] = None
        if artist_only:
            ds_candidates = list(artist_tracks)
            excluded_ids = set()
            if scrobbles:
                excluded_ids = self._compute_excluded_from_scrobbles(
                    ds_candidates,
                    scrobbles,
                    lookback_days=self.config.recently_played_lookback_days,
                    seed_id=seed_tracks[0].get("rating_key") if seed_tracks else None,
                )
                if excluded_ids and len(ds_candidates) - len(excluded_ids) <= 1:
                    logger.warning("Recency exclusions would empty artist-only DS candidate set; skipping recency exclusion.")
                    excluded_ids = set()
            allowed_track_ids = [t.get("rating_key") for t in ds_candidates if t.get("rating_key") and str(t.get("rating_key")) not in excluded_ids]
            if allowed_track_ids:
                logger.info("DS candidate ids (sample): %s", [str(i) for i in allowed_track_ids[:3]])
            logger.info("DS scope: artist_only")
        else:
            logger.info("DS scope: library")
            # Apply library-wide exclusions pre-DS
            all_ids = [str(t.get("rating_key")) for t in all_library_tracks if t.get("rating_key")]
            excluded_ids = set()
            if scrobbles:
                excluded_ids = self._compute_excluded_from_scrobbles(
                    all_library_tracks,
                    scrobbles,
                    lookback_days=self.config.recently_played_lookback_days,
                    seed_id=seed_tracks[0].get("rating_key") if seed_tracks else None,
                )
                if excluded_ids and len(all_ids) - len(excluded_ids) <= 1:
                    logger.warning("Recency exclusions would empty DS library candidate set; skipping recency exclusion.")
                    excluded_ids = set()
            allowed_track_ids = None  # avoid huge lists; use exclusions instead
        if os.environ.get("PLAYLIST_DIAG_RECENCY"):
            logger.info(
                "Recency diag: scope=%s excluded=%d allowed=%s threshold=10000",
                "artist_only" if artist_only else "library",
                len(excluded_ids),
                len(allowed_track_ids) if allowed_track_ids else 0,
            )

        # Try each seed track until we find one that's in the artifact
        ds_tracks = None
        last_error = None
        for i, seed_track in enumerate(seed_tracks):
            seed_id = seed_track.get('rating_key')
            if not seed_id:
                continue
            try:
                ds_tracks = self._maybe_generate_ds_playlist(
                    seed_track_id=seed_id,
                    target_length=track_count,
                    mode_override=ds_mode_override or ("dynamic" if dynamic else None),
                    seed_artist=artist_name,
                    allowed_track_ids=allowed_track_ids or None,
                    excluded_track_ids=excluded_ids or None,
                    anchor_seed_tracks=seed_tracks,  # Pass full seed track info for title+artist resolution
                )
                if i > 0:
                    logger.info(f"Successfully used alternative seed #{i+1}: {seed_track.get('artist')} - {seed_track.get('title')}")
                break  # Success!
            except ValueError as e:
                if "not found in artifact" in str(e):
                    last_error = e
                    logger.debug(f"Seed #{i+1} not in artifact, trying next seed...")
                    continue
                else:
                    raise  # Re-raise if it's a different error

        if ds_tracks is None:
            # All seeds failed - provide helpful error
            raise ValueError(
                f"None of the {len(seed_tracks)} seed tracks for '{artist_name}' were found in the artifact. "
                f"This usually means beat3tower features haven't been extracted for these tracks. "
                f"Run 'python scripts/update_sonic.py --beat3tower' to extract features, then rebuild the artifact."
            )

        # Apply recency filtering if available
        filtered_ds_tracks = ds_tracks
        if self.lastfm:
            try:
                scrobbles = scrobbles or self._get_lastfm_scrobbles_raw(use_cache=True)
                filtered_ds_tracks = self._filter_by_scrobbles(
                    ds_tracks,
                    scrobbles,
                    lookback_days=self.config.recently_played_lookback_days,
                    exempt_tracks=seed_tracks,
                )
            except Exception as exc:
                logger.warning("Last.FM scrobble filter failed; using unfiltered DS tracks (%s)", exc, exc_info=True)
        else:
            filtered_ds_tracks = self.filter_tracks(ds_tracks, history, exempt_tracks=seed_tracks)

        final_tracks = self._ensure_seed_tracks_present(seed_tracks, filtered_ds_tracks, track_count)

        if not final_tracks:
            raise RuntimeError("DS pipeline tracks were filtered out by recency rules.")

        # Recompute edge scores for the final track order
        if getattr(self, "_last_ds_report", None):
            recomputed_edges = self._compute_edge_scores_from_artifact(
                final_tracks,
                self._last_ds_report.get("artifact_path"),
                transition_floor=self._last_ds_report.get("transition_floor"),
                transition_gamma=self._last_ds_report.get("transition_gamma"),
                center_transitions=bool(self._last_ds_report.get("transition_centered")),
                embedding_random_seed=self._last_ds_report.get("random_seed"),
                verbose=verbose,
                sonic_variant=self._last_ds_report.get("sonic_variant"),
            )
            self._last_ds_report["edge_scores"] = recomputed_edges
            playlist_stats = self._last_ds_report.get("playlist_stats") or {}
            playlist_stats_playlist = playlist_stats.get("playlist") or {}
            playlist_stats_playlist["edge_scores"] = recomputed_edges
            playlist_stats["playlist"] = playlist_stats_playlist
            self._last_ds_report["playlist_stats"] = playlist_stats
            if os.environ.get("PLAYLIST_DIAG_RECENCY"):
                self._log_recency_edge_diff(final_tracks, final_tracks)
        title = f"Auto: {artist_name}"
        self._print_playlist_report(final_tracks, artist_name=artist_name, dynamic=dynamic, verbose_edges=verbose)
        return {
            'title': title,
            'artists': (artist_name,),
            'genres': [],
            'tracks': final_tracks,
        }


    #         Args:
    #             artists: List of artist names
    #             num_pairs: Number of pairs to create
    #             min_similarity: Minimum similarity threshold for pairing (0.0-1.0)

    #         Returns:
    #             List of artist pairs (tuples)
    #         """
    #         if min_similarity is None:
    #             min_similarity = self.config.similarity_min_threshold

    #         logger.info(f"\n{'='*70}")
    #         logger.info(f"Pairing {len(artists)} artists by similarity:")
    #         logger.info(f"{'='*70}\n")

    #         if len(artists) < num_pairs * 2:
    #             logger.warning(f"Not enough artists to create {num_pairs} pairs")
    #             num_pairs = len(artists) // 2

    #         # Build similarity matrix between all artists
    #         # Create dummy seeds with just artist names for similarity calculation
    #         artist_seeds_dummy = [{'artist': a, 'genres': []} for a in artists]

    #         # Fetch similar artists for each
    #         for seed in artist_seeds_dummy:
    #             artist = seed['artist']
    #             similar = self._get_similar_artists(artist)
    #             seed['similar_artists'] = similar

    #         # Calculate pairwise similarity
    #         similarity_matrix = {}
    #         for i, seed1 in enumerate(artist_seeds_dummy):
    #             for j, seed2 in enumerate(artist_seeds_dummy):
    #                 if i >= j:
    #                     continue
    #                 sim = self._calculate_artist_similarity(seed1, seed2)
    #                 similarity_matrix[(i, j)] = sim

    #         # Find and remove most dissimilar pair
    #         if len(artists) > num_pairs * 2:
    #             sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1])
    #             worst_pair_idx, worst_sim = sorted_pairs[0]
    #             idx1, idx2 = worst_pair_idx
    #             artist1, artist2 = artists[idx1], artists[idx2]

    #             logger.info(f">> REMOVING MOST DISSIMILAR PAIR:")
    #             logger.info(f"   {artist1} <-> {artist2} (similarity = {worst_sim:.2f})")
    #             logger.info(f"   This ensures all remaining pairs are cohesive\n")

    #             # Remove both artists from the pool
    #             artists = [a for i, a in enumerate(artists) if i not in [idx1, idx2]]

    #         # Pair up remaining artists by highest similarity
    #         logger.info("Creating artist pairs by similarity:\n")
    #         pairs = []
    #         used = set()

    #         # Build new similarity matrix with current artist indices
    #         new_similarity_matrix = {}
    #         for i in range(len(artists)):
    #             for j in range(i + 1, len(artists)):
    #                 # Calculate similarity between these two artists
    #                 seed1 = {'artist': artists[i], 'genres': []}
    #                 seed2 = {'artist': artists[j], 'genres': []}
    #                 seed1['similar_artists'] = self._get_similar_artists(artists[i])
    #                 seed2['similar_artists'] = self._get_similar_artists(artists[j])

    #                 sim = self._calculate_artist_similarity(seed1, seed2)
    #                 new_similarity_matrix[(i, j)] = sim

    #         similarity_scores = [(sim, i, j) for (i, j), sim in new_similarity_matrix.items()]

    #         # Sort by similarity (highest first)
    #         similarity_scores.sort(reverse=True)

    #         logger.info(f"Pairing artists (minimum similarity = {min_similarity}):\n")

    #         for sim, i, j in similarity_scores:
    #             if i not in used and j not in used and len(pairs) < num_pairs:
    #                 # Only pair if similarity meets threshold
    #                 if sim >= min_similarity:
    #                     pairs.append((artists[i], artists[j]))
    #                     used.add(i)
    #                     used.add(j)
    #                     logger.info(f"  Pair {len(pairs)}: {artists[i]} <-> {artists[j]} (similarity = {sim:.2f})")
    #                 else:
    #                     # Rejected pairs don't meet threshold (debug logging removed for verbosity)
    #                     pass

    #         if len(pairs) < num_pairs:
    #             logger.warning(f"\nOnly created {len(pairs)} pairs (requested {num_pairs})")
    #             logger.warning(f"Could not find enough artist pairs with similarity >= {min_similarity}")
    #             logger.warning(f"Consider increasing artist pool or lowering similarity threshold\n")
    #         else:
    #             logger.info(f"\nCreated {len(pairs)} artist pairs (all with similarity >= {min_similarity})\n")

    #         return pairs

    def _generate_playlist_title(self, artist1: str, artist2: str, genres: List[str]) -> str:
        """
        Generate playlist title based on artist(s) and configured format

        Args:
            artist1: First artist name (or only artist for single-artist playlists)
            artist2: Second artist name (empty string for single-artist playlists)
            genres: Combined genre tags

        Returns:
            Generated playlist title (without prefix)
        """
        name_format = self.config.get('playlists', 'name_format', 'artists')

        # Determine if single or paired artists
        is_single_artist = not artist2 or artist2 == ""

        if name_format == 'artists':
            # Format: Artist1 or Artist1 + Artist2
            if is_single_artist:
                return artist1
            else:
                return f"{artist1} + {artist2}"

        elif name_format == 'genres':
            # Format: Dominant Genre Style
            if genres:
                # Take first two genres
                genre_text = " & ".join(genres[:2]).title()
                return f"{genre_text} Mix"
            else:
                # Fallback to artist names if no genres
                if is_single_artist:
                    return artist1
                else:
                    return f"{artist1} + {artist2}"

        elif name_format == 'hybrid':
            # Format: Artist1 (Genre) or Artist1 + Artist2 (Genre)
            if genres:
                genre_text = genres[0].title()
                if is_single_artist:
                    return f"{artist1} ({genre_text})"
                else:
                    return f"{artist1} + {artist2} ({genre_text})"
            else:
                if is_single_artist:
                    return artist1
                else:
                    return f"{artist1} + {artist2}"

        else:
            # Default to artists
            if is_single_artist:
                return artist1
            else:
                return f"{artist1} + {artist2}"


    #         Args:
    #             artist_pairs: List of (artist1, artist2) tuples
    #             artist_seeds: Dict mapping artist name to their seed tracks
    #             history: Play history for filtering

    #         Returns:
    #             List of playlist dictionaries with tracks and metadata
    #         """
    #         logger.info(f"{'='*70}")
    #         logger.info(f"Generating playlists from artist pairs:")
    #         logger.info(f"{'='*70}\n")

    #         playlists = []

    #         for idx, (artist1, artist2) in enumerate(artist_pairs, 1):
    #             logger.info(f"Playlist {idx}: {artist1} + {artist2}")

    #             # Get seed tracks for both artists (4 total)
    #             seeds = artist_seeds.get(artist1, []) + artist_seeds.get(artist2, [])

    #             for seed in seeds:
    #                 logger.info(f"  Seed: {seed.get('artist')} - {seed.get('title')}")

    #             # Collect all genres from seeds
    #             all_genres = []
    #             for seed in seeds:
    #                 all_genres.extend(seed.get('genres', []))
    #             # Get unique genres while preserving order
    #             seen = set()
    #             unique_genres = [g for g in all_genres if not (g in seen or seen.add(g))]

    #             # Generate similar tracks using similarity engine
    #             similar_tracks = self.generate_similar_tracks(seeds, dynamic=dynamic)
    #             logger.info(f"  Generated {len(similar_tracks)} similar tracks")

    #             # Add seed tracks to the candidate pool (they're guaranteed to be in playlist)
    #             all_tracks = seeds + similar_tracks
    #             logger.info(f"  Total candidates: {len(all_tracks)} tracks ({len(seeds)} seeds + {len(similar_tracks)} similar)")

    #             # Filter out recently played (seeds are exempt)
    #             fresh_tracks = self.filter_tracks(all_tracks, history, exempt_tracks=seeds)
    #             logger.info(f"  After filtering: {len(fresh_tracks)} fresh tracks")

    #             # Diversify
    #             diverse_tracks = self.diversify_tracks(fresh_tracks)
    #             logger.info(f"  After diversification: {len(diverse_tracks)} tracks")

    #             # Optimize track order using TSP (end→beginning transitions)
    #             final_tracks = self._optimize_playlist_order_tsp(diverse_tracks, seeds)
    #             logger.info(f"  TSP optimization complete: {len(final_tracks)} tracks")

    #             # Limit artist frequency in 8-song windows (max 1 per window)
    #             final_tracks = self._limit_artist_frequency_in_window(final_tracks, window_size=8, max_per_window=1)
    #             logger.info(f"  After limiting artist frequency: {len(final_tracks)} tracks")

    #             # Ensure playlist meets minimum duration
    #             min_duration_ms = self.config.min_duration_minutes * 60 * 1000  # Convert minutes to milliseconds
    #             total_duration_ms = 0
    #             duration_limited_tracks = []

    #             for track in final_tracks:
    #                 duration_limited_tracks.append(track)
    #                 total_duration_ms += (track.get('duration') or 0)

    #                 # Stop once we've reached the minimum duration
    #                 if total_duration_ms >= min_duration_ms:
    #                     break

    #             final_tracks = duration_limited_tracks

    #             # Log final duration
    #             total_minutes = total_duration_ms / 1000 / 60
    #             logger.info(f"  Playlist duration: {total_minutes:.1f} minutes")

    #             if final_tracks:
    #                 # Generate title for this playlist
    #                 title = self._generate_playlist_title(artist1, artist2, unique_genres)

    #                 # Print detailed track report
    #                 self._print_playlist_report(final_tracks, artist_name=f"{artist1} + {artist2}", dynamic=dynamic)

    #                 playlists.append({
    #                     'title': title,
    #                     'artists': (artist1, artist2),
    #                     'genres': unique_genres,
    #                     'tracks': final_tracks
    #                 })
    #                 logger.info(f"  Title: {title}")
    #                 logger.info(f"  Final: {len(final_tracks)} tracks from {len(set(t.get('artist') for t in final_tracks))} artists\n")
    #             else:
    #                 logger.warning(f"  No tracks generated for this pair\n")

    #         return playlists

    def _create_playlists_from_single_artists(
        self,
        artist_seeds: Dict[str, List[Dict[str, Any]]],
        history: List[Dict[str, Any]],
        dynamic: bool = False,
        pipeline_override: Optional[str] = None,
        ds_mode_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create playlists from single artists with advanced requirements:
        - At least 1/8 of tracks must be by the seed artist
        - Tracks ordered by sequential sonic similarity
        - No artist plays twice in a row

        Args:
            artist_seeds: Dict mapping artist name to their seed tracks
            history: Play history for filtering
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info(f"{'='*70}")
        logger.info(f"Generating playlists from single artists:")
        logger.info(f"{'='*70}\n")

        playlists = []

        for idx, (artist, seeds) in enumerate(artist_seeds.items(), 1):
            logger.info(f"Playlist {idx}: {artist}")

            for seed in seeds:
                logger.info(f"  Seed: {artist} - {seed.get('title')}")

            # Collect genres from seeds
            all_genres = []
            for seed in seeds:
                all_genres.extend(seed.get('genres', []))
            seen = set()
            unique_genres = [g for g in all_genres if not (g in seen or seen.add(g))]

            # DS pipeline override (if enabled)
            target_playlist_size = self.config.get('playlists', 'tracks_per_playlist', 30)
            ds_tracks = self._maybe_generate_ds_playlist(
                seed_track_id=seeds[0].get('rating_key') if seeds else None,
                target_length=target_playlist_size,
                pipeline_override=pipeline_override,
                mode_override=ds_mode_override or ("dynamic" if dynamic else None),
                seed_artist=artist,
                anchor_seed_tracks=seeds,  # Pass full seed track info for title+artist resolution
            )
            filtered_ds_tracks = ds_tracks
            if self.lastfm:
                scrobbles = self._get_lastfm_scrobbles_raw()
                filtered_ds_tracks = self._filter_by_scrobbles(
                    ds_tracks,
                    scrobbles,
                    lookback_days=self.config.recently_played_lookback_days,
                    exempt_tracks=seeds,
                )
            else:
                filtered_ds_tracks = self.filter_tracks(ds_tracks, history, exempt_tracks=seeds)

            final_tracks = self._ensure_seed_tracks_present(seeds, filtered_ds_tracks, target_playlist_size)

            if not final_tracks:
                raise RuntimeError("DS pipeline tracks filtered out by freshness rules for artist playlist generation.")

            title = self._generate_playlist_title(artist, "", unique_genres)
            self._print_playlist_report(final_tracks, artist_name=artist, dynamic=dynamic)
            playlists.append(
                {
                    'title': title,
                    'artists': (artist,),
                    'genres': unique_genres,
                    'tracks': final_tracks,
                }
            )
            logger.info(f"  Title: {title}")
            logger.info(f"  Final: {len(final_tracks)} tracks from {len(set(t.get('artist') for t in final_tracks))} artists")
            logger.info(f"  Seed artist ({artist}): {sum(1 for t in final_tracks if t.get('artist') == artist)} tracks\n")

        return playlists

    def _get_additional_artist_tracks(self, artist: str, history: List[Dict[str, Any]],
                                      existing_tracks: List[Dict[str, Any]],
                                      count: int) -> List[Dict[str, Any]]:
        """
        Get additional tracks from the seed artist to meet quota

        Args:
            artist: Artist name
            history: Listening history
            existing_tracks: Tracks already in the playlist
            count: Number of additional tracks needed

        Returns:
            List of additional artist tracks
        """
        existing_keys = {t.get('rating_key') for t in existing_tracks}
        artist_tracks_from_history = [t for t in history if t.get('artist') == artist
                                      and t.get('rating_key') not in existing_keys]

        # Sort by play count
        artist_tracks_from_history.sort(key=lambda x: x.get('play_count', 0), reverse=True)

        # Return up to 'count' tracks
        return artist_tracks_from_history[:count]

    def _select_canonical_track(self, artist_tracks: List[Dict[str, Any]], target_title: str) -> Optional[Dict[str, Any]]:
        """
        Select a canonical track for a given artist/title by normalizing titles and
        preferring non-live/demo/remix variants.

        Phase 2: Delegates to utils.select_canonical_track()
        """
        return utils.select_canonical_track(
            artist_tracks=artist_tracks,
            target_title=target_title,
        )

    def _order_by_sequential_similarity(self, tracks: List[Dict[str, Any]],
                                        seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order tracks so each track is sonically similar to adjacent tracks

        In dynamic mode, this ensures genre-matched songs are interleaved based on
        their sonic similarity rather than clustered at the end.

        Uses a greedy nearest-neighbor approach starting from seed tracks

        Args:
            tracks: Tracks to order
            seeds: Seed tracks to start from

        Returns:
            Ordered list of tracks

        Phase 7: Delegates to ordering.order_by_sequential_similarity()
        """
        result = ordering.order_by_sequential_similarity(
            tracks=tracks,
            seeds=seeds,
            library_client=self.library,
            limit_similar_tracks=self.config.limit_similar_tracks,
        )
        return result.ordered_tracks

    def _remove_consecutive_artists(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorder tracks to ensure no artist plays twice in a row

        Args:
            tracks: Ordered tracks

        Returns:
            Tracks with no consecutive artists

        Phase 6: Delegates to diversity.remove_consecutive_artists()
        """
        return diversity.remove_consecutive_artists(tracks=tracks)



    def _enforce_artist_window(self, tracks: List[Dict[str, Any]], window_size: int = 8) -> List[Dict[str, Any]]:
        """
        Enforce artist window constraint on track sequence

        Ensures no artist appears within window_size tracks of their last appearance.
        When violations found, attempts to swap with later tracks to fix.

        Args:
            tracks: Ordered track list (from TSP or other ordering)
            window_size: Number of tracks to check back

        Returns:
            Track list with artist window violations removed/fixed

        Phase 6: Delegates to diversity.enforce_artist_window()
        """
        return diversity.enforce_artist_window(
            tracks=tracks,
            window_size=window_size,
        )


    def _limit_artist_frequency_in_window(self, tracks: List[Dict[str, Any]], window_size: int = None, max_per_window: int = None) -> List[Dict[str, Any]]:
        """
        Ensure no artist appears more than max_per_window times within any window_size span

        This version preserves the interleaving of sonic/genre tracks by considering
        source when there are multiple valid candidates.

        Args:
            tracks: Ordered tracks
            window_size: Size of the sliding window (uses config if None)
            max_per_window: Maximum appearances per artist in window (uses config if None)

        Returns:
            Tracks with artist frequency constraint enforced

        Phase 6: Delegates to diversity.limit_artist_frequency_in_window()
        """
        if window_size is None:
            window_size = self.config.artist_window_size
        if max_per_window is None:
            max_per_window = self.config.max_artist_per_window

        return diversity.limit_artist_frequency_in_window(
            tracks=tracks,
            window_size=window_size,
            max_per_window=max_per_window,
        )

    def _get_diverse_seed_pool(self, history: List[Dict[str, Any]],
                               pool_size: int) -> List[Dict[str, Any]]:
        """Get a larger pool of diverse seed tracks with genre clustering"""
        play_counts = Counter()
        track_metadata = {}

        for track in history:
            key = track.get('rating_key')
            if key:
                play_counts[key] += 1
                track_metadata[key] = track

        # Get diverse seeds
        seeds = self._select_diverse_seeds(play_counts, track_metadata, pool_size)

        return seeds

    def _get_similar_artists(self, artist_name: str) -> List[str]:
        """
        Get similar artists (Last.FM disabled; returns empty list)

        Args:
            artist_name: Artist name to find similar artists for

        Returns:
            List of similar artist names
        """
        return []

    def _get_similar_genres(self, genre: str) -> List[str]:
        """
        Get similar genres (Last.FM disabled; returns empty list)

        Args:
            genre: Genre name to find similar genres for

        Returns:
            List of similar genre names
        """
        return []

    def _calculate_artist_similarity(self, seed1: Dict[str, Any],
                                     seed2: Dict[str, Any]) -> float:
        """
        Calculate artist similarity score using genre overlap only

        Args:
            seed1: First seed track with artist and genres
            seed2: Second seed track with artist and genres

        Returns:
            Similarity score (0.0 to 1.0)
        """
        artist1 = seed1.get('artist', 'Unknown')
        artist2 = seed2.get('artist', 'Unknown')

        # Strategy: genre overlap
        genres1 = set(seed1.get('genres', []))
        genres2 = set(seed2.get('genres', []))

        if genres1 and genres2:
            direct_overlap = len(genres1 & genres2)
            if direct_overlap > 0:
                # Jaccard similarity for direct overlap
                jaccard = direct_overlap / len(genres1 | genres2)
                return jaccard

        return 0.0  # No similarity found

    def _create_diverse_playlists(self, all_seeds: List[Dict[str, Any]],
                                  history: List[Dict[str, Any]],
                                  count: int) -> List[List[Dict[str, Any]]]:
        """
        Create distinct playlists using genre-clustered seed subsets

        Args:
            all_seeds: Large pool of seed tracks (with genres)
            history: Play history for filtering
            count: Number of playlists to create

        Returns:
            List of distinct playlists
        """
        # Cluster seeds by genre
        genre_clusters = self._cluster_seeds_by_genre(all_seeds, count)

        playlists = []

        for i, cluster_seeds in enumerate(genre_clusters, 1):
            if not cluster_seeds:
                logger.warning(f"No seeds in cluster {i}")
                continue

            # Log genre info for this cluster
            cluster_genres = set()
            for seed in cluster_seeds:
                cluster_genres.update(seed.get('genres', [])[:2])

            logger.info(f"Playlist {i} cluster genres: {', '.join(list(cluster_genres)[:5])}")
            logger.info(f"Playlist {i} seed artists: {[s['artist'] for s in cluster_seeds]}")

            # Generate similar tracks for this seed subset
            similar_tracks = self.generate_similar_tracks(cluster_seeds)

            # Add seed tracks to the candidate pool (they're guaranteed to be in playlist)
            all_tracks = cluster_seeds + similar_tracks

            # Filter out recently played (seeds are exempt)
            fresh_tracks = self.filter_tracks(all_tracks, history, exempt_tracks=cluster_seeds)

            # Diversify
            diverse_tracks = self.diversify_tracks(fresh_tracks)

            if diverse_tracks:
                playlists.append(diverse_tracks)
                logger.info(f"Playlist {i}: {len(diverse_tracks)} tracks from {len(set(t.get('artist') for t in diverse_tracks))} artists")
            else:
                logger.warning(f"No tracks for playlist {i}")

        return playlists

    def _cluster_seeds_by_genre(self, seeds: List[Dict[str, Any]],
                                num_clusters: int) -> List[List[Dict[str, Any]]]:
        """
        Cluster seeds by genre similarity using stored metadata

        Args:
            seeds: List of seed tracks with genre tags
            num_clusters: Number of clusters (playlists) to create

        Returns:
            List of seed lists (one per cluster)
        """
        logger.info(f"Clustering {len(seeds)} seeds into {num_clusters} genre-similar groups...")

        seed_count_per_playlist = self.config.get('playlists', 'seed_count', default=5)

        # If we don't have enough seeds, just split evenly
        if len(seeds) < num_clusters * 2:
            logger.warning("Not enough seeds for proper clustering, using simple split")
            return self._simple_split_seeds(seeds, num_clusters, seed_count_per_playlist)

        # Build similarity matrix between all seeds
        logger.info("="*60)
        logger.info("Building artist similarity matrix using metadata genres...")
        logger.info("="*60)
        similarity_matrix = {}
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i >= j:
                    continue
                similarity = self._calculate_artist_similarity(seed1, seed2)
                similarity_matrix[(i, j)] = similarity

        logger.info(f"\nSimilarity matrix complete - {len(similarity_matrix)} pairs calculated")
        logger.info("="*60)

        # Greedy clustering: Start with most similar seeds
        clusters = []
        assigned = set()

        # Sort seed pairs by similarity (highest first)
        sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)

        logger.info("Starting greedy clustering (similarity threshold = 0.2):")
        logger.info(f"Top 10 most similar pairs:")
        for (i, j), sim in sorted_pairs[:10]:
            logger.info(f"  {seeds[i]['artist']} <-> {seeds[j]['artist']}: {sim:.2f}")

        # Build initial clusters from highly similar seeds
        for (i, j), similarity in sorted_pairs:
            if i in assigned or j in assigned:
                continue

            if similarity >= 0.2:  # Threshold for considering seeds similar
                # Find which cluster to add to, or create new one
                cluster_found = False
                for cluster_idx, cluster in enumerate(clusters):
                    if len(cluster) < seed_count_per_playlist:
                        # Check if both seeds are similar to existing cluster members
                        cluster_compatible = True
                        for existing_idx in cluster:
                            # Check compatibility with cluster
                            sim_i = similarity_matrix.get((min(i, existing_idx), max(i, existing_idx)), 0.0)
                            sim_j = similarity_matrix.get((min(j, existing_idx), max(j, existing_idx)), 0.0)
                            if sim_i < 0.2 or sim_j < 0.2:
                                cluster_compatible = False
                                break

                        if cluster_compatible:
                            logger.info(f"Adding pair to cluster {cluster_idx+1}: {seeds[i]['artist']} + {seeds[j]['artist']} (sim={similarity:.2f})")
                            cluster.append(i)
                            cluster.append(j)
                            assigned.add(i)
                            assigned.add(j)
                            cluster_found = True
                            break

                if not cluster_found and len(clusters) < num_clusters:
                    # Create new cluster
                    logger.info(f"Creating NEW cluster {len(clusters)+1} with: {seeds[i]['artist']} + {seeds[j]['artist']} (sim={similarity:.2f})")
                    clusters.append([i, j])
                    assigned.add(i)
                    assigned.add(j)

        # Fill remaining slots in clusters with compatible seeds
        logger.info("Filling remaining cluster slots with best-fit seeds:")
        for cluster_idx, cluster in enumerate(clusters):
            while len(cluster) < seed_count_per_playlist and len(assigned) < len(seeds):
                best_candidate = None
                best_avg_similarity = 0.0

                for idx in range(len(seeds)):
                    if idx in assigned:
                        continue

                    # Calculate average similarity to cluster members
                    similarities = []
                    for cluster_member_idx in cluster:
                        sim = similarity_matrix.get((min(idx, cluster_member_idx), max(idx, cluster_member_idx)), 0.0)
                        similarities.append(sim)

                    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                    if avg_sim > best_avg_similarity:
                        best_avg_similarity = avg_sim
                        best_candidate = idx

                if best_candidate is not None:
                    cluster_members = [seeds[i]['artist'] for i in cluster]
                    logger.info(f"Adding to cluster {cluster_idx+1}: {seeds[best_candidate]['artist']} (avg_sim={best_avg_similarity:.2f} to {cluster_members[:2]}...)")
                    cluster.append(best_candidate)
                    assigned.add(best_candidate)
                else:
                    break

        # Create any remaining clusters with unassigned seeds
        unassigned = [i for i in range(len(seeds)) if i not in assigned]
        while unassigned and len(clusters) < num_clusters:
            new_cluster = []
            for _ in range(min(seed_count_per_playlist, len(unassigned))):
                if unassigned:
                    new_cluster.append(unassigned.pop(0))
            if new_cluster:
                clusters.append(new_cluster)

        # Convert indices back to seed objects
        seed_clusters = [[seeds[idx] for idx in cluster] for cluster in clusters]

        # Log cluster composition
        for i, cluster in enumerate(seed_clusters, 1):
            if cluster:
                cluster_genres = Counter()
                for seed in cluster:
                    for genre in seed.get('genres', ['unknown'])[:3]:
                        cluster_genres[genre] += 1
                logger.info(f"Cluster {i} genres: {dict(cluster_genres.most_common(5))}")
                logger.debug(f"Cluster {i} artists: {[s.get('artist') for s in cluster]}")

        return seed_clusters

    def _simple_split_seeds(self, seeds: List[Dict[str, Any]], num_clusters: int,
                           seeds_per_cluster: int) -> List[List[Dict[str, Any]]]:
        """Fallback: Simple split when not enough seeds for clustering"""
        clusters = []
        for i in range(0, len(seeds), seeds_per_cluster):
            cluster = seeds[i:i + seeds_per_cluster]
            if cluster:
                clusters.append(cluster)
            if len(clusters) >= num_clusters:
                break
        return clusters

    def _get_lastfm_history(self) -> List[Dict[str, Any]]:
        """
        Get listening history from Last.FM and match to library

        Returns:
            List of matched tracks with play counts
        """
        logger.info("Fetching Last.FM listening history")

        # Get history from Last.FM
        history_days = self.config.lastfm_history_days
        lastfm_tracks = self.lastfm.get_recent_tracks(days=history_days)

        if not lastfm_tracks:
            logger.warning("No Last.FM history found")
            return []

        logger.info(f"Retrieved {len(lastfm_tracks)} tracks from Last.FM")

        # Match to library
        matched_tracks: List[Dict[str, Any]] = []
        try:
            matched_tracks = self.matcher.match_lastfm_to_library(lastfm_tracks)
            logger.debug("Last.FM history match stats: raw=%d matched=%d", len(lastfm_tracks), len(matched_tracks))
        except Exception as exc:
            logger.warning("Last.FM history matching failed (%s); skipping matched history.", exc, exc_info=True)
            return []

        if not matched_tracks:
            logger.warning("No tracks could be matched to library")
            return []

        # Aggregate play counts
        aggregated = self.matcher.aggregate_play_counts(matched_tracks)

        return aggregated

    def _get_lastfm_scrobbles_raw(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get raw Last.FM scrobbles without library matching (for recency exclusion only)."""
        if not self.lastfm:
            return []
        history_days = self.config.lastfm_history_days
        scrobbles = self.lastfm.get_recent_tracks(days=history_days, use_cache=use_cache)
        if not scrobbles:
            logger.info("No Last.FM scrobbles found for recency filtering")
            return []
        logger.info("Retrieved %d raw scrobbles from Last.FM for recency filtering", len(scrobbles))
        return scrobbles

    def _get_local_history(self) -> List[Dict[str, Any]]:
        """
        Get listening history from local library (fallback method)

        Returns:
            List of tracks from play history
        """
        logger.info("Fetching local listening history (Last.FM not available)")

        history_days = self.config.get('playlists', 'history_days', default=14)
        history = self.library.get_play_history(None, days=history_days)

        return history
    
    def _split_into_playlists(self, tracks: List[Dict[str, Any]], 
                             count: int) -> List[List[Dict[str, Any]]]:
        """
        Split tracks into multiple playlists
        
        Args:
            tracks: Available tracks
            count: Number of playlists to create
            
        Returns:
            List of track lists
        """
        tracks_per_playlist = self.config.get('playlists', 'tracks_per_playlist', default=30)
        
        # Sort by weight (higher weight = more likely based on listening)
        sorted_tracks = sorted(tracks, key=lambda x: x.get('weight', 0), reverse=True)
        
        playlists = []
        
        for i in range(count):
            # Create playlist with different strategies
            if i == 0:
                # First playlist: Top weighted tracks
                playlist = sorted_tracks[:tracks_per_playlist]
            elif i == 1:
                # Second playlist: More diverse selection
                playlist = self._select_diverse(sorted_tracks, tracks_per_playlist)
            else:
                # Additional playlists: Random sampling
                population = sorted_tracks[:len(sorted_tracks)//2]
                sample_size = min(tracks_per_playlist, len(population))
                playlist = random.sample(population, sample_size) if sample_size > 0 else []
            
            if playlist:
                playlists.append(playlist)
                logger.info(f"Playlist {i+1}: {len(playlist)} tracks")
        
        return playlists
    
    def _select_diverse(self, tracks: List[Dict[str, Any]], 
                       count: int) -> List[Dict[str, Any]]:
        """
        Select tracks ensuring artist diversity
        
        Args:
            tracks: Available tracks
            count: Number to select
            
        Returns:
            Diverse selection of tracks
        """
        selected = []
        used_artists = set()
        
        # First pass: one track per artist
        for track in tracks:
            artist = track.get('artist')
            if artist not in used_artists:
                selected.append(track)
                used_artists.add(artist)
                
                if len(selected) >= count:
                    break
        
        # Second pass: fill remaining slots
        if len(selected) < count:
            remaining = [t for t in tracks if t not in selected]
            needed = count - len(selected)
            selected.extend(remaining[:needed])
        
        return selected

    def _print_playlist_report(
        self,
        tracks: List[Dict[str, Any]],
        artist_name: str = None,
        dynamic: bool = False,
        verbose_edges: bool = False,
    ):
        """
        Print detailed track report showing how each track was selected

        Args:
            tracks: Final playlist tracks
            artist_name: Name of seed artist (if applicable)
            dynamic: Whether dynamic mode was used
            verbose_edges: Whether to print per-edge scores when available (DS)
        """
        # Phase 9: Delegate to reporter module
        reporter.print_playlist_report(
            tracks=tracks,
            artist_name=artist_name,
            dynamic=dynamic,
            verbose_edges=verbose_edges,
            last_ds_report=getattr(self, "_last_ds_report", None),
            last_scope=getattr(self, "_last_scope", None),
            last_ds_mode=getattr(self, "_last_ds_mode", None),
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Playlist Generator module loaded")
