"""
Playlist Generator - Core logic for creating Data Science-powered playlists
"""
from typing import List, Dict, Any, Tuple, Optional, Set, Sequence
from collections import Counter, defaultdict
import random
import logging
import time
import os
import numpy as np
from .artist_utils import extract_primary_artist
from .similarity_calculator import SimilarityCalculator
from .string_utils import normalize_artist_key, normalize_genre, normalize_song_title
from .string_utils import normalize_match_string
from .title_dedupe import TitleDedupeTracker
from src.features.artifacts import load_artifact_bundle
from src.similarity.hybrid import transition_similarity_end_to_start
from src.similarity.sonic_variant import compute_sonic_variant_norm, get_variant_from_env, resolve_sonic_variant
from src.playlist.ds_pipeline_runner import DsRunResult, generate_playlist_ds as run_ds_pipeline
from src.playlist.artist_style import (
    ArtistStyleConfig,
    build_balanced_candidate_pool,
    cluster_artist_tracks,
    get_internal_connectors,
    order_clusters,
)
from src.playlist.pier_bridge_builder import PierBridgeConfig, resolve_pier_bridge_tuning
from src.playlist.config import default_ds_config, get_min_sonic_similarity
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


# DS pipeline config wiring helper (CLI/GUI share this codepath).
def build_ds_overrides(ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the `overrides` dict passed into the DS pipeline from `playlists.ds_pipeline`.

    This centralizes config wiring so CLI and GUI runs resolve pier-bridge tuning
    consistently (including `ds_pipeline.pier_bridge.*`).
    """
    legacy_overrides = ds_cfg.get("overrides") or {}
    overrides: Dict[str, Any] = {
        "scoring": ds_cfg.get("scoring", {}),
        "constraints": ds_cfg.get("constraints", {}),
        "candidate_pool": ds_cfg.get("candidate_pool", {}),
        "pier_bridge": ds_cfg.get("pier_bridge", {}),
        "repair": ds_cfg.get("repair", {}),
        "tower_weights": ds_cfg.get("tower_weights"),
        "transition_weights": ds_cfg.get("transition_weights"),
        "tower_pca_dims": ds_cfg.get("tower_pca_dims"),
        "embedding": ds_cfg.get("embedding", {}),
    }

    # Merge legacy overrides (if any) on top
    for key in ["candidate", "construct", "repair"]:
        if key in legacy_overrides:
            overrides[key] = legacy_overrides[key]

    # Back-compat: allow `center_transitions` at top-level ds_pipeline or in constraints.
    center_transitions = bool(ds_cfg.get("center_transitions", False))
    if overrides.get("constraints", {}).get("center_transitions"):
        center_transitions = True
    if center_transitions:
        constraints_updates = {**overrides.get("constraints", {}), "center_transitions": True}
        overrides = {**overrides, "constraints": constraints_updates}

    return overrides


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


def safe_get_artist_key(track: Dict[str, Any]) -> str:
    """Safely get normalized artist key from a track dictionary."""
    return utils.safe_get_artist_key(track)


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
        # Runtime flags (CLI/GUI) for optional pier-bridge backoff + run audits.
        # Default OFF; config.yaml keys can still enable these without flags.
        self._pb_backoff_enabled: bool = False
        self._audit_run_enabled: bool = False
        self._audit_run_dir: Optional[str] = None

    def _ensure_metadata_client(self):
        if self.metadata is None:
            try:
                db_path = self.config.get('library', {}).get('database_path', 'data/metadata.db')
                from src.metadata_client import MetadataClient

                self.metadata = MetadataClient(db_path)
            except Exception as exc:
                logger.warning("Metadata DB unavailable (%s)", exc)
                return None
        return self.metadata

    def _get_blacklisted_track_ids(self) -> Set[str]:
        """Fetch blacklisted track ids from the metadata database."""
        metadata = self._ensure_metadata_client()
        if metadata is None:
            return set()
        try:
            return set(metadata.fetch_blacklisted_track_ids())
        except Exception as exc:
            logger.warning("Failed to fetch blacklist ids (%s)", exc)
            return set()

    def _build_duration_exclusions_for_ds(
        self,
        *,
        seed_track_ids: Sequence[str],
        allowed_track_ids: Optional[List[str]],
        ds_cfg: Dict[str, Any],
    ) -> Set[str]:
        metadata = self._ensure_metadata_client()
        if metadata is None:
            return set()

        min_seconds = self.config.get("playlists", "min_track_duration_seconds", default=47)
        min_ms = int(min_seconds * 1000)
        max_ms = 0
        cutoff_multiplier = float(
            (ds_cfg.get("candidate_pool") or {}).get("duration_cutoff_multiplier", 2.5)
        )

        seed_durations: List[int] = []
        if seed_track_ids:
            seed_duration_map = metadata.fetch_track_durations([str(t) for t in seed_track_ids])
            for tid in seed_track_ids:
                dur = int(seed_duration_map.get(str(tid), 0))
                if dur > 0:
                    seed_durations.append(dur)

        seed_median_ms = 0
        if seed_durations:
            seed_durations.sort()
            mid = len(seed_durations) // 2
            if len(seed_durations) % 2 == 1:
                seed_median_ms = seed_durations[mid]
            else:
                seed_median_ms = int((seed_durations[mid - 1] + seed_durations[mid]) / 2)

        cutoff_ms = int(seed_median_ms * cutoff_multiplier) if seed_median_ms else 0

        excluded: Set[str] = set()
        if allowed_track_ids:
            allowed_set = {str(t) for t in allowed_track_ids}
            duration_map = metadata.fetch_track_durations(list(allowed_set))
            for tid in allowed_set:
                duration_ms = int(duration_map.get(tid, 0))
                if duration_ms <= 0:
                    excluded.add(tid)
                    continue
                if min_ms > 0 and duration_ms < min_ms:
                    excluded.add(tid)
                    continue
                if cutoff_ms > 0 and duration_ms > cutoff_ms:
                    excluded.add(tid)
        else:
            excluded = metadata.fetch_track_ids_by_duration_limits(
                min_ms=min_ms,
                max_ms=max_ms,
                cutoff_ms=cutoff_ms,
            )

        if excluded:
            logger.info(
                "Duration exclusions (pre-order): excluded=%d min_ms=%s cutoff_ms=%s seed_median_ms=%s",
                len(excluded),
                min_ms if min_ms > 0 else "n/a",
                cutoff_ms if cutoff_ms > 0 else "n/a",
                seed_median_ms if seed_median_ms > 0 else "n/a",
            )
        return excluded

    def _apply_blacklist_to_ids(
        self,
        *,
        allowed_track_ids: Optional[List[str]],
        excluded_track_ids: Optional[Set[str]],
        blacklist_ids: Set[str],
        context: str,
    ) -> tuple[Optional[List[str]], Set[str], int]:
        """Apply blacklist to allowed/excluded sets and return updated values."""
        if not blacklist_ids:
            return allowed_track_ids, excluded_track_ids or set(), 0
        removed_from_allowed = 0
        if allowed_track_ids:
            before = len(allowed_track_ids)
            allowed_track_ids = [
                tid for tid in allowed_track_ids if str(tid) not in blacklist_ids
            ]
            removed_from_allowed = before - len(allowed_track_ids)
        updated_excluded = set(excluded_track_ids or set())
        updated_excluded.update(blacklist_ids)
        excluded_from_pool = removed_from_allowed if allowed_track_ids is not None else len(blacklist_ids)
        logger.info(
            "Blacklist exclusions: context=%s blacklisted_total=%d excluded_from_pool=%d",
            context,
            len(blacklist_ids),
            excluded_from_pool,
        )
        return allowed_track_ids, updated_excluded, removed_from_allowed

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
        pier_bridge_config=None,
        internal_connector_ids: Optional[List[str]] = None,
        internal_connector_max_per_segment: int = 0,
        internal_connector_priority: bool = True,
        artist_style_enabled: bool = False,
        artist_playlist: bool = False,
        pool_source: Optional[str] = None,
        dry_run: bool = False,
        audit_context_extra: Optional[Dict[str, Any]] = None,
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

        overrides = build_ds_overrides(ds_cfg)
        # Runtime flags (CLI/GUI) can enable optional behaviors without editing config.yaml.
        # These merge on top of config-driven `playlists.ds_pipeline.pier_bridge.*`.
        pb_overrides = overrides.get("pier_bridge")
        if not isinstance(pb_overrides, dict):
            pb_overrides = {}
        if getattr(self, "_pb_backoff_enabled", False):
            ih = pb_overrides.get("infeasible_handling")
            if not isinstance(ih, dict):
                ih = {}
            ih["enabled"] = True
            pb_overrides["infeasible_handling"] = ih
        if getattr(self, "_audit_run_enabled", False):
            ar = pb_overrides.get("audit_run")
            if not isinstance(ar, dict):
                ar = {}
            ar["enabled"] = True
            if getattr(self, "_audit_run_dir", None):
                ar["out_dir"] = str(getattr(self, "_audit_run_dir"))
            pb_overrides["audit_run"] = ar
        overrides["pier_bridge"] = pb_overrides

        seed_to_use = str(seed_track_id)
        anchor_seed_ids_resolved: Optional[List[str]] = None
        try:
            bundle = load_artifact_bundle(artifact_path)
            if seed_track_id not in bundle.track_id_to_index and seed_artist:
                artist_norm = normalize_artist_key(seed_artist)
                # try track_artists match
                if bundle.track_artists is not None:
                    for idx, artist in enumerate(bundle.track_artists):
                        if normalize_artist_key(str(artist)) == artist_norm:
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
                    seed_artist_name = normalize_artist_key(str(seed_track.get('artist', '')))
                    if not seed_title or not seed_artist_name:
                        continue
                    # Find matching track in bundle
                    for idx in range(len(bundle.track_artists)):
                        bundle_artist = normalize_artist_key(str(bundle.track_artists[idx]))
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

            blacklist_ids = self._get_blacklisted_track_ids()
            if blacklist_ids:
                if str(seed_to_use) in blacklist_ids:
                    raise ValueError(
                        f"Seed track {seed_to_use} is blacklisted; select another seed."
                    )
                if anchor_seed_ids_resolved:
                    before = len(anchor_seed_ids_resolved)
                    anchor_seed_ids_resolved = [
                        sid
                        for sid in anchor_seed_ids_resolved
                        if str(sid) not in blacklist_ids
                    ]
                    removed = before - len(anchor_seed_ids_resolved)
                    if removed:
                        logger.info(
                            "Blacklist exclusions: removed %d anchor seeds",
                            removed,
                        )
                allowed_track_ids, excluded_track_ids, _ = self._apply_blacklist_to_ids(
                    allowed_track_ids=allowed_track_ids,
                    excluded_track_ids=excluded_track_ids,
                    blacklist_ids=blacklist_ids,
                    context="ds_pipeline",
                )
                if allowed_track_ids is not None and not allowed_track_ids:
                    raise ValueError("All allowed tracks were blacklisted; aborting DS run.")

            seed_ids_for_duration = [str(seed_to_use)]
            if anchor_seed_ids_resolved:
                seed_ids_for_duration.extend([str(sid) for sid in anchor_seed_ids_resolved])
            duration_excluded = self._build_duration_exclusions_for_ds(
                seed_track_ids=seed_ids_for_duration,
                allowed_track_ids=allowed_track_ids,
                ds_cfg=ds_cfg,
            )
            if duration_excluded:
                updated_excluded = set(excluded_track_ids or set())
                updated_excluded.update(duration_excluded)
                excluded_track_ids = updated_excluded
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
        if not genre_enabled:
            genre_weight = 0.0
            sonic_weight = genre_cfg.get('sonic_weight', 1.0) or 1.0

        mode_overrides_active = bool(
            playlists_cfg.get("genre_mode") or playlists_cfg.get("sonic_mode")
        )
        if mode == "sonic_only" and not mode_overrides_active:
            genre_enabled = False
            min_genre_sim = None
            genre_method = None
            genre_weight = 0.0
            sonic_weight = 1.0

        if artist_style_enabled and not allowed_track_ids:
            raise ValueError(
                "Artist style mode enabled but allowed_track_ids is empty; refusing to run DS without clamp."
            )

        pool_source_effective = pool_source or ("restricted" if allowed_track_ids else "unrestricted")

        logger.info(
            "Invoking DS pipeline seed=%s mode=%s target_length=%d allowed_ids=%d artist_style_enabled=%s pool_source=%s genre_gate=%s",
            seed_to_use,
            mode,
            target_length,
            len(allowed_track_ids or []),
            bool(artist_style_enabled),
            pool_source_effective,
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

        audit_context_extra_effective: Optional[Dict[str, Any]] = None
        if audit_context_extra is not None:
            audit_context_extra_effective = dict(audit_context_extra)
        else:
            audit_context_extra_effective = {}
        recency_extra = audit_context_extra_effective.get("recency")
        if not isinstance(recency_extra, dict):
            recency_extra = {}
        if "lookback_days" not in recency_extra:
            recency_extra["lookback_days"] = int(getattr(self.config, "recently_played_lookback_days", 0))
        audit_context_extra_effective["recency"] = recency_extra
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
            pier_bridge_config=pier_bridge_config,
            dry_run=bool(dry_run),
            pool_source=pool_source_effective,
            artist_style_enabled=bool(artist_style_enabled),
            artist_playlist=bool(artist_playlist),
            audit_context_extra=audit_context_extra_effective,
            # Genre similarity parameters
            sonic_weight=sonic_weight,
            genre_weight=genre_weight,
            min_genre_similarity=min_genre_sim,
            genre_method=genre_method,
            internal_connector_ids=internal_connector_ids,
            internal_connector_max_per_segment=internal_connector_max_per_segment,
            internal_connector_priority=internal_connector_priority,
        )

        tracks: List[Dict[str, Any]] = []
        ds_stats = getattr(ds_result, "playlist_stats", {}) or {}
        playlist_stats_only = ds_stats.get("playlist") or {}
        pool_seed_sonic = (ds_stats.get("candidate_pool") or {}).get("seed_sonic_sim_track_ids") or {}
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

        if pool_seed_sonic:
            for track in tracks:
                tid = track.get("rating_key") or track.get("id") or track.get("track_id")
                if tid is None:
                    continue
                sonic_val = pool_seed_sonic.get(str(tid))
                if sonic_val is None and str(tid) == str(ds_result.track_ids[0]):
                    sonic_val = 1.0  # ensure seed shows perfect similarity
                if sonic_val is not None:
                    track["sonic_similarity"] = sonic_val

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
                artist_key = normalize_artist_key(artist_name)
                if any(safe_get_artist_key(seed) == artist_key for seed in seeds):
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
            stage="candidate_pool",
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
            stage="candidate_pool",
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
            stage="candidate_pool",
        )

        # Convert filtered tracks to excluded IDs (tracks that were removed)
        filtered_ids = {str(t.get("rating_key")) for t in result.filtered_tracks if t.get("rating_key")}
        all_ids = {str(t.get("rating_key")) for t in candidates if t.get("rating_key")}
        excluded = all_ids - filtered_ids

        if seed_id and seed_id in excluded:
            logger.warning("Recency exclusions contained seed_id %s; preserving seed.", seed_id)

        return excluded

    def _compute_excluded_from_history(
        self,
        candidates: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        lookback_days: int,
        exempt_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> Set[str]:
        """
        Return rating_keys to exclude based on local play history.

        This is a pre-order operation (candidate_pool stage). The returned set
        contains track_ids that should be excluded from DS candidate selection.

        Phase 3: Delegates to filtering.filter_by_recently_played() and converts
        result to excluded IDs.
        """
        if (
            not history
            or lookback_days <= 0
            or not getattr(self.config, "recently_played_filter_enabled", False)
        ):
            return set()

        result = filtering.filter_by_recently_played(
            tracks=candidates,
            play_history=history,
            lookback_days=lookback_days,
            min_playcount=getattr(self.config, "recently_played_min_playcount", 0),
            exempt_tracks=exempt_tracks,
            stage="candidate_pool",
        )
        filtered_ids = {str(t.get("rating_key")) for t in result.filtered_tracks if t.get("rating_key")}
        all_ids = {str(t.get("rating_key")) for t in candidates if t.get("rating_key")}
        return all_ids - filtered_ids

    def _post_order_validate_ds_output(
        self,
        *,
        ordered_tracks: List[Dict[str, Any]],
        expected_length: int,
        excluded_track_ids: Set[str],
        exempt_pier_track_ids: Set[str],
        audit_path: Optional[str] = None,
    ) -> None:
        """
        Post-order validation for DS pier-bridge runs (NO mutations allowed).

        Invariant: recency filtering must occur pre-order (candidate selection).
        After DS ordering completes, we validate only; we never filter/drop/reorder.
        """
        ordered_ids: List[str] = []
        for t in ordered_tracks:
            tid = t.get("rating_key") or t.get("id") or t.get("track_id")
            if tid is not None:
                ordered_ids.append(str(tid))

        overlap = [
            tid
            for tid in ordered_ids
            if tid in excluded_track_ids and tid not in exempt_pier_track_ids
        ]
        logger.info(
            "stage=post_order_validation | recency_overlap=%d | final_size=%d | expected=%d",
            len(overlap),
            len(ordered_ids),
            int(expected_length),
        )

        errors: List[str] = []
        if expected_length > 0 and len(ordered_ids) != int(expected_length):
            errors.append(f"length_mismatch final={len(ordered_ids)} expected={int(expected_length)}")

        if overlap:
            offenders: List[str] = []
            for tid in overlap[:10]:
                track = next((x for x in ordered_tracks if str(x.get("rating_key") or x.get("id") or x.get("track_id")) == tid), None)
                if track:
                    artist = utils.sanitize_for_logging(str(track.get("artist", "")))
                    title = utils.sanitize_for_logging(str(track.get("title", "")))
                    offenders.append(f"{tid} ({artist} - {title})")
                else:
                    offenders.append(tid)
            errors.append(f"recency_overlap={len(overlap)} offenders={offenders}")

        if errors:
            msg = "post_order_validation_failed: " + " | ".join(errors)
            if audit_path:
                msg += f" (audit: {audit_path})"
            raise ValueError(msg)
    
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
        track_titles: Optional[List[str]] = None,
        dynamic: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        ds_mode_override: Optional[str] = None,
        artist_only: bool = False,
        anchor_seed_ids: Optional[List[str]] = None,
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

        # Filter to just this artist (normalized key match)
        artist_key = normalize_artist_key(artist_name)
        artist_tracks = [
            t for t in all_library_tracks
            if safe_get_artist_key(t) == artist_key
        ]

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

        fixed_seed_tracks: List[Dict[str, Any]] = []
        fixed_anchor_ids: Optional[List[str]] = None
        if anchor_seed_ids:
            fixed_anchor_ids = [str(tid) for tid in anchor_seed_ids if str(tid).strip()]
            if fixed_anchor_ids:
                if track_title:
                    logger.info("Both track_title and anchor_seed_ids provided; using anchor_seed_ids.")
                if track_titles:
                    logger.info("Both track_titles and anchor_seed_ids provided; using anchor_seed_ids.")
                if hasattr(self.library, "get_tracks_by_ids"):
                    fetched = self.library.get_tracks_by_ids(fixed_anchor_ids)
                    lookup = {str(t.get("rating_key")): t for t in fetched or []}
                    for tid in fixed_anchor_ids:
                        track = lookup.get(str(tid))
                        if track:
                            fixed_seed_tracks.append(track)
                else:
                    for tid in fixed_anchor_ids:
                        track = self.library.get_track_by_key(str(tid))
                        if track:
                            fixed_seed_tracks.append(track)

                if fixed_seed_tracks:
                    artist_key_norm = normalize_artist_key(artist_name)
                    mismatched = [
                        t for t in fixed_seed_tracks
                        if safe_get_artist_key(t) != artist_key_norm
                    ]
                    if mismatched:
                        logger.warning(
                            "Fixed pier seeds include non-matching artists (%d/%d); filtering to requested artist.",
                            len(mismatched),
                            len(fixed_seed_tracks),
                        )
                        fixed_seed_tracks = [
                            t for t in fixed_seed_tracks
                            if safe_get_artist_key(t) == artist_key_norm
                        ]
                    if not fixed_seed_tracks:
                        logger.warning(
                            "No fixed pier seeds matched artist '%s'; falling back to default seeds.",
                            artist_name,
                        )
                else:
                    logger.warning("No tracks found for anchor_seed_ids; falling back to default seeds.")
        elif track_titles:
            normalized_titles = [str(t).strip() for t in track_titles if str(t).strip()]
            if normalized_titles:
                seen_titles: set[str] = set()
                for title in normalized_titles:
                    key = title.casefold()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    selected = self._select_canonical_track(artist_tracks, title)
                    if selected:
                        fixed_seed_tracks.append(selected)
                    else:
                        logger.warning(
                            "Requested seed track '%s' not found for artist %s",
                            title,
                            artist_name,
                        )
                if fixed_seed_tracks:
                    logger.info(
                        "Using %d fixed seed tracks from title list for %s",
                        len(fixed_seed_tracks),
                        artist_name,
                    )
                else:
                    logger.warning(
                        "No requested seed tracks found for artist %s; falling back to default seeds.",
                        artist_name,
                    )
            else:
                logger.warning("track_titles provided but empty after normalization; falling back to default seeds.")

        # If a specific track title is provided, select canonical match and keep seeds focused
        seed_tracks: List[Dict[str, Any]] = []
        if fixed_seed_tracks:
            seed_tracks = fixed_seed_tracks
            logger.info("Using %d fixed pier seeds for %s", len(seed_tracks), artist_name)
        elif track_title:
            selected = self._select_canonical_track(artist_tracks, track_title)
            if selected:
                seed_tracks = [selected]
                logger.info("Using specified seed track: %s - %s", selected.get("artist"), selected.get("title"))
            else:
                logger.warning("Requested seed track '%s' not found for artist %s; falling back to random seeds.", track_title, artist_name)

        if not seed_tracks:
            # Get 4 seed tracks (2 most played + 2 random from top 10)
            # Since we don't have play counts, just pick 4 random tracks
            # Filter by duration before selecting (exclude short/long tracks)
            from src.playlist.filtering import is_valid_duration
            valid_tracks = [t for t in artist_tracks if is_valid_duration(t, min_seconds=47, max_seconds=720)]
            if len(valid_tracks) == 0:
                raise ValueError(f"Artist '{artist_name}' has no tracks in valid duration range (47s-720s)")
            if len(valid_tracks) < 4:
                logger.warning(f"Artist has only {len(valid_tracks)} valid-duration tracks (requested 4); using all valid tracks")
            import random
            seed_tracks = random.sample(valid_tracks, min(4, len(valid_tracks)))

        anchor_seed_ids_override = None
        if fixed_seed_tracks:
            anchor_seed_ids_override = [
                str(t.get("rating_key")) for t in seed_tracks if t.get("rating_key")
            ]
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
            # IMPORTANT: Filter artist_tracks by duration for DS candidate pool
            # This ensures no tracks outside valid duration range (47s-720s) are selected
            from src.playlist.filtering import is_valid_duration
            ds_candidates = [t for t in artist_tracks if is_valid_duration(t, min_seconds=47, max_seconds=720)]
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

        ds_tracks = None
        last_error = None

        # ─────────────────────────────────────────────────────────────────────
        # Style-aware artist mode (optional; config-gated)
        # ─────────────────────────────────────────────────────────────────────
        ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
        style_cfg_raw = ds_cfg.get("artist_style", {}) or {}
        style_cfg = ArtistStyleConfig(
            enabled=bool(style_cfg_raw.get("enabled", False)),
            cluster_k_min=style_cfg_raw.get("cluster_k_min", 3),
            cluster_k_max=style_cfg_raw.get("cluster_k_max", 6),
            cluster_k_heuristic_enabled=style_cfg_raw.get("cluster_k_heuristic_enabled", True),
            piers_per_cluster=style_cfg_raw.get("piers_per_cluster", 1),
            per_cluster_candidate_pool_size=style_cfg_raw.get("per_cluster_candidate_pool_size", 400),
            pool_balance_mode=style_cfg_raw.get("pool_balance_mode", "equal"),
            internal_connector_priority=style_cfg_raw.get("internal_connector_priority", True),
            internal_connector_max_per_segment=style_cfg_raw.get("internal_connector_max_per_segment", 2),
            bridge_floor_narrow=style_cfg_raw.get("bridge_floor", {}).get("narrow", 0.08),
            bridge_floor_dynamic=style_cfg_raw.get("bridge_floor", {}).get("dynamic", 0.03),
            bridge_weight=style_cfg_raw.get("bridge_score_weights", {}).get("bridge", 0.7),
            transition_weight=style_cfg_raw.get("bridge_score_weights", {}).get("transition", 0.3),
            genre_tiebreak_weight=style_cfg_raw.get("genre_tiebreak_weight", 0.05),
        )
        ds_mode_effective = ds_mode_override or ("dynamic" if dynamic else ds_cfg.get("mode", "dynamic"))
        artifact_path = ds_cfg.get("artifact_path")
        pool_source = "legacy"

        logger.info(
            "Artist style mode %s: artist=%s ds_mode=%s",
            "ENABLED" if style_cfg.enabled else "DISABLED",
            artist_name,
            ds_mode_effective,
        )

        using_artist_style = False
        pier_cfg = None
        style_seed_track_id: Optional[str] = None
        style_anchor_tracks: Optional[List[Dict[str, Any]]] = None
        style_anchor_ids: Optional[List[str]] = None
        style_allowed_track_ids: Optional[List[str]] = None
        internal_connector_ids: Optional[List[str]] = None

        if style_cfg.enabled and artifact_path and (not artist_only) and (not track_title) and not fixed_seed_tracks:
            try:
                bundle = load_artifact_bundle(artifact_path)
                sonic_cfg = self.config.get("playlists", "sonic", default={}) or {}
                sonic_variant_cfg = resolve_sonic_variant(
                    explicit_variant=getattr(self, "sonic_variant", None),
                    config_variant=ds_cfg.get("sonic_variant") or sonic_cfg.get("sim_variant"),
                )
                clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
                    bundle=bundle,
                    artist_name=artist_name,
                    cfg=style_cfg,
                    random_seed=ds_cfg.get("random_seed", 0),
                    sonic_variant=sonic_variant_cfg,
                )
                if not medoids:
                    raise ValueError("Style clustering returned no medoids")
                ordered_medoids = order_clusters(medoids, X_norm)
                cluster_piers = medoids_by_cluster

                # Global admission floor (same as DS candidate admission)
                min_sonic = get_min_sonic_similarity(ds_cfg.get("candidate_pool", {}), ds_mode_effective)
                artist_key_norm = normalize_artist_key(artist_name)

                external_pool = build_balanced_candidate_pool(
                    bundle=bundle,
                    cluster_piers=cluster_piers,
                    X_norm=X_norm,
                    per_cluster_size=style_cfg.per_cluster_candidate_pool_size,
                    pool_balance_mode=style_cfg.pool_balance_mode,
                    global_floor=min_sonic,
                    artist_key=artist_key_norm,
                )
                internal_connector_ids = (
                    get_internal_connectors(
                        bundle=bundle,
                        artist_key=artist_key_norm,
                        exclude_indices=medoids,
                        global_floor=min_sonic,
                        pier_indices=medoids,
                        X_norm=X_norm,
                    )
                    if style_cfg.internal_connector_priority
                    else []
                )
                pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]
                style_allowed_track_ids = list(dict.fromkeys(pier_ids + external_pool + list(internal_connector_ids or [])))
                if not style_allowed_track_ids:
                    raise ValueError("Artist style allowed pool empty")

                style_seed_track_id = pier_ids[0]
                style_anchor_ids = pier_ids[1:]
                style_anchor_tracks = [
                    {
                        "rating_key": str(bundle.track_ids[m]),
                        "title": str(bundle.track_titles[m] or ""),
                        "artist": str(bundle.track_artists[m] or artist_name),
                    }
                    for m in ordered_medoids
                ]
                style_summary = {
                    "artist": str(artist_name),
                    "ds_mode": str(ds_mode_effective),
                    "sonic_variant": str(sonic_variant_cfg),
                    "global_sonic_floor": float(min_sonic),
                    "clusters": [
                        {
                            "cluster_index": int(i),
                            "artist_track_count": int(len(members)),
                            "pier_track_ids": [str(bundle.track_ids[p]) for p in (medoids_by_cluster[i] if i < len(medoids_by_cluster) else [])],
                        }
                        for i, members in enumerate(clusters)
                    ],
                    "ordered_piers": [
                        {
                            "track_id": str(bundle.track_ids[m]),
                            "artist": str(bundle.track_artists[m] or artist_name),
                            "title": str(bundle.track_titles[m] or ""),
                        }
                        for m in ordered_medoids
                    ],
                    "allowed_ids_count": int(len(style_allowed_track_ids)),
                    "external_pool_count": int(len(external_pool)),
                    "internal_connectors_count": int(len(internal_connector_ids or [])),
                }
                center_transitions = bool(ds_cfg.get("center_transitions", False)) or bool(
                    ds_cfg.get("constraints", {}).get("center_transitions", False)
                )
                tw_raw = ds_cfg.get("transition_weights")
                transition_weights = None
                if isinstance(tw_raw, dict):
                    transition_weights = (
                        float(tw_raw.get("rhythm", 0.4)),
                        float(tw_raw.get("timbre", 0.35)),
                        float(tw_raw.get("harmony", 0.25)),
                    )
                elif isinstance(tw_raw, (list, tuple)) and len(tw_raw) == 3:
                    transition_weights = (
                        float(tw_raw[0]),
                        float(tw_raw[1]),
                        float(tw_raw[2]),
                    )
                ds_defaults = default_ds_config(ds_mode_effective, playlist_len=track_count, overrides=ds_cfg)
                pb_tuning = resolve_pier_bridge_tuning(ds_cfg, ds_mode_effective)

                # Artist-style can override per-mode pier-bridge weights, but defaults
                # should follow the global pier-bridge tuning for the mode.
                weight_bridge = float(pb_tuning["weight_bridge"])
                weight_transition = float(pb_tuning["weight_transition"])
                weights_raw = style_cfg_raw.get("bridge_score_weights")
                if isinstance(weights_raw, dict):
                    by_mode = weights_raw.get(ds_mode_effective)
                    if isinstance(by_mode, dict):
                        weight_bridge = float(by_mode.get("bridge", weight_bridge))
                        weight_transition = float(by_mode.get("transition", weight_transition))
                    else:
                        weight_bridge = float(weights_raw.get("bridge", weight_bridge))
                        weight_transition = float(weights_raw.get("transition", weight_transition))

                genre_tiebreak_weight = float(
                    style_cfg_raw.get("genre_tiebreak_weight", pb_tuning["genre_tiebreak_weight"])
                )
                bridge_floor = float(pb_tuning["bridge_floor"])
                bridge_floor_raw = style_cfg_raw.get("bridge_floor")
                if isinstance(bridge_floor_raw, dict):
                    mode_val = bridge_floor_raw.get(ds_mode_effective)
                    if isinstance(mode_val, (int, float)):
                        bridge_floor = float(mode_val)
                elif isinstance(bridge_floor_raw, (int, float)):
                    bridge_floor = float(bridge_floor_raw)

                pier_cfg = PierBridgeConfig(
                    transition_floor=float(ds_defaults.construct.transition_floor),
                    bridge_floor=bridge_floor,
                    center_transitions=bool(ds_defaults.construct.center_transitions),
                    transition_weights=transition_weights,
                    sonic_variant=sonic_variant_cfg,
                    weight_bridge=weight_bridge,
                    weight_transition=weight_transition,
                    genre_tiebreak_weight=genre_tiebreak_weight,
                    genre_penalty_threshold=float(pb_tuning["genre_penalty_threshold"]),
                    genre_penalty_strength=float(pb_tuning["genre_penalty_strength"]),
                )

                using_artist_style = True
                pool_source = "artist_style"
                logger.info(
                    "Artist style mode ENABLED: artist=%s ds_mode=%s clusters=%d piers=%d allowed_ids=%d internal_connectors=%d",
                    artist_name,
                    ds_mode_effective,
                    len(clusters),
                    len(ordered_medoids),
                    len(style_allowed_track_ids),
                    len(internal_connector_ids or []),
                )
            except Exception as exc:
                logger.warning(
                    "Artist style mode fallback to legacy (reason=%s)",
                    exc,
                    exc_info=True,
                )
                using_artist_style = False
                pool_source = "legacy_fallback"

        if using_artist_style and style_seed_track_id and style_allowed_track_ids:
            ds_tracks = self._maybe_generate_ds_playlist(
                seed_track_id=style_seed_track_id,
                target_length=track_count,
                mode_override=ds_mode_effective,
                seed_artist=artist_name,
                allowed_track_ids=style_allowed_track_ids,
                excluded_track_ids=excluded_ids or None,
                anchor_seed_tracks=style_anchor_tracks,
                anchor_seed_ids=style_anchor_ids,
                pier_bridge_config=pier_cfg,
                internal_connector_ids=internal_connector_ids,
                internal_connector_max_per_segment=style_cfg.internal_connector_max_per_segment,
                internal_connector_priority=style_cfg.internal_connector_priority,
                artist_style_enabled=True,
                artist_playlist=True,
                pool_source=pool_source,
                dry_run=bool(dry_run),
                audit_context_extra={"style_summary": style_summary},
            )
        else:
            logger.info("Artist style mode DISABLED: using legacy seed selection")
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
                        anchor_seed_ids=anchor_seed_ids_override,
                        artist_style_enabled=False,
                        artist_playlist=True,
                        pool_source=pool_source,
                        dry_run=bool(dry_run),
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

        # Skip seed insertion for pier-bridge mode - pier-bridge already handles seed placement
        # and only includes seeds that were found in the artifact bundle
        last_report = getattr(self, "_last_ds_report", None) or {}
        is_pier_bridge = (last_report.get("metrics") or {}).get("strategy") == "pier_bridge"

        # Post-order validation ONLY (no filtering/removal/reordering allowed).
        # Recency exclusions must be applied pre-order via DS `excluded_track_ids`.
        audit_path = None
        try:
            audit_path = (
                (getattr(self, "_last_ds_report", None) or {}).get("playlist_stats") or {}
            ).get("playlist", {}).get("audit_path")
        except Exception:
            audit_path = None

        pier_tracks_for_exemption = (
            style_anchor_tracks if (using_artist_style and style_anchor_tracks) else seed_tracks
        )
        exempt_pier_ids = {
            str(t.get("rating_key"))
            for t in (pier_tracks_for_exemption or [])
            if t.get("rating_key")
        }
        self._post_order_validate_ds_output(
            ordered_tracks=ds_tracks,
            expected_length=track_count,
            excluded_track_ids=set(excluded_ids or set()),
            exempt_pier_track_ids=exempt_pier_ids,
            audit_path=str(audit_path) if audit_path else None,
        )

        if is_pier_bridge:
            final_tracks = ds_tracks
            logger.debug("Pier-bridge mode: no post-order filtering (validation only).")
        else:
            final_tracks = self._ensure_seed_tracks_present(seed_tracks, ds_tracks, track_count)

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
                logger.info("Recency diag: post-order filtering disabled; no edge diff computed.")
        title = f"Auto: {artist_name}"
        self._print_playlist_report(final_tracks, artist_name=artist_name, dynamic=dynamic, verbose_edges=verbose)
        return {
            'title': title,
            'artists': (artist_name,),
            'genres': [],
            'tracks': final_tracks,
            'ds_report': getattr(self, "_last_ds_report", None),
        }

    def create_playlist_for_genre(
        self,
        genre_name: str,
        track_count: int = 30,
        dynamic: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        ds_mode_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single playlist for a specific genre without requiring listening history

        Args:
            genre_name: Name of the genre to create playlist for
            track_count: Target number of tracks in playlist
            dynamic: Use dynamic mode (lower genre similarity threshold)
            dry_run: Preview mode (for testing)
            verbose: Enable verbose edge scoring output
            ds_mode_override: Override DS mode (narrow/dynamic/discover/sonic_only)

        Returns:
            Playlist dictionary with tracks and metadata, or None if unable to create
        """
        from src.genre_normalization import normalize_genre_token

        # Normalize the genre name
        normalized_genre = normalize_genre_token(genre_name)
        if not normalized_genre:
            logger.warning(f"Genre '{genre_name}' could not be normalized")
            return None

        logger.info(f"Creating playlist for genre: {genre_name} (normalized: {normalized_genre})")

        # Get tracks matching this genre
        genre_tracks = self.library.get_tracks_for_genre(normalized_genre, limit=1000)

        if not genre_tracks:
            # No exact matches - suggest alternatives
            suggestions = self.library.suggest_similar_genres(normalized_genre, limit=10)
            logger.warning(f"No tracks found for genre '{normalized_genre}'")
            if suggestions:
                logger.info(f"Did you mean one of these genres? {', '.join(suggestions)}")
            return None

        if len(genre_tracks) < 4:
            logger.warning(f"Genre '{normalized_genre}' has only {len(genre_tracks)} tracks, need at least 4")
            return None

        logger.info(f"Found {len(genre_tracks)} tracks for genre '{normalized_genre}'")

        # Add play count (0 for all, since we don't use history for genre mode)
        for track in genre_tracks:
            track['play_count'] = 0

        # Select seed tracks (4 seeds for consistency with artist mode)
        # Use deterministic selection based on random_seed from config
        # Filter by duration before selecting (exclude short/long tracks)
        from src.playlist.filtering import is_valid_duration
        valid_tracks = [t for t in genre_tracks if is_valid_duration(t, min_seconds=47, max_seconds=720)]
        if len(valid_tracks) == 0:
            raise ValueError(f"Genre '{genre_name}' has no tracks in valid duration range (47s-720s)")
        if len(valid_tracks) < 10:
            raise ValueError(
                f"Genre '{genre_name}' has only {len(valid_tracks)} valid-duration tracks "
                f"(minimum 10 required for playlist generation). "
                f"Total tracks in genre: {len(genre_tracks)}"
            )
        if len(valid_tracks) < 4:
            logger.warning(f"Genre has only {len(valid_tracks)} valid-duration tracks (requested 4); using all valid tracks")

        import random
        ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
        random_seed = ds_cfg.get('random_seed', 0)
        rng = random.Random(random_seed)
        seed_count = min(4, len(valid_tracks))
        seed_tracks = rng.sample(valid_tracks, seed_count)

        seed_info = [f"{t.get('artist')} - {t.get('title')}" for t in seed_tracks]
        logger.info(f"Seeds ({len(seed_tracks)}): {', '.join(seed_info[:3])}{'...' if len(seed_info) > 3 else ''}")

        # Prepare recency data (scrobbles for filtering)
        scrobbles: List[Dict[str, Any]] = []
        if self.lastfm:
            try:
                scrobbles = self._get_lastfm_scrobbles_raw(use_cache=True)
            except Exception as exc:
                logger.warning("Last.FM scrobble fetch failed; skipping scrobble recency filter (%s)", exc, exc_info=True)

        # DS scope: use genre tracks as candidate pool (not entire library)
        # IMPORTANT: Use valid_tracks (duration-filtered) not genre_tracks (unfiltered)
        ds_candidates = valid_tracks
        excluded_ids = set()

        if scrobbles:
            excluded_ids = self._compute_excluded_from_scrobbles(
                ds_candidates,
                scrobbles,
                lookback_days=self.config.recently_played_lookback_days,
                seed_id=seed_tracks[0].get('rating_key') if seed_tracks else None,
            )
            logger.info(f"stage=candidate_pool | Last.fm recency exclusions: before={len(ds_candidates)} after={len(ds_candidates) - len(excluded_ids)} excluded={len(excluded_ids)}")

        allowed_track_ids = [t.get("rating_key") for t in ds_candidates if t.get("rating_key") and str(t.get("rating_key")) not in excluded_ids]

        logger.info(f"DS scope: genre (allowed_ids={len(allowed_track_ids)})")

        # Determine DS mode
        ds_mode = ds_mode_override or (self.ds_mode_override if hasattr(self, 'ds_mode_override') and self.ds_mode_override else None)
        if not ds_mode:
            ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
            ds_mode = "dynamic" if dynamic else ds_cfg.get('mode', 'dynamic')

        logger.info(f"Running pipeline with mode={ds_mode}")

        # Genre mode doesn't use artist_style clustering (no single artist to cluster)
        # But we still use pier-bridge with the genre seeds as piers
        style_seed_track_id = seed_tracks[0].get('rating_key') if seed_tracks else None
        anchor_seed_ids = [s.get('rating_key') for s in seed_tracks if s.get('rating_key')]

        # Call DS pipeline
        # Note: Pier-bridge may drop 1-2 tracks for cross-segment gap enforcement (quality control)
        # Accept results within tolerance rather than requiring exact match
        ds_tracks = None
        import re

        try:
            ds_tracks = self._maybe_generate_ds_playlist(
                seed_track_id=style_seed_track_id,
                target_length=track_count,
                mode_override=ds_mode,
                allowed_track_ids=allowed_track_ids,
                excluded_track_ids=excluded_ids,
                anchor_seed_tracks=seed_tracks,  # Pass full seed track info
                anchor_seed_ids=anchor_seed_ids,
                artist_style_enabled=False,  # No artist clustering for genre mode
                artist_playlist=False,  # Genre mode, not artist mode
                pool_source="library",
                dry_run=bool(dry_run),
                audit_context_extra={
                    "genre": normalized_genre,
                    "seed_mode": "genre",
                },
                internal_connector_ids=None,
                internal_connector_max_per_segment=0,
                internal_connector_priority=False,
            )
        except ValueError as e:
            error_msg = str(e)
            if "length_mismatch" in error_msg and "final=" in error_msg:
                # Pier-bridge drops tracks for gap enforcement - accept if close enough
                match = re.search(r'final=(\d+)', error_msg)
                if match:
                    actual_length = int(match.group(1))
                    tolerance = max(3, int(track_count * 0.1))  # Within 3 tracks or 10%
                    min_acceptable = max(15, int(track_count * 0.5))  # At least 50% or 15 tracks

                    if actual_length < min_acceptable:
                        logger.error(
                            f"Genre '{normalized_genre}' candidate pool too small: "
                            f"got {actual_length} tracks (minimum {min_acceptable}). "
                            f"Try --ds-mode dynamic or discover for larger pool."
                        )
                        raise

                    if actual_length >= track_count - tolerance:
                        # Close enough - retry with exact actual length to pass validation
                        logger.warning(
                            f"Genre mode: Pier-bridge produced {actual_length} tracks (target {track_count}), "
                            f"accepting result (within tolerance)"
                        )
                        ds_tracks = self._maybe_generate_ds_playlist(
                            seed_track_id=style_seed_track_id,
                            target_length=actual_length,  # Use actual achievable length
                            mode_override=ds_mode,
                            allowed_track_ids=allowed_track_ids,
                            excluded_track_ids=excluded_ids,
                            anchor_seed_tracks=seed_tracks,
                            anchor_seed_ids=anchor_seed_ids,
                            artist_style_enabled=False,
                            artist_playlist=False,
                            pool_source="library",
                            dry_run=bool(dry_run),
                            audit_context_extra={
                                "genre": normalized_genre,
                                "seed_mode": "genre",
                                "adjusted_target": True,
                            },
                            internal_connector_ids=None,
                            internal_connector_max_per_segment=0,
                            internal_connector_priority=False,
                        )
                    else:
                        logger.error(
                            f"Genre '{normalized_genre}': Pier-bridge produced {actual_length} tracks "
                            f"(target {track_count}, tolerance {tolerance}). "
                            f"Try --ds-mode dynamic or reduce --tracks to {actual_length}."
                        )
                        raise
                else:
                    raise  # Can't parse length, re-raise
            else:
                raise  # Not a length mismatch error, re-raise

        if not ds_tracks:
            logger.warning(f"DS pipeline returned no tracks for genre '{normalized_genre}'")
            return None

        final_tracks = ds_tracks

        logger.info(f"Generated {len(final_tracks)} tracks for genre '{normalized_genre}'")

        # Print summary
        title = f"Auto: {normalized_genre.title()}"
        self._print_playlist_report(final_tracks, artist_name=None, dynamic=dynamic, verbose_edges=verbose)

        return {
            'title': title,
            'artists': tuple(set(t.get('artist') for t in final_tracks if t.get('artist'))),
            'genres': [normalized_genre],
            'tracks': final_tracks,
            'ds_report': getattr(self, "_last_ds_report", None),
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

        # Pre-order recency exclusions (applied via DS `excluded_track_ids`).
        # IMPORTANT: No recency filtering is allowed after DS ordering.
        excluded_ids_batch: Set[str] = set()
        recency_candidates: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for t in history or []:
            tid = t.get("rating_key")
            if not tid:
                continue
            sid = str(tid)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            recency_candidates.append(t)

        lookback_days = getattr(self.config, "recently_played_lookback_days", 0)
        scrobbles: List[Dict[str, Any]] = []
        if self.lastfm:
            try:
                scrobbles = self._get_lastfm_scrobbles_raw(use_cache=True)
            except Exception as exc:
                logger.warning(
                    "Last.FM scrobble fetch failed; skipping scrobble recency exclusions (%s)",
                    exc,
                    exc_info=True,
                )
        if scrobbles:
            excluded_ids_batch = self._compute_excluded_from_scrobbles(
                recency_candidates,
                scrobbles,
                lookback_days=lookback_days,
                seed_id=None,
            )
        elif history and getattr(self.config, "recently_played_filter_enabled", False):
            excluded_ids_batch = self._compute_excluded_from_history(
                recency_candidates,
                history,
                lookback_days=lookback_days,
                exempt_tracks=None,
            )

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
            ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
            style_cfg_raw = ds_cfg.get("artist_style", {}) or {}
            style_cfg = ArtistStyleConfig(
                enabled=bool(style_cfg_raw.get("enabled", False)),
                cluster_k_min=style_cfg_raw.get("cluster_k_min", 3),
                cluster_k_max=style_cfg_raw.get("cluster_k_max", 6),
                cluster_k_heuristic_enabled=style_cfg_raw.get("cluster_k_heuristic_enabled", True),
                piers_per_cluster=style_cfg_raw.get("piers_per_cluster", 1),
                per_cluster_candidate_pool_size=style_cfg_raw.get("per_cluster_candidate_pool_size", 400),
                pool_balance_mode=style_cfg_raw.get("pool_balance_mode", "equal"),
                internal_connector_priority=style_cfg_raw.get("internal_connector_priority", True),
                internal_connector_max_per_segment=style_cfg_raw.get("internal_connector_max_per_segment", 2),
                bridge_floor_narrow=style_cfg_raw.get("bridge_floor", {}).get("narrow", 0.08),
                bridge_floor_dynamic=style_cfg_raw.get("bridge_floor", {}).get("dynamic", 0.03),
                bridge_weight=style_cfg_raw.get("bridge_score_weights", {}).get("bridge", 0.7),
                transition_weight=style_cfg_raw.get("bridge_score_weights", {}).get("transition", 0.3),
                genre_tiebreak_weight=style_cfg_raw.get("genre_tiebreak_weight", 0.05),
            )
            ds_mode_effective = ds_mode_override or ("dynamic" if dynamic else ds_cfg.get("mode", "dynamic"))
            style_seed_track_id = seeds[0].get('rating_key') if seeds else None
            style_anchor_tracks = seeds
            style_anchor_ids = None
            allowed_track_ids = None
            pier_cfg = None
            internal_connectors = None
            artifact_path = ds_cfg.get("artifact_path")
            pool_source = "legacy"
            using_artist_style = False
            logger.info(
                "Artist style mode %s: artist=%s ds_mode=%s",
                "ENABLED" if style_cfg.enabled else "DISABLED",
                artist,
                ds_mode_effective,
            )
            if style_cfg.enabled and artifact_path:
                try:
                    bundle = load_artifact_bundle(artifact_path)
                    sonic_cfg = self.config.get("playlists", "sonic", default={}) or {}
                    sonic_variant_cfg = resolve_sonic_variant(
                        explicit_variant=getattr(self, "sonic_variant", None),
                        config_variant=ds_cfg.get("sonic_variant") or sonic_cfg.get("sim_variant"),
                    )
                    clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
                        bundle=bundle,
                        artist_name=artist,
                        cfg=style_cfg,
                        random_seed=ds_cfg.get("random_seed", 0),
                        sonic_variant=sonic_variant_cfg,
                    )
                    if not medoids:
                        raise ValueError("Style clustering returned no medoids")
                    ordered_medoids = order_clusters(medoids, X_norm)
                    cluster_piers = medoids_by_cluster
                    min_sonic = get_min_sonic_similarity(ds_cfg.get("candidate_pool", {}), ds_mode_effective)
                    artist_key = normalize_artist_key(artist)
                    external_pool = build_balanced_candidate_pool(
                        bundle=bundle,
                        cluster_piers=cluster_piers,
                        X_norm=X_norm,
                        per_cluster_size=style_cfg.per_cluster_candidate_pool_size,
                        pool_balance_mode=style_cfg.pool_balance_mode,
                        global_floor=min_sonic,
                        artist_key=artist_key,
                    )
                    internal_connectors = get_internal_connectors(
                        bundle=bundle,
                        artist_key=artist_key,
                        exclude_indices=medoids,
                        global_floor=min_sonic,
                        pier_indices=medoids,
                        X_norm=X_norm,
                    ) if style_cfg.internal_connector_priority else []
                    pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]
                    allowed_track_ids = list(dict.fromkeys(pier_ids + external_pool + internal_connectors))
                    if not allowed_track_ids:
                        raise ValueError("Artist style allowed pool empty")
                    style_seed_track_id = pier_ids[0]
                    style_anchor_ids = pier_ids[1:]
                    style_anchor_tracks = [
                        {
                            "rating_key": str(bundle.track_ids[m]),
                            "title": str(bundle.track_titles[m] or ""),
                            "artist": str(bundle.track_artists[m] or artist),
                        }
                        for m in ordered_medoids
                    ]
                    center_transitions = bool(ds_cfg.get("center_transitions", False)) or bool(
                        ds_cfg.get("constraints", {}).get("center_transitions", False)
                    )
                    tw_raw = ds_cfg.get("transition_weights")
                    transition_weights = None
                    if isinstance(tw_raw, dict):
                        transition_weights = (
                            float(tw_raw.get("rhythm", 0.4)),
                            float(tw_raw.get("timbre", 0.35)),
                            float(tw_raw.get("harmony", 0.25)),
                        )
                    elif isinstance(tw_raw, (list, tuple)) and len(tw_raw) == 3:
                        transition_weights = (
                            float(tw_raw[0]),
                            float(tw_raw[1]),
                            float(tw_raw[2]),
                        )
                    ds_defaults = default_ds_config(ds_mode_effective, playlist_len=track_count, overrides=ds_cfg)
                    pb_tuning = resolve_pier_bridge_tuning(ds_cfg, ds_mode_effective)

                    weight_bridge = float(pb_tuning["weight_bridge"])
                    weight_transition = float(pb_tuning["weight_transition"])
                    weights_raw = style_cfg_raw.get("bridge_score_weights")
                    if isinstance(weights_raw, dict):
                        by_mode = weights_raw.get(ds_mode_effective)
                        if isinstance(by_mode, dict):
                            weight_bridge = float(by_mode.get("bridge", weight_bridge))
                            weight_transition = float(by_mode.get("transition", weight_transition))
                        else:
                            weight_bridge = float(weights_raw.get("bridge", weight_bridge))
                            weight_transition = float(weights_raw.get("transition", weight_transition))

                    genre_tiebreak_weight = float(
                        style_cfg_raw.get("genre_tiebreak_weight", pb_tuning["genre_tiebreak_weight"])
                    )
                    bridge_floor = float(pb_tuning["bridge_floor"])
                    bridge_floor_raw = style_cfg_raw.get("bridge_floor")
                    if isinstance(bridge_floor_raw, dict):
                        mode_val = bridge_floor_raw.get(ds_mode_effective)
                        if isinstance(mode_val, (int, float)):
                            bridge_floor = float(mode_val)
                    elif isinstance(bridge_floor_raw, (int, float)):
                        bridge_floor = float(bridge_floor_raw)

                    pier_cfg = PierBridgeConfig(
                        transition_floor=float(ds_defaults.construct.transition_floor),
                        bridge_floor=bridge_floor,
                        center_transitions=bool(ds_defaults.construct.center_transitions),
                        transition_weights=transition_weights,
                        sonic_variant=sonic_variant_cfg,
                        weight_bridge=weight_bridge,
                        weight_transition=weight_transition,
                        genre_tiebreak_weight=genre_tiebreak_weight,
                        genre_penalty_threshold=float(pb_tuning["genre_penalty_threshold"]),
                        genre_penalty_strength=float(pb_tuning["genre_penalty_strength"]),
                    )
                    using_artist_style = True
                    pool_source = "artist_style"
                    logger.info(
                        "Artist style mode ENABLED: artist=%s ds_mode=%s clusters=%d piers=%d allowed_ids=%d internal_connectors=%d",
                        artist,
                        ds_mode_effective,
                        len(clusters),
                        len(ordered_medoids),
                        len(allowed_track_ids),
                        len(internal_connectors),
                    )
                except Exception as exc:
                    logger.warning("Artist style clustering failed (%s); using default artist flow", exc, exc_info=True)
                    allowed_track_ids = None
                    pier_cfg = None
                    internal_connectors = None
                    using_artist_style = False
                    pool_source = "legacy_fallback"

            ds_tracks = self._maybe_generate_ds_playlist(
                seed_track_id=style_seed_track_id,
                target_length=target_playlist_size,
                pipeline_override=pipeline_override,
                mode_override=ds_mode_effective,
                seed_artist=artist,
                anchor_seed_tracks=style_anchor_tracks,  # Pass full seed track info for title+artist resolution
                allowed_track_ids=allowed_track_ids,
                excluded_track_ids=excluded_ids_batch or None,
                anchor_seed_ids=style_anchor_ids,
                pier_bridge_config=pier_cfg,
                internal_connector_ids=internal_connectors if style_cfg.internal_connector_priority else None,
                internal_connector_max_per_segment=style_cfg.internal_connector_max_per_segment,
                internal_connector_priority=style_cfg.internal_connector_priority,
                artist_style_enabled=using_artist_style,
                artist_playlist=True,
                pool_source=pool_source,
            )

            last_report = getattr(self, "_last_ds_report", None) or {}
            is_pier_bridge = (last_report.get("metrics") or {}).get("strategy") == "pier_bridge"

            audit_path = None
            try:
                audit_path = (
                    (getattr(self, "_last_ds_report", None) or {}).get("playlist_stats") or {}
                ).get("playlist", {}).get("audit_path")
            except Exception:
                audit_path = None

            pier_tracks_for_exemption = style_anchor_tracks if using_artist_style else seeds
            exempt_pier_ids = {
                str(t.get("rating_key"))
                for t in (pier_tracks_for_exemption or [])
                if t.get("rating_key")
            }
            self._post_order_validate_ds_output(
                ordered_tracks=ds_tracks,
                expected_length=target_playlist_size,
                excluded_track_ids=set(excluded_ids_batch or set()),
                exempt_pier_track_ids=exempt_pier_ids,
                audit_path=str(audit_path) if audit_path else None,
            )

            if is_pier_bridge:
                final_tracks = ds_tracks
            else:
                final_tracks = self._ensure_seed_tracks_present(seeds, ds_tracks, target_playlist_size)

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

    def create_playlist_from_seed_tracks(
        self,
        seed_tracks: List[str],
        *,
        track_count: int = 30,
        dynamic: bool = False,
        ds_mode_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a playlist from explicit seed tracks without requiring an artist name.
        Seed tracks should be provided in "Title - Artist" format (autocomplete default).
        """
        if not seed_tracks:
            raise ValueError("No seed tracks provided.")

        all_library_tracks = self.library.get_all_tracks()

        def _parse_seed_entry(entry: str) -> tuple[str, str]:
            title = entry.strip()
            artist = ""
            if " - " in title:
                title_part, artist_part = title.split(" - ", 1)
                title = title_part.strip()
                artist = artist_part.strip()
                if " (" in artist:
                    artist = artist.split(" (", 1)[0].strip()
            return title, artist

        resolved_seeds: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for entry in seed_tracks:
            title, artist = _parse_seed_entry(str(entry))
            if not title:
                continue
            if not artist:
                logger.warning(
                    "Seed entry '%s' missing artist; skipping.",
                    entry,
                )
                continue
            artist_key = normalize_artist_key(artist)
            artist_tracks = [
                t for t in all_library_tracks if safe_get_artist_key(t) == artist_key
            ]
            if not artist_tracks:
                logger.warning(
                    "No tracks found for seed artist '%s' (entry '%s')",
                    artist,
                    entry,
                )
                continue
            selected = self._select_canonical_track(artist_tracks, title)
            if not selected:
                logger.warning(
                    "Seed track '%s' not found for artist '%s'",
                    title,
                    artist,
                )
                continue
            key = str(selected.get("rating_key") or selected.get("track_id") or "")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            selected["play_count"] = 0
            resolved_seeds.append(selected)

        if not resolved_seeds:
            raise ValueError("No valid seed tracks resolved from selection.")

        seed_titles = [t.get("title") for t in resolved_seeds]
        logger.info("Seeds (%d): %s", len(resolved_seeds), ", ".join(seed_titles))

        scrobbles: List[Dict[str, Any]] = []
        if self.lastfm:
            try:
                scrobbles = self._get_lastfm_scrobbles_raw(use_cache=True)
            except Exception as exc:
                logger.warning(
                    "Last.FM scrobble fetch failed; skipping scrobble recency filter (%s)",
                    exc,
                    exc_info=True,
                )

        excluded_ids = set()
        if scrobbles:
            excluded_ids = self._compute_excluded_from_scrobbles(
                all_library_tracks,
                scrobbles,
                lookback_days=self.config.recently_played_lookback_days,
                seed_id=resolved_seeds[0].get("rating_key"),
            )
            if excluded_ids and len(all_library_tracks) - len(excluded_ids) <= 1:
                logger.warning(
                    "Recency exclusions would empty DS library candidate set; skipping recency exclusion."
                )
                excluded_ids = set()

        ds_tracks = self._maybe_generate_ds_playlist(
            seed_track_id=resolved_seeds[0].get("rating_key"),
            target_length=track_count,
            mode_override=ds_mode_override,
            anchor_seed_tracks=resolved_seeds,
            anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
            excluded_track_ids=excluded_ids or None,
            artist_playlist=False,
            pool_source="seeded",
        )

        if ds_tracks is None:
            raise ValueError("DS pipeline returned no tracks.")

        last_report = getattr(self, "_last_ds_report", None) or {}
        is_pier_bridge = (last_report.get("metrics") or {}).get("strategy") == "pier_bridge"

        audit_path = None
        try:
            audit_path = (last_report.get("playlist_stats") or {}).get("playlist", {}).get("audit_path")
        except Exception:
            audit_path = None

        exempt_pier_ids = {
            str(t.get("rating_key")) for t in resolved_seeds if t.get("rating_key")
        }
        self._post_order_validate_ds_output(
            ordered_tracks=ds_tracks,
            expected_length=track_count,
            excluded_track_ids=set(excluded_ids or set()),
            exempt_pier_track_ids=exempt_pier_ids,
            audit_path=str(audit_path) if audit_path else None,
        )

        if is_pier_bridge:
            final_tracks = ds_tracks
        else:
            final_tracks = self._ensure_seed_tracks_present(
                resolved_seeds,
                ds_tracks,
                track_count,
            )

        title = "Auto: Seeded"
        self._print_playlist_report(final_tracks, artist_name="Seeded", dynamic=dynamic)
        return {
            "title": title,
            "artists": tuple(sorted({t.get("artist") for t in resolved_seeds if t.get("artist")})),
            "genres": [],
            "tracks": final_tracks,
            "ds_report": getattr(self, "_last_ds_report", None),
        }

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
    logger.info("Playlist Generator module loaded")
