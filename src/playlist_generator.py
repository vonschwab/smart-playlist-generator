"""
Playlist Generator - Core logic for creating Data Science-powered playlists
"""
from typing import List, Dict, Any, Optional, Set, Sequence
from collections import Counter
from dataclasses import replace
import random
import logging
import os
import math
from .string_utils import normalize_artist_key
from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_variant import resolve_sonic_variant
from src.playlist.ds_pipeline_runner import DsRunResult, generate_playlist_ds as run_ds_pipeline
from src.playlist.artist_style import (
    ArtistStyleConfig,
    build_balanced_candidate_pool,
    build_genre_neighbor_candidate_pool,
    cluster_artist_tracks,
    order_clusters,
    select_popular_piers,
    _select_k,
    _artist_indices_in_bundle,
)
from src.playlist.pier_bridge_builder import PierBridgeConfig, resolve_pier_bridge_tuning
from src.playlist.pier_bridge.config import roam_kwargs_from_dict
from src.playlist.config import default_ds_config, get_min_sonic_similarity, resolve_cohesion_mode
from src.playlist.genre_ds_params import resolve_genre_ds_params
# Phase 2: Import utilities from refactored module
from src.playlist import utils
# Phase 3: Import filtering from refactored module
from src.playlist import filtering

# Phase 4: Import history_analyzer from refactored module
from src.playlist import history_analyzer

# Phase 9: Import reporter from refactored module
from src.playlist import reporter

# Title artifact detection for audit logging
from src.playlist.title_quality import detect_title_artifacts

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
        "genre_graph": ds_cfg.get("genre_graph", {}),
        "genre_source": ds_cfg.get("genre_source", "legacy"),
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


def _build_edge_audit_rows(
    edge_scores_list: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    transition_floor: float = 0.20,
    beam_components: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build per-edge audit row dicts for emit_selected_edge_audit.

    Zips the enriched edge_scores_list with the track list to populate
    from_artist, from_title, to_artist, to_title, and below_transition_floor.
    When beam_components is provided, merges per-edge component dicts by
    position so beam-internal fields (bridge_score, trans_score_in_beam,
    progress_t, progress_jump, local_sonic_raw_cos, penalties) are populated
    instead of rendering as 'n/a' in the audit table.
    """
    if not edge_scores_list or not tracks:
        return []
    rows: List[Dict[str, Any]] = []
    for i, edge in enumerate(edge_scores_list):
        if not isinstance(edge, dict):
            continue
        # Tracks[i] is "from" and tracks[i+1] is "to" (edges[0] = tracks[0]->tracks[1])
        from_track = tracks[i] if i < len(tracks) else {}
        to_track = tracks[i + 1] if (i + 1) < len(tracks) else {}
        t_val = edge.get("T")
        below_floor = (
            isinstance(t_val, (int, float)) and t_val == t_val and t_val < float(transition_floor)
        )
        # Merge beam-captured component dict for this edge position (if available)
        beam_comp: Dict[str, Any] = {}
        if beam_components is not None and i < len(beam_components):
            bc = beam_components[i]
            if isinstance(bc, dict):
                beam_comp = bc
        to_title = str(to_track.get("title") or "?")
        row = {
            "from_idx": edge.get("prev_idx"),
            "to_idx": edge.get("cur_idx"),
            "from_artist": str(from_track.get("artist") or "?"),
            "from_title": str(from_track.get("title") or "?"),
            "to_artist": str(to_track.get("artist") or "?"),
            "to_title": to_title,
            "T": t_val,
            "T_centered_cos": edge.get("T_centered_cos"),
            "S": edge.get("S"),
            "G": edge.get("G"),
            # Beam-captured fields: prefer beam_comp values, fall back to edge dict
            "bridge_score": beam_comp.get("bridge_score") if beam_comp else edge.get("bridge_score"),
            "trans_score_in_beam": beam_comp.get("trans_score_in_beam") if beam_comp else edge.get("trans_score_in_beam"),
            "progress_t": beam_comp.get("progress_t") if beam_comp else edge.get("progress_t"),
            "progress_jump": beam_comp.get("progress_jump") if beam_comp else edge.get("progress_jump"),
            "local_sonic_raw_cos": beam_comp.get("local_sonic_raw_cos") if beam_comp else edge.get("local_sonic_raw_cos"),
            "local_sonic_penalty_applied": beam_comp.get("local_sonic_penalty_applied") if beam_comp else edge.get("local_sonic_penalty_applied"),
            "genre_penalty_applied": beam_comp.get("genre_penalty_applied") if beam_comp else edge.get("genre_penalty_applied"),
            "below_transition_floor": below_floor,
            "to_title_flags": detect_title_artifacts(to_title),
            "bpm_a": edge.get("bpm_a"),
            "bpm_b": edge.get("bpm_b"),
            "bpm_log_dist": edge.get("bpm_log_dist"),
        }
        rows.append(row)
    return rows


def _resolve_popularity_rank_cutoff(popularity_mode: str, bangers_cfg: dict) -> Optional[int]:
    """Oops, All Bangers admission-gate cutoff. off -> None (gate disabled);
    on -> rank_cutoff_on (default 50); oops -> rank_cutoff_oops (default 10)."""
    m = str(popularity_mode or "off").lower()
    if m == "on":
        return int((bangers_cfg or {}).get("rank_cutoff_on", 50))
    if m == "oops":
        return int((bangers_cfg or {}).get("rank_cutoff_oops", 10))
    return None


def _resolve_popular_seeds_mode(popular_seeds_mode: str, popularity_mode: str) -> str:
    """Popular-seed pier mode: off / on / fire. OOPS (the all-bangers bridge gate) forces
    'fire' so the piers are unambiguous hits too. Artist-mode-only by construction (this is
    called on the artist-mode entry point; seed mode never reaches here)."""
    if str(popularity_mode or "off").lower() == "oops":
        return "fire"
    m = str(popular_seeds_mode or "off").lower()
    return m if m in {"off", "on", "fire"} else "off"


class PlaylistGenerator:
    """Generates playlists based on listening history and similarity"""

    def __init__(
        self,
        library_client=None,
        config=None,
        lastfm_client=None,
        track_matcher=None,
        metadata_client=None,
        config_path: Optional[str] = None,
    ):
        if config_path is not None:
            from src.config_loader import Config
            from src.local_library_client import LocalLibraryClient

            config = config or Config(config_path)
            library_client = library_client or LocalLibraryClient(
                db_path=config.get("library", "database_path", default="data/metadata.db")
            )
        if library_client is None or config is None:
            raise TypeError("PlaylistGenerator requires library_client and config, or config_path")

        self.library = library_client
        self.config = config
        self.lastfm = lastfm_client
        self.matcher = track_matcher
        self.metadata = metadata_client  # Metadata database for genre lookups
        self.pipeline_override: Optional[str] = None
        self._logged_ds_artifact_warning = False
        self._last_ds_report: Optional[Dict[str, Any]] = None
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

    def _title_exclusion_settings(self) -> tuple[bool, tuple[str, ...]]:
        ds_cfg = self.config.get("playlists", "ds_pipeline", default={}) or {}
        candidate_cfg = ds_cfg.get("candidate_pool", {}) if isinstance(ds_cfg, dict) else {}
        enabled = bool(candidate_cfg.get("title_exclusion_enabled", False))
        raw_words = candidate_cfg.get("title_exclusion_words", ()) or ()
        if isinstance(raw_words, str):
            words = (raw_words,)
        else:
            words = tuple(str(word) for word in raw_words)
        return enabled, words

    def _filter_title_excluded_tracks(
        self,
        tracks: List[Dict[str, Any]],
        *,
        context: str,
    ) -> List[Dict[str, Any]]:
        enabled, words = self._title_exclusion_settings()
        if not enabled or not words:
            return list(tracks)
        filtered = filtering.filter_by_title_exclusions(
            tracks=list(tracks),
            exclusion_words=words,
        )
        removed = len(tracks) - len(filtered)
        if removed:
            logger.info(
                "Title exclusions: context=%s before=%d after=%d excluded=%d",
                context,
                len(tracks),
                len(filtered),
                removed,
            )
        return filtered

    def _filter_title_excluded_bundle_indices(
        self,
        bundle: Any,
        indices: List[int],
        *,
        context: str,
    ) -> List[int]:
        enabled, words = self._title_exclusion_settings()
        if not enabled or not words or getattr(bundle, "track_titles", None) is None:
            return list(indices)
        filtered = [
            int(idx) for idx in indices
            if not filtering.is_title_excluded(bundle.track_titles[int(idx)], words)
        ]
        removed = len(indices) - len(filtered)
        if removed:
            logger.info(
                "Title exclusions: context=%s before=%d after=%d excluded=%d",
                context,
                len(indices),
                len(filtered),
                removed,
            )
        return filtered

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
        transition_weights: Optional[tuple[float, float, float]] = None,
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
            transition_weights=transition_weights,
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
        pace_mode: Optional[str] = None,
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
                    seed_album = str(seed_track.get('album', '')).strip()
                    # Check both track_id and rating_key for MD5 hash
                    seed_track_id_hash = str(seed_track.get('track_id') or seed_track.get('rating_key') or '')

                    if not seed_title or not seed_artist_name:
                        continue

                    # Find ALL matching tracks in bundle (may be multiple versions)
                    candidate_indices = []
                    for idx in range(len(bundle.track_artists)):
                        bundle_artist = normalize_artist_key(str(bundle.track_artists[idx]))
                        bundle_title = str(bundle.track_titles[idx]).strip().lower()
                        if bundle_artist == seed_artist_name and bundle_title == seed_title:
                            candidate_indices.append(idx)

                    if not candidate_indices:
                        continue

                    # If multiple matches, try to match by track_id (MD5) first
                    matched_idx = None
                    if len(candidate_indices) > 1 and seed_track_id_hash:
                        for idx in candidate_indices:
                            if str(bundle.track_ids[idx]) == seed_track_id_hash:
                                matched_idx = idx
                                logger.debug(
                                    "Matched seed '%s - %s' by exact track_id",
                                    seed_artist_name, seed_title
                                )
                                break

                    # If no exact track_id match and we have multiple candidates, use album-based disambiguation
                    if matched_idx is None and len(candidate_indices) > 1:
                        # Get album info for each candidate from library
                        candidate_albums = []
                        for idx in candidate_indices:
                            candidate_track_id = str(bundle.track_ids[idx])
                            candidate_track = self.library.get_track_by_key(candidate_track_id)
                            candidate_album = str(candidate_track.get('album', '')) if candidate_track else ''
                            candidate_albums.append((idx, candidate_album))

                        if seed_album:
                            # Check if seed album has version keywords
                            album_lower = seed_album.lower()
                            version_keywords = ['instrumental', 'live', 'acoustic', 'remix', 'demo', 'remaster']
                            seed_has_version = any(kw in album_lower for kw in version_keywords)

                            if seed_has_version:
                                # Seed wants a specific version - prefer exact album match or matching version keyword
                                for idx, candidate_album in candidate_albums:
                                    if candidate_album.lower() == album_lower:
                                        matched_idx = idx
                                        logger.debug(
                                            "Matched seed '%s - %s' (album=%s) by exact album match",
                                            seed_artist_name, seed_title, seed_album
                                        )
                                        break
                                # If no exact match, look for matching version keyword
                                if matched_idx is None:
                                    for kw in version_keywords:
                                        if kw in album_lower:
                                            for idx, candidate_album in candidate_albums:
                                                if kw in candidate_album.lower():
                                                    matched_idx = idx
                                                    logger.debug(
                                                        "Matched seed '%s - %s' (album=%s) by version keyword '%s'",
                                                        seed_artist_name, seed_title, seed_album, kw
                                                    )
                                                    break
                                            if matched_idx is not None:
                                                break
                            else:
                                # Seed wants standard/album version - avoid tracks with version keywords in album
                                # Prefer candidates WITHOUT version keywords in their album names
                                best_idx = None
                                for idx, candidate_album in candidate_albums:
                                    candidate_album_lower = candidate_album.lower()
                                    has_version_in_album = any(kw in candidate_album_lower for kw in version_keywords)
                                    if not has_version_in_album:
                                        best_idx = idx
                                        break
                                if best_idx is not None:
                                    matched_idx = best_idx
                                    logger.debug(
                                        "Matched seed '%s - %s' (album=%s) by preferring non-version album",
                                        seed_artist_name, seed_title, seed_album
                                    )

                    # Use first match if only one candidate or no disambiguation worked
                    if matched_idx is None:
                        matched_idx = candidate_indices[0]

                    anchor_seed_ids_resolved.append(str(bundle.track_ids[matched_idx]))

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
                    logger.warning(
                        "Seed track %s is blacklisted; searching for alternative seed from same artist",
                        seed_to_use
                    )
                    # Try to find a non-blacklisted track by the same artist
                    alternative_seed = None
                    if seed_artist and bundle.track_artists is not None:
                        artist_norm = normalize_artist_key(seed_artist)
                        for idx, artist in enumerate(bundle.track_artists):
                            candidate_id = str(bundle.track_ids[idx])
                            if (normalize_artist_key(str(artist)) == artist_norm and
                                candidate_id not in blacklist_ids):
                                alternative_seed = candidate_id
                                logger.info(
                                    "Found alternative non-blacklisted seed: %s (artist: %s)",
                                    alternative_seed,
                                    seed_artist
                                )
                                break

                    if alternative_seed:
                        seed_to_use = alternative_seed
                    else:
                        raise ValueError(
                            f"Seed track {seed_to_use} is blacklisted and no non-blacklisted alternative found for artist '{seed_artist or 'unknown'}'"
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
        pace_mode_effective = pace_mode or playlists_cfg.get("pace_mode") or "dynamic"
        # Genre gate + hybrid weights — resolved via the shared helper so the
        # gui_fidelity harness (and any other caller) resolves them identically
        # (these are explicit generate_playlist_ds params, NOT carried in overrides).
        _genre_params = resolve_genre_ds_params(playlists_cfg, mode)
        sonic_weight = _genre_params["sonic_weight"]
        genre_weight = _genre_params["genre_weight"]
        min_genre_sim = _genre_params["min_genre_similarity"]
        genre_method = _genre_params["genre_method"]
        genre_enabled = min_genre_sim is not None

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
            pace_mode=str(pace_mode_effective),
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
            "transition_weights": playlist_stats_only.get("transition_weights"),
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
                "Candidate pool EXCLUSIONS: total_candidates=%d below_floor=%d below_genre=%d below_bpm_floor=%d artist_cap_excluded=%d eligible=%d",
                pool_stats.get("total_candidates_considered", 0),
                pool_stats.get("below_similarity_floor", 0),
                pool_stats.get("below_genre_similarity", 0),
                pool_stats.get("below_bpm_floor", 0),
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
        # Store the emit_selected_edge_audit flag so the caller can invoke the audit after edge score recompute
        _emit_audit_flag = bool(pb_overrides.get("emit_selected_edge_audit", False))
        if pier_bridge_config is not None:
            _emit_audit_flag = bool(getattr(pier_bridge_config, "emit_selected_edge_audit", _emit_audit_flag))
        self._last_ds_report["emit_selected_edge_audit"] = _emit_audit_flag
        return tracks


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
        tracks = self._filter_title_excluded_tracks(
            tracks,
            context="candidate_filter",
        )

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

        title_exclusion_enabled, title_exclusion_words = self._title_exclusion_settings()
        if title_exclusion_enabled and title_exclusion_words:
            title_offenders = [
                track for track in ordered_tracks
                if filtering.is_title_excluded(track.get("title", ""), title_exclusion_words)
            ]
            if title_offenders:
                labels = [
                    f"{utils.sanitize_for_logging(str(track.get('artist', '')))} - "
                    f"{utils.sanitize_for_logging(str(track.get('title', '')))}"
                    for track in title_offenders[:10]
                ]
                errors.append(
                    f"title_exclusion_overlap={len(title_offenders)} offenders={labels}"
                )

        if errors:
            msg = "post_order_validation_failed: " + " | ".join(errors)
            if audit_path:
                msg += f" (audit: {audit_path})"
            raise ValueError(msg)

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
        tracks = self._filter_title_excluded_tracks(
            tracks,
            context="history_seed_selection",
        )

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
        cohesion_mode_override: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Create multiple playlists with single seed artist per playlist

        Args:
            count: Number of playlists to create
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info("="*70)
        logger.info(f"Creating {count} playlists (1 seed artist per playlist)")
        if dynamic:
            logger.info("Dynamic mode enabled: 60% sonic similarity + 40% genre-based discovery")
        logger.info("="*70)

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
        logger.info("Each artist will provide 4 seed tracks\n")

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
            cohesion_mode_override=cohesion_mode_override,
        )

        return playlists

    def create_playlist_for_artist(
        self,
        artist_name: Optional[str] = None,
        track_count: int = 30,
        track_title: Optional[str] = None,
        track_titles: Optional[List[str]] = None,
        dynamic: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        cohesion_mode_override: Optional[str] = None,
        artist_only: bool = False,
        anchor_seed_ids: Optional[List[str]] = None,
        seed_epoch: int = 0,
        include_collaborations: bool = False,
        exclude_seed_tracks_from_recency: bool = False,
        artist: Optional[str] = None,
        num_tracks: Optional[int] = None,
        mode: Optional[str] = None,
        random_seed: Optional[int] = None,
        popular_seeds_mode: str = "off",
        popularity_mode: str = "off",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single playlist for a specific artist without requiring listening history

        Args:
            artist_name: Name of the artist to create playlist for
            track_count: Target number of tracks in playlist

        Returns:
            Playlist dictionary with tracks and metadata, or None if unable to create
        """
        if artist_name is None:
            artist_name = artist
        if artist_name is None:
            raise ValueError("artist_name is required")
        if num_tracks is not None:
            track_count = num_tracks
        if mode is not None:
            cohesion_mode_override = mode
            dynamic = mode == "dynamic"
        if random_seed is not None:
            self.config.config.setdefault("playlists", {}).setdefault("ds_pipeline", {})["random_seed"] = random_seed

        # Oops, All Bangers: OOPS mode forces popular-seed pier selection so piers
        # are the artist's hits, not just the centroid medoids.
        popular_seeds_mode = _resolve_popular_seeds_mode(popular_seeds_mode, popularity_mode)

        logger.info(f"Creating playlist for artist: {artist_name}")

        # Get all tracks by this artist from local library
        all_library_tracks = self.library.get_all_tracks()

        # Filter to just this artist (normalized key match)
        artist_key = normalize_artist_key(artist_name)
        artist_tracks = [
            t for t in all_library_tracks
            if safe_get_artist_key(t) == artist_key
        ]

        # Mix in collaborations when explicitly requested, or as a fallback when
        # the artist has too few solo tracks to seed from.
        solo_count = len(artist_tracks)
        seek_collaborations = include_collaborations or solo_count < 4
        if seek_collaborations:
            if include_collaborations:
                logger.info(
                    "Include-collaborations enabled (%d solo tracks); searching for collaborations...",
                    solo_count,
                )
            else:
                logger.info(
                    "Artist has only %d exact match tracks, searching for collaborations...",
                    solo_count,
                )

            solo_ids = {t.get('rating_key') for t in artist_tracks if t.get('rating_key')}
            collaboration_tracks = [
                t for t in all_library_tracks
                if t.get('rating_key') not in solo_ids
                and self._is_collaboration_of(t.get('artist', ''), artist_name)
            ]

            if collaboration_tracks:
                collab_artists: Dict[str, int] = {}
                for track in collaboration_tracks:
                    collab_artist = track.get('artist', '')
                    collab_artists[collab_artist] = collab_artists.get(collab_artist, 0) + 1
                for collab_artist, count in collab_artists.items():
                    logger.info(f"  Found collaboration: {collab_artist} ({count} tracks)")

                artist_tracks.extend(collaboration_tracks)
                logger.info(
                    "Found %d total tracks (%d solo, %d collaborations)",
                    len(artist_tracks), solo_count, len(collaboration_tracks),
                )
            elif solo_count < 4:
                logger.warning(
                    "Artist has only %d tracks and no collaborations found, need at least 4",
                    solo_count,
                )
                return None
            elif include_collaborations:
                logger.info("No collaboration tracks found for %s; using solo tracks only", artist_name)

        if len(artist_tracks) < 4:
            logger.warning(f"Artist has only {len(artist_tracks)} total tracks (including collaborations), need at least 4")
            return None

        logger.info(f"Using {len(artist_tracks)} tracks for {artist_name}")

        # Add play count (0 for all, since we don't have history)
        for track in artist_tracks:
            track['play_count'] = 0

        ds_cfg = self.config.get('playlists', 'ds_pipeline', default={}) or {}
        scrobbles: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        seed_recency_excluded_ids: Set[str] = set()
        if self.lastfm:
            try:
                scrobbles = self._get_lastfm_scrobbles_raw(use_cache=True)
            except Exception as exc:
                logger.warning("Last.FM scrobble fetch failed; skipping scrobble recency filter (%s)", exc, exc_info=True)
        elif exclude_seed_tracks_from_recency and getattr(self.config, "recently_played_filter_enabled", False):
            try:
                history = self._get_local_history()
            except Exception as exc:
                logger.warning("Local history fetch failed; skipping seed freshness filter (%s)", exc, exc_info=True)

        if (
            exclude_seed_tracks_from_recency
            and getattr(self.config, "recently_played_filter_enabled", False)
        ):
            if scrobbles:
                seed_recency_excluded_ids = self._compute_excluded_from_scrobbles(
                    all_library_tracks,
                    scrobbles,
                    lookback_days=self.config.recently_played_lookback_days,
                    seed_id=None,
                )
            elif history:
                seed_recency_excluded_ids = self._compute_excluded_from_history(
                    all_library_tracks,
                    history,
                    lookback_days=self.config.recently_played_lookback_days,
                )
            logger.info(
                "Seed freshness filter: %s (%d excluded tracks)",
                "enabled" if seed_recency_excluded_ids else "enabled; no recent seed candidates found",
                len(seed_recency_excluded_ids),
            )

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

        if seed_tracks:
            before = len(seed_tracks)
            seed_tracks = self._filter_title_excluded_tracks(
                seed_tracks,
                context="artist_seed_selection",
            )
            if seed_recency_excluded_ids:
                before_freshness = len(seed_tracks)
                seed_tracks = [
                    t for t in seed_tracks
                    if str(t.get("rating_key")) not in seed_recency_excluded_ids
                ]
                removed = before_freshness - len(seed_tracks)
                if removed:
                    logger.info(
                        "Seed freshness filter: removed %d selected seed tracks for %s",
                        removed,
                        artist_name,
                    )
            if before and not seed_tracks:
                logger.warning(
                    "All selected seed tracks for %s were removed by title/freshness exclusions; falling back to automatic seeds.",
                    artist_name,
                )

        if not seed_tracks:
            # Get 4 seed tracks (2 most played + 2 random from top 10)
            # Since we don't have play counts, just pick 4 random tracks
            # Filter by duration before selecting (exclude short/long tracks)
            from src.playlist.filtering import is_valid_duration
            valid_tracks = [t for t in artist_tracks if is_valid_duration(t, min_seconds=47, max_seconds=720)]
            valid_tracks = self._filter_title_excluded_tracks(
                valid_tracks,
                context="artist_seed_selection",
            )
            if seed_recency_excluded_ids:
                before = len(valid_tracks)
                valid_tracks = [
                    t for t in valid_tracks
                    if str(t.get("rating_key")) not in seed_recency_excluded_ids
                ]
                removed = before - len(valid_tracks)
                if removed:
                    logger.info(
                        "Seed freshness filter: removed %d automatic seed candidates for %s",
                        removed,
                        artist_name,
                    )
            if len(valid_tracks) == 0:
                raise ValueError(
                    f"Artist '{artist_name}' has no tracks after duration/title/freshness exclusions "
                    "(duration=47s-720s)"
                )
            if len(valid_tracks) < 4:
                logger.warning(f"Artist has only {len(valid_tracks)} valid-duration tracks (requested 4); using all valid tracks")
            import random
            base_seed = int(ds_cfg.get("random_seed", 0) or 0)
            seed_epoch_val = int(seed_epoch or 0)
            rng = random.Random(base_seed + seed_epoch_val)
            seed_tracks = rng.sample(valid_tracks, min(4, len(valid_tracks)))

        anchor_seed_ids_override = None
        if fixed_seed_tracks:
            anchor_seed_ids_override = [
                str(t.get("rating_key")) for t in seed_tracks if t.get("rating_key")
            ]
        seed_titles = [track.get('title') for track in seed_tracks]
        logger.info(f"Seeds ({len(seed_tracks)}): {', '.join(seed_titles)}")

        # Genre tags are sourced from database/file tags only (Last.FM disabled for genres)

        # DS scope selection
        allowed_track_ids: Optional[List[str]] = None
        if artist_only:
            # IMPORTANT: Filter artist_tracks by duration for DS candidate pool
            # This ensures no tracks outside valid duration range (47s-720s) are selected
            from src.playlist.filtering import is_valid_duration
            ds_candidates = [t for t in artist_tracks if is_valid_duration(t, min_seconds=47, max_seconds=720)]
            ds_candidates = self._filter_title_excluded_tracks(
                ds_candidates,
                context="artist_only_candidate_scope",
            )
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
            medoid_top_k=style_cfg_raw.get("medoid_top_k", 5),
            bridge_floor_strict=style_cfg_raw.get("bridge_floor", {}).get("strict", 0.10),
            bridge_floor_narrow=style_cfg_raw.get("bridge_floor", {}).get("narrow", 0.05),
            bridge_floor_dynamic=style_cfg_raw.get("bridge_floor", {}).get("dynamic", 0.02),
            bridge_weight=style_cfg_raw.get("bridge_score_weights", {}).get("bridge", 0.7),
            transition_weight=style_cfg_raw.get("bridge_score_weights", {}).get("transition", 0.3),
            genre_tiebreak_weight=style_cfg_raw.get("genre_tiebreak_weight", 0.05),
            genre_neighbor_pool_enabled=bool(style_cfg_raw.get("genre_neighbor_pool_enabled", False)),
            genre_neighbor_pool_size=int(style_cfg_raw.get("genre_neighbor_pool_size", 500)),
            genre_neighbor_min_similarity=float(style_cfg_raw.get("genre_neighbor_min_similarity", 0.25)),
            genre_neighbor_min_confidence=(
                None
                if style_cfg_raw.get("genre_neighbor_min_confidence") is None
                else float(style_cfg_raw.get("genre_neighbor_min_confidence", 0.50))
            ),
            genre_neighbor_compatible_threshold=float(style_cfg_raw.get("genre_neighbor_compatible_threshold", 0.35)),
            genre_neighbor_conflict_threshold=float(style_cfg_raw.get("genre_neighbor_conflict_threshold", 0.15)),
            medoid_energy_weight=float(style_cfg_raw.get("medoid_energy_weight", 0.0)),
            energy_feature=str(style_cfg_raw.get("energy_feature", "arousal_p50")),
            energy_slot_lo_pct=float(style_cfg_raw.get("energy_slot_lo_pct", 10.0)),
            energy_slot_hi_pct=float(style_cfg_raw.get("energy_slot_hi_pct", 90.0)),
            dedupe_versions=bool(style_cfg_raw.get("dedupe_versions", True)),
            medoid_popularity_weight=float(style_cfg_raw.get("medoid_popularity_weight", 0.0)),
        )
        playlists_cfg = self.config.config.get("playlists", {}) or {}
        cohesion_mode_effective = cohesion_mode_override or ("dynamic" if dynamic else resolve_cohesion_mode(playlists_cfg))
        artifact_path = ds_cfg.get("artifact_path")
        pool_source = "legacy"

        logger.info(
            "Artist style mode %s: artist=%s cohesion_mode=%s",
            "ENABLED" if style_cfg.enabled else "DISABLED",
            artist_name,
            cohesion_mode_effective,
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
                base_seed = int(ds_cfg.get("random_seed", 0) or 0)
                seed_epoch_val = int(seed_epoch or 0)
                cluster_seed = base_seed + seed_epoch_val

                # Calculate medoid_top_k based on presence setting (max_artist_fraction)
                # to ensure the playlist has the desired number of seed artist piers
                max_artist_fraction = ds_cfg.get("candidate_pool", {}).get("max_artist_fraction", 0.125)
                target_pier_count = max(3, round(track_count * max_artist_fraction))

                # Predict number of clusters to calculate piers per cluster
                artist_key_norm = normalize_artist_key(artist_name)
                artist_track_count = len(_artist_indices_in_bundle(
                    bundle, artist_name, include_collaborations=include_collaborations
                ))
                predicted_k = _select_k(artist_track_count, style_cfg)

                # Calculate medoid_top_k (piers per cluster) to achieve target pier count
                medoid_top_k_calculated = max(1, math.ceil(target_pier_count / predicted_k))

                # Override with explicit config value or seed_epoch logic if present
                if seed_epoch_val > 0 and style_cfg.medoid_top_k:
                    medoid_top_k = max(1, int(style_cfg.medoid_top_k))
                else:
                    medoid_top_k = medoid_top_k_calculated

                logger.info(
                    "Artist presence pier calculation: max_artist_fraction=%.3f target_piers=%d "
                    "predicted_clusters=%d medoid_top_k=%d (track_count=%d)",
                    max_artist_fraction, target_pier_count, predicted_k, medoid_top_k, track_count
                )

                # Popular-seeds activation: when enabled and a Last.fm client exists,
                # override the medoid popularity weight and load per-track popularity
                # values aligned to the bundle. Default path: popularity_values=None
                # (cluster_artist_tracks treats None as no-op, byte-identical to today).
                popularity_values = None
                if popular_seeds_mode in {"on", "fire"} and getattr(self, "lastfm", None) is not None:
                    from datetime import datetime, timezone
                    from src.analyze.popularity_runner import (
                        enrichment_db_path,
                        load_artist_popularity_values,
                    )
                    pop_w = float(style_cfg_raw.get("popular_seeds_weight", 1.0))
                    style_cfg = replace(style_cfg, medoid_popularity_weight=pop_w)
                    popularity_values = load_artist_popularity_values(
                        bundle, artist_name, client=self.lastfm,
                        db_path=enrichment_db_path(),
                        metadata_db_path=self.config.get("library", "database_path", default="data/metadata.db"),
                        limit=int((self.config.config.get("lastfm", {}) or {}).get("artist_top_tracks_limit", 50)),
                        max_age_days=int(style_cfg_raw.get("popularity_max_age_days", 30)),
                        now_iso=datetime.now(timezone.utc).isoformat(),
                        include_collaborations=include_collaborations,
                    )

                clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
                    bundle=bundle,
                    artist_name=artist_name,
                    cfg=style_cfg,
                    random_seed=cluster_seed,
                    sonic_variant=sonic_variant_cfg,
                    medoid_top_k=medoid_top_k,
                    include_collaborations=include_collaborations,
                    excluded_track_ids=seed_recency_excluded_ids if exclude_seed_tracks_from_recency else None,
                    popularity_values=popularity_values,
                    metadata_db_path=self.config.get("library", "database_path", default="data/metadata.db"),
                )
                # 🔥 Pure-hits piers: override cluster medoids with the artist's top-N
                # most-popular tracks (selection only — order_clusters still sequences them).
                if popular_seeds_mode == "fire" and popularity_values is not None:
                    _all_members = [i for _cluster in clusters for i in _cluster]
                    _fire_piers = select_popular_piers(_all_members, popularity_values, target_pier_count)
                    if _fire_piers:
                        logger.info(
                            "Popular Seeds 🔥: overriding %d cluster-medoid piers with top-%d popular tracks",
                            len(medoids), len(_fire_piers),
                        )
                        medoids = _fire_piers
                    else:
                        logger.warning(
                            "Popular Seeds 🔥: no popular piers resolved (uncached artist?) — "
                            "falling back to cluster-medoid piers",
                        )
                if not medoids:
                    raise ValueError("Style clustering returned no medoids")
                ordered_medoids = order_clusters(medoids, X_norm)

                # Cap medoids to target_pier_count to avoid ceiling overshoot
                # (e.g., 5 clusters × ceil(6/5)=2 per cluster = 10, but we want 6)
                if len(ordered_medoids) > target_pier_count:
                    logger.info(
                        "Capping medoids from %d to target_pier_count=%d",
                        len(ordered_medoids), target_pier_count
                    )
                    ordered_medoids = ordered_medoids[:target_pier_count]

                ordered_medoids = self._filter_title_excluded_bundle_indices(
                    bundle,
                    ordered_medoids,
                    context="artist_style_piers",
                )
                if not ordered_medoids:
                    raise ValueError("Artist style piers empty after title exclusions")

                cluster_piers = medoids_by_cluster

                # Global admission floor (same as DS candidate admission)
                min_sonic = get_min_sonic_similarity(ds_cfg.get("candidate_pool", {}), cohesion_mode_effective)
                artist_key_norm = normalize_artist_key(artist_name)

                external_pool = build_balanced_candidate_pool(
                    bundle=bundle,
                    cluster_piers=cluster_piers,
                    X_norm=X_norm,
                    per_cluster_size=style_cfg.per_cluster_candidate_pool_size,
                    pool_balance_mode=style_cfg.pool_balance_mode,
                    global_floor=min_sonic,
                    artist_key=artist_key_norm,
                    artist_name=artist_name,
                    include_collaborations=include_collaborations,
                )
                genre_neighbor_pool: List[str] = []
                if style_cfg.genre_neighbor_pool_enabled:
                    genre_method_cfg = (
                        self.config.get("playlists", "genre_similarity", default={}) or {}
                    ).get("method", "ensemble")
                    genre_neighbor_pool = build_genre_neighbor_candidate_pool(
                        bundle=bundle,
                        pier_indices=ordered_medoids,
                        artist_key=artist_key_norm,
                        pool_size=style_cfg.genre_neighbor_pool_size,
                        min_similarity=style_cfg.genre_neighbor_min_similarity,
                        min_confidence=style_cfg.genre_neighbor_min_confidence,
                        compatible_threshold=style_cfg.genre_neighbor_compatible_threshold,
                        conflict_threshold=style_cfg.genre_neighbor_conflict_threshold,
                        genre_method=genre_method_cfg or "ensemble",
                        artist_name=artist_name,
                        include_collaborations=include_collaborations,
                    )
                # Internal connectors disabled for Artist mode - seed artist should ONLY appear as piers
                internal_connector_ids = []
                pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]
                if popular_seeds_mode in {"on", "fire"}:
                    # Diagnostic: log where each chosen pier sits on the artist's
                    # Last.fm popularity list (the cache is warm from the lazy fetch).
                    from src.analyze.popularity_runner import (
                        enrichment_db_path,
                        log_seed_popularity,
                    )
                    _titles = getattr(bundle, "track_titles", None)
                    pier_titles = [
                        str(_titles[m]) if _titles is not None else "" for m in ordered_medoids
                    ]
                    log_seed_popularity(
                        artist_name, pier_ids, pier_titles, db_path=enrichment_db_path()
                    )
                style_allowed_track_ids = list(dict.fromkeys(
                    pier_ids + external_pool + genre_neighbor_pool + list(internal_connector_ids or [])
                ))
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
                    "cohesion_mode": str(cohesion_mode_effective),
                    "sonic_variant": str(sonic_variant_cfg),
                    "seed_epoch": int(seed_epoch or 0),
                    "medoid_top_k": int(medoid_top_k),
                    "global_sonic_floor": float(min_sonic) if min_sonic is not None else None,
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
                    "genre_neighbor_pool_count": int(len(genre_neighbor_pool)),
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
                ds_defaults = default_ds_config(cohesion_mode_effective, playlist_len=track_count, overrides=ds_cfg)
                pb_tuning = resolve_pier_bridge_tuning(ds_cfg, cohesion_mode_effective)

                # Artist-style can override per-mode pier-bridge weights, but defaults
                # should follow the global pier-bridge tuning for the mode.
                weight_bridge = float(pb_tuning["weight_bridge"])
                weight_transition = float(pb_tuning["weight_transition"])
                weights_raw = style_cfg_raw.get("bridge_score_weights")
                if isinstance(weights_raw, dict):
                    by_mode = weights_raw.get(cohesion_mode_effective)
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
                    mode_val = bridge_floor_raw.get(cohesion_mode_effective)
                    if isinstance(mode_val, (int, float)):
                        bridge_floor = float(mode_val)
                elif isinstance(bridge_floor_raw, (int, float)):
                    bridge_floor = float(bridge_floor_raw)

                # Oops, All Bangers: three-stop popularity mode -> beam penalty strength.
                # Config-tunable (not in the GUI): playlists.bangers.strength_{on,oops}.
                # NOTE: this is the beam-penalty lever (re-ranks within the pool); the real
                # "only bangers" lever is the popularity admission gate (separate work).
                _bangers_cfg = self.config.get("playlists", "bangers", default={}) or {}
                _pop_strength = {
                    "off": 0.0,
                    "on": float(_bangers_cfg.get("strength_on", 0.25)),
                    "oops": float(_bangers_cfg.get("strength_oops", 0.60)),
                }.get(str(popularity_mode or "off"), 0.0)
                _pop_rank_cutoff = _resolve_popularity_rank_cutoff(popularity_mode, _bangers_cfg)

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
                    popularity_penalty_strength=_pop_strength,
                    popularity_rank_cutoff=_pop_rank_cutoff,
                    genre_steering_enabled=bool(pb_tuning.get("genre_steering_enabled", False)),
                    genre_steering_source=str(pb_tuning.get("genre_steering_source", "taxonomy")),
                    weight_genre=float(pb_tuning.get("weight_genre", 0.0)),
                    genre_arc_floor=float(pb_tuning.get("genre_arc_floor", 0.0)),
                    genre_arc_floor_percentile=float(pb_tuning.get("genre_arc_floor_percentile", 0.0)),
                    genre_admission_percentile=float(pb_tuning.get("genre_admission_percentile", 0.0)),
                    variable_bridge_length=bool((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_length", False)),
                    variable_bridge_flex=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_flex", 2)),
                    variable_bridge_band=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_band", 5)),
                    variable_bridge_min_edge=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_min_edge", 0.30)),
                    variable_bridge_epsilon=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_epsilon", 0.02)),
                )
                # Roam corridors (Phase-1): the artist path builds PierBridgeConfig
                # explicitly, so it must apply the roam override itself (no-op if absent).
                pier_cfg = replace(
                    pier_cfg, **roam_kwargs_from_dict((ds_cfg.get("pier_bridge") or {}).get("roam"))
                )

                using_artist_style = True
                pool_source = "artist_style"
                logger.info(
                    "Artist style mode ENABLED: artist=%s cohesion_mode=%s clusters=%d piers=%d allowed_ids=%d internal_connectors=%d",
                    artist_name,
                    cohesion_mode_effective,
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

        fallback_used = False
        if using_artist_style and style_seed_track_id and style_allowed_track_ids:
            # Try with genre gating first, fallback to no-genre-gate mode if genre isolation detected
            try:
                ds_tracks = self._maybe_generate_ds_playlist(
                    seed_track_id=style_seed_track_id,
                    target_length=track_count,
                    mode_override=cohesion_mode_effective,
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
            except ValueError as e:
                error_msg = str(e)
                # Catch any pool-too-small infeasibility: pool_after_gate N < interior_len M
                if "pool_after_gate" in error_msg and "interior_len" in error_msg:
                    logger.warning(
                        "❌ Candidate pool too small for '%s': %s",
                        artist_name, error_msg.split('\n')[0]
                    )
                    logger.info(
                        "🔄 Fallback: Retrying without genre gating (artist=%s, mode=%s)",
                        artist_name,
                        cohesion_mode_effective
                    )

                    # Disable genre gating via playlists.genre_similarity.enabled (the key
                    # _maybe_generate_ds_playlist actually reads).
                    gs_cfg = self.config.config.setdefault("playlists", {}).setdefault("genre_similarity", {})
                    original_gate_enabled = gs_cfg.get("enabled", True)
                    try:
                        gs_cfg["enabled"] = False

                        ds_tracks = self._maybe_generate_ds_playlist(
                            seed_track_id=style_seed_track_id,
                            target_length=track_count,
                            mode_override=cohesion_mode_effective,
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
                        fallback_used = True
                        logger.warning(
                            "✓ Fallback succeeded: Generated playlist for '%s' without genre filtering (quality may be lower)",
                            artist_name
                        )
                    finally:
                        gs_cfg["enabled"] = original_gate_enabled
                else:
                    # Different error, re-raise
                    raise
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
                        mode_override=cohesion_mode_effective,
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
                transition_weights=(
                    ((self._last_ds_report.get("playlist_stats") or {}).get("playlist") or {}).get("transition_weights")
                ),
            )
            self._last_ds_report["edge_scores"] = recomputed_edges
            playlist_stats = self._last_ds_report.get("playlist_stats") or {}
            playlist_stats_playlist = playlist_stats.get("playlist") or {}
            playlist_stats_playlist["edge_scores"] = recomputed_edges
            t_values = [
                float(edge.get("T"))
                for edge in recomputed_edges
                if isinstance(edge, dict)
                and isinstance(edge.get("T"), (int, float))
                and edge.get("T") == edge.get("T")
            ]
            transition_floor = self._last_ds_report.get("transition_floor")
            if transition_floor is None:
                transition_floor = playlist_stats_playlist.get("transition_floor")
            if t_values:
                playlist_stats_playlist["min_transition"] = min(t_values)
                playlist_stats_playlist["mean_transition"] = sum(t_values) / len(t_values)
                if transition_floor is not None:
                    playlist_stats_playlist["below_floor_count"] = sum(
                        1 for value in t_values if value < float(transition_floor)
                    )
            artist_counts: Dict[str, int] = {}
            for track in final_tracks:
                artist = str(track.get("artist") or "Unknown")
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            playlist_stats_playlist["artist_counts"] = artist_counts
            playlist_stats_playlist["distinct_artists"] = len(artist_counts)
            playlist_stats_playlist["edge_metric_source"] = "final_emitted_playlist"
            playlist_stats["playlist"] = playlist_stats_playlist
            self._last_ds_report["playlist_stats"] = playlist_stats
            metrics = self._last_ds_report.get("metrics") or {}
            metrics["below_floor"] = playlist_stats_playlist.get("below_floor_count")
            metrics["min_transition"] = playlist_stats_playlist.get("min_transition")
            metrics["mean_transition"] = playlist_stats_playlist.get("mean_transition")
            metrics["artist_counts"] = artist_counts
            metrics["distinct_artists"] = len(artist_counts)
            metrics["edge_metric_source"] = "final_emitted_playlist"
            self._last_ds_report["metrics"] = metrics
            if os.environ.get("PLAYLIST_DIAG_RECENCY"):
                logger.info("Recency diag: post-order filtering disabled; no edge diff computed.")
        title = f"Auto: {artist_name}"
        self._print_playlist_report(final_tracks, artist_name=artist_name, dynamic=dynamic, verbose_edges=verbose)

        _swap_log = (
            (self._last_ds_report.get("playlist_stats") or {})
            .get("playlist", {})
            .get("edge_repair_swap_log")
            if getattr(self, "_last_ds_report", None)
            else None
        )
        if _swap_log:
            from src.playlist.reporter import emit_edge_repair_log as _emit_repair_log
            _emit_repair_log(_swap_log)

        # Diagnostic: per-edge audit table (opt-in via emit_selected_edge_audit: true in config)
        if bool(getattr(self, "_last_ds_report", {}) and self._last_ds_report.get("emit_selected_edge_audit", False)):
            from src.playlist.reporter import emit_selected_edge_audit as _emit_edge_audit
            _beam_comps = (
                (self._last_ds_report.get("playlist_stats") or {})
                .get("playlist", {})
                .get("beam_edge_components")
            )
            _audit_rows = _build_edge_audit_rows(
                edge_scores_list=self._last_ds_report.get("edge_scores") or [],
                tracks=final_tracks,
                transition_floor=float(self._last_ds_report.get("transition_floor") or 0.20),
                beam_components=_beam_comps or None,
            )
            _emit_edge_audit(
                _audit_rows,
                transition_floor=float(self._last_ds_report.get("transition_floor") or 0.20),
            )

        # Add fallback info if genre-gating was bypassed
        result = {
            'title': title,
            'name': title,
            'artists': (artist_name,),
            'genres': [],
            'tracks': final_tracks,
            'track_ids': [str(t.get('rating_key') or t.get('track_id') or '') for t in final_tracks],
            'ds_report': getattr(self, "_last_ds_report", None),
        }
        if fallback_used:
            result['genre_gate_fallback'] = True
            result['quality_warning'] = f"{artist_name}'s genres are isolated from your library - playlist generated without genre filtering"
        return result

    def create_playlist_for_genre(
        self,
        genre_name: Optional[str] = None,
        track_count: int = 30,
        dynamic: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        cohesion_mode_override: Optional[str] = None,
        genre: Optional[str] = None,
        num_tracks: Optional[int] = None,
        mode: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single playlist for a specific genre without requiring listening history

        Args:
            genre_name: Name of the genre to create playlist for
            track_count: Target number of tracks in playlist
            dynamic: Use dynamic mode (lower genre similarity threshold)
            dry_run: Preview mode (for testing)
            verbose: Enable verbose edge scoring output
            cohesion_mode_override: Override cohesion mode (narrow/dynamic/discover/sonic_only)

        Returns:
            Playlist dictionary with tracks and metadata, or None if unable to create
        """
        from src.genre.normalize_unified import normalize_genre_token

        if genre_name is None:
            genre_name = genre
        if genre_name is None:
            raise ValueError("genre_name is required")
        if num_tracks is not None:
            track_count = num_tracks
        if mode is not None:
            cohesion_mode_override = mode
            dynamic = mode == "dynamic"
        if random_seed is not None:
            self.config.config.setdefault("playlists", {}).setdefault("ds_pipeline", {})["random_seed"] = random_seed

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
        valid_tracks = self._filter_title_excluded_tracks(
            valid_tracks,
            context="genre_seed_selection",
        )
        if len(valid_tracks) == 0:
            raise ValueError(
                f"Genre '{genre_name}' has no tracks after duration/title exclusions "
                "(duration=47s-720s)"
            )
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

        # Determine cohesion mode
        playlists_cfg = self.config.config.get("playlists", {}) or {}
        cohesion_mode_effective = cohesion_mode_override or ("dynamic" if dynamic else resolve_cohesion_mode(playlists_cfg))

        logger.info(f"Running pipeline with mode={cohesion_mode_effective}")

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
                mode_override=cohesion_mode_effective,
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
                            mode_override=cohesion_mode_effective,
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
            'name': title,
            'artists': tuple(set(t.get('artist') for t in final_tracks if t.get('artist'))),
            'genres': [normalized_genre],
            'tracks': final_tracks,
            'track_ids': [str(t.get('rating_key') or t.get('track_id') or '') for t in final_tracks],
            'ds_report': getattr(self, "_last_ds_report", None),
        }

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


    def _create_playlists_from_single_artists(
        self,
        artist_seeds: Dict[str, List[Dict[str, Any]]],
        history: List[Dict[str, Any]],
        dynamic: bool = False,
        pipeline_override: Optional[str] = None,
        cohesion_mode_override: Optional[str] = None,
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
        logger.info("Generating playlists from single artists:")
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
                bridge_floor_strict=style_cfg_raw.get("bridge_floor", {}).get("strict", 0.10),
                bridge_floor_narrow=style_cfg_raw.get("bridge_floor", {}).get("narrow", 0.05),
                bridge_floor_dynamic=style_cfg_raw.get("bridge_floor", {}).get("dynamic", 0.02),
                bridge_weight=style_cfg_raw.get("bridge_score_weights", {}).get("bridge", 0.7),
                transition_weight=style_cfg_raw.get("bridge_score_weights", {}).get("transition", 0.3),
                genre_tiebreak_weight=style_cfg_raw.get("genre_tiebreak_weight", 0.05),
                genre_neighbor_pool_enabled=bool(style_cfg_raw.get("genre_neighbor_pool_enabled", False)),
                genre_neighbor_pool_size=int(style_cfg_raw.get("genre_neighbor_pool_size", 500)),
                genre_neighbor_min_similarity=float(style_cfg_raw.get("genre_neighbor_min_similarity", 0.25)),
                genre_neighbor_min_confidence=(
                    None
                    if style_cfg_raw.get("genre_neighbor_min_confidence") is None
                    else float(style_cfg_raw.get("genre_neighbor_min_confidence", 0.50))
                ),
                genre_neighbor_compatible_threshold=float(style_cfg_raw.get("genre_neighbor_compatible_threshold", 0.35)),
                genre_neighbor_conflict_threshold=float(style_cfg_raw.get("genre_neighbor_conflict_threshold", 0.15)),
                medoid_energy_weight=float(style_cfg_raw.get("medoid_energy_weight", 0.0)),
                energy_feature=str(style_cfg_raw.get("energy_feature", "arousal_p50")),
                energy_slot_lo_pct=float(style_cfg_raw.get("energy_slot_lo_pct", 10.0)),
                energy_slot_hi_pct=float(style_cfg_raw.get("energy_slot_hi_pct", 90.0)),
                dedupe_versions=bool(style_cfg_raw.get("dedupe_versions", True)),
                medoid_popularity_weight=float(style_cfg_raw.get("medoid_popularity_weight", 0.0)),
            )
            playlists_cfg = self.config.config.get("playlists", {}) or {}
            cohesion_mode_effective = cohesion_mode_override or ("dynamic" if dynamic else resolve_cohesion_mode(playlists_cfg))
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
                "Artist style mode %s: artist=%s cohesion_mode=%s",
                "ENABLED" if style_cfg.enabled else "DISABLED",
                artist,
                cohesion_mode_effective,
            )
            if style_cfg.enabled and artifact_path:
                try:
                    bundle = load_artifact_bundle(artifact_path)
                    sonic_cfg = self.config.get("playlists", "sonic", default={}) or {}
                    sonic_variant_cfg = resolve_sonic_variant(
                        explicit_variant=getattr(self, "sonic_variant", None),
                        config_variant=ds_cfg.get("sonic_variant") or sonic_cfg.get("sim_variant"),
                    )

                    # Calculate medoid_top_k based on presence setting (max_artist_fraction)
                    max_artist_fraction = ds_cfg.get("candidate_pool", {}).get("max_artist_fraction", 0.125)
                    target_pier_count = max(3, round(target_playlist_size * max_artist_fraction))

                    # Predict number of clusters to calculate piers per cluster
                    artist_key_norm = normalize_artist_key(artist)
                    artist_track_count = sum(
                        1 for ak in bundle.artist_keys
                        if normalize_artist_key(str(ak)) == artist_key_norm
                    )
                    predicted_k = _select_k(artist_track_count, style_cfg)
                    medoid_top_k = max(1, math.ceil(target_pier_count / predicted_k))

                    logger.info(
                        "Artist presence pier calculation: max_artist_fraction=%.3f target_piers=%d "
                        "predicted_clusters=%d medoid_top_k=%d (track_count=%d)",
                        max_artist_fraction, target_pier_count, predicted_k, medoid_top_k, target_playlist_size
                    )

                    clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
                        bundle=bundle,
                        artist_name=artist,
                        cfg=style_cfg,
                        random_seed=ds_cfg.get("random_seed", 0),
                        sonic_variant=sonic_variant_cfg,
                        medoid_top_k=medoid_top_k,
                    )
                    if not medoids:
                        raise ValueError("Style clustering returned no medoids")
                    ordered_medoids = order_clusters(medoids, X_norm)

                    # Cap medoids to target_pier_count to avoid ceiling overshoot
                    # (e.g., 5 clusters × ceil(6/5)=2 per cluster = 10, but we want 6)
                    if len(ordered_medoids) > target_pier_count:
                        logger.info(
                            "Capping medoids from %d to target_pier_count=%d",
                            len(ordered_medoids), target_pier_count
                        )
                        ordered_medoids = ordered_medoids[:target_pier_count]

                    ordered_medoids = self._filter_title_excluded_bundle_indices(
                        bundle,
                        ordered_medoids,
                        context="artist_style_piers",
                    )
                    if not ordered_medoids:
                        raise ValueError("Artist style piers empty after title exclusions")

                    cluster_piers = medoids_by_cluster
                    min_sonic = get_min_sonic_similarity(ds_cfg.get("candidate_pool", {}), cohesion_mode_effective)
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
                    genre_neighbor_pool: List[str] = []
                    if style_cfg.genre_neighbor_pool_enabled:
                        genre_method_cfg = (
                            self.config.get("playlists", "genre_similarity", default={}) or {}
                        ).get("method", "ensemble")
                        genre_neighbor_pool = build_genre_neighbor_candidate_pool(
                            bundle=bundle,
                            pier_indices=ordered_medoids,
                            artist_key=artist_key,
                            pool_size=style_cfg.genre_neighbor_pool_size,
                            min_similarity=style_cfg.genre_neighbor_min_similarity,
                            min_confidence=style_cfg.genre_neighbor_min_confidence,
                            compatible_threshold=style_cfg.genre_neighbor_compatible_threshold,
                            conflict_threshold=style_cfg.genre_neighbor_conflict_threshold,
                            genre_method=genre_method_cfg or "ensemble",
                            artist_name=artist,
                        )
                    # Internal connectors disabled for Artist mode - seed artist should ONLY appear as piers
                    internal_connectors = []
                    pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]
                    allowed_track_ids = list(dict.fromkeys(
                        pier_ids + external_pool + genre_neighbor_pool + internal_connectors
                    ))
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
                    ds_defaults = default_ds_config(cohesion_mode_effective, playlist_len=target_playlist_size, overrides=ds_cfg)
                    pb_tuning = resolve_pier_bridge_tuning(ds_cfg, cohesion_mode_effective)

                    weight_bridge = float(pb_tuning["weight_bridge"])
                    weight_transition = float(pb_tuning["weight_transition"])
                    weights_raw = style_cfg_raw.get("bridge_score_weights")
                    if isinstance(weights_raw, dict):
                        by_mode = weights_raw.get(cohesion_mode_effective)
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
                        mode_val = bridge_floor_raw.get(cohesion_mode_effective)
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
                        genre_steering_enabled=bool(pb_tuning.get("genre_steering_enabled", False)),
                        genre_steering_source=str(pb_tuning.get("genre_steering_source", "taxonomy")),
                        weight_genre=float(pb_tuning.get("weight_genre", 0.0)),
                        genre_arc_floor=float(pb_tuning.get("genre_arc_floor", 0.0)),
                        genre_arc_floor_percentile=float(pb_tuning.get("genre_arc_floor_percentile", 0.0)),
                        genre_admission_percentile=float(pb_tuning.get("genre_admission_percentile", 0.0)),
                        variable_bridge_length=bool((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_length", False)),
                        variable_bridge_flex=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_flex", 2)),
                        variable_bridge_band=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_band", 5)),
                        variable_bridge_min_edge=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_min_edge", 0.30)),
                        variable_bridge_epsilon=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_epsilon", 0.02)),
                    )
                    # Roam corridors (Phase-1): apply the roam override on the explicit
                    # artist-path PierBridgeConfig (no-op if absent).
                    pier_cfg = replace(
                        pier_cfg, **roam_kwargs_from_dict((ds_cfg.get("pier_bridge") or {}).get("roam"))
                    )
                    using_artist_style = True
                    pool_source = "artist_style"
                    logger.info(
                        "Artist style mode ENABLED: artist=%s cohesion_mode=%s clusters=%d piers=%d allowed_ids=%d internal_connectors=%d",
                        artist,
                        cohesion_mode_effective,
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
                mode_override=cohesion_mode_effective,
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
        cohesion_mode_override: Optional[str] = None,
        seed_track_ids: Optional[List[str]] = None,
        popularity_mode: str = "off",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a playlist from explicit seed tracks without requiring an artist name.
        Seed tracks should be provided in "Title - Artist" format (autocomplete default).
        If seed_track_ids are provided, they will be used for exact track matching.
        """
        if not seed_tracks:
            raise ValueError("No seed tracks provided.")

        # Oops, All Bangers: inject the popularity gate cutoff into the DS pipeline
        # config so core.generate_playlist_ds picks it up via pb_overrides (the seed
        # path calls _maybe_generate_ds_playlist with pier_bridge_config=None, so the
        # gate must be read from config). Do NOT force popular_seeds / override piers —
        # seed-mode piers are the user's explicitly chosen seeds.
        _bangers_cfg = self.config.get("playlists", "bangers", default={}) or {}
        _seed_cutoff = _resolve_popularity_rank_cutoff(popularity_mode, _bangers_cfg)
        self.config.config.setdefault("playlists", {}).setdefault("ds_pipeline", {}) \
            .setdefault("pier_bridge", {})["popularity_rank_cutoff"] = _seed_cutoff

        all_library_tracks = self.library.get_all_tracks()

        resolved_seeds: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        # If track IDs are provided, use direct lookup for exact matching
        if seed_track_ids and len(seed_track_ids) == len(seed_tracks):
            logger.info("Using track IDs for exact seed track matching")
            for track_id, display_str in zip(seed_track_ids, seed_tracks):
                track = self.library.get_track_by_key(track_id)
                if not track:
                    logger.warning(
                        "Seed track ID '%s' (%s) not found in library; skipping",
                        track_id,
                        display_str,
                    )
                    continue
                key = str(track.get("rating_key") or track.get("track_id") or "")
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                track["play_count"] = 0
                resolved_seeds.append(track)
        else:
            # Fall back to display string parsing (legacy behavior)
            if seed_track_ids:
                logger.warning(
                    "Track ID count (%d) doesn't match seed count (%d); using display string parsing",
                    len(seed_track_ids),
                    len(seed_tracks),
                )

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

        # Try generation with progressive fallback on insufficient pool
        ds_tracks = None
        fallback_attempts = []

        # Attempt 1: Normal generation
        try:
            ds_tracks = self._maybe_generate_ds_playlist(
                seed_track_id=resolved_seeds[0].get("rating_key"),
                target_length=track_count,
                mode_override=cohesion_mode_override,
                anchor_seed_tracks=resolved_seeds,
                anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
                excluded_track_ids=excluded_ids or None,
                artist_playlist=False,
                pool_source="seeded",
            )
        except ValueError as e:
            error_msg = str(e)
            # Check if this is the specific "insufficient pool" error
            if "pool_after_gate" in error_msg and "interior_len" in error_msg:
                logger.warning("Initial generation failed due to insufficient pool: %s", error_msg)
                fallback_attempts.append(("normal", error_msg))

                # Attempt 2: Allow pier artists in interiors + reduce artist spacing
                logger.info("Fallback 1: Allowing pier artists in interiors, reducing artist spacing to 3")
                try:
                    original_disallow = self.config.config.get("playlists", {}).get("ds_pipeline", {}).get("pier_bridge", {}).get("disallow_pier_artists_in_interiors", True)
                    original_min_gap = self.config.config.get("playlists", {}).get("ds_pipeline", {}).get("artist_spacing_min_gap", 6)

                    # Temporarily relax constraints
                    self.config.config.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})["disallow_pier_artists_in_interiors"] = False
                    self.config.config["playlists"]["ds_pipeline"]["artist_spacing_min_gap"] = 3

                    ds_tracks = self._maybe_generate_ds_playlist(
                        seed_track_id=resolved_seeds[0].get("rating_key"),
                        target_length=track_count,
                        mode_override=cohesion_mode_override,
                        anchor_seed_tracks=resolved_seeds,
                        anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
                        excluded_track_ids=excluded_ids or None,
                        artist_playlist=False,
                        pool_source="seeded",
                    )
                    logger.info("✓ Fallback 1 succeeded (pier artists allowed, min_gap=3)")
                    try:
                        self.config.config["playlists"]["ds_pipeline"]["pier_bridge"]["disallow_pier_artists_in_interiors"] = original_disallow
                        self.config.config["playlists"]["ds_pipeline"]["artist_spacing_min_gap"] = original_min_gap
                    except Exception:
                        pass
                except ValueError as e2:
                    fallback_attempts.append(("allow_pier_gap3", str(e2)))
                    logger.warning("Fallback 1 failed: %s", e2)

                    # Attempt 3: Reduce artist spacing to 0
                    if ds_tracks is None:
                        logger.info("Fallback 2: Reducing artist spacing to 0")
                        try:
                            self.config.config["playlists"]["ds_pipeline"]["artist_spacing_min_gap"] = 0

                            ds_tracks = self._maybe_generate_ds_playlist(
                                seed_track_id=resolved_seeds[0].get("rating_key"),
                                target_length=track_count,
                                mode_override=cohesion_mode_override,
                                anchor_seed_tracks=resolved_seeds,
                                anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
                                excluded_track_ids=excluded_ids or None,
                                artist_playlist=False,
                                pool_source="seeded",
                            )
                            logger.info("✓ Fallback 2 succeeded (pier artists allowed, min_gap=0)")
                        except ValueError as e3:
                            fallback_attempts.append(("allow_pier_gap0", str(e3)))
                            logger.warning("Fallback 2 failed: %s", e3)

                    # Attempt 4: Reduce playlist length to 20 tracks
                    if ds_tracks is None and track_count > 20:
                        logger.info("Fallback 3: Reducing target length to 20 tracks")
                        try:
                            ds_tracks = self._maybe_generate_ds_playlist(
                                seed_track_id=resolved_seeds[0].get("rating_key"),
                                target_length=20,
                                mode_override=cohesion_mode_override,
                                anchor_seed_tracks=resolved_seeds,
                                anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
                                excluded_track_ids=excluded_ids or None,
                                artist_playlist=False,
                                pool_source="seeded",
                            )
                            logger.info("✓ Fallback 3 succeeded (20 tracks, pier artists allowed, min_gap=0)")
                        except ValueError as e4:
                            fallback_attempts.append(("shorter_20", str(e4)))
                            logger.warning("Fallback 3 failed: %s", e4)

                    # Attempt 5: Disable recency filter as last resort
                    if ds_tracks is None and excluded_ids:
                        logger.info("Fallback 4 (LAST RESORT): Disabling recency filter")
                        try:
                            ds_tracks = self._maybe_generate_ds_playlist(
                                seed_track_id=resolved_seeds[0].get("rating_key"),
                                target_length=min(20, track_count),
                                mode_override=cohesion_mode_override,
                                anchor_seed_tracks=resolved_seeds,
                                anchor_seed_ids=[s.get("rating_key") for s in resolved_seeds if s.get("rating_key")],
                                excluded_track_ids=None,  # Disable recency exclusions
                                artist_playlist=False,
                                pool_source="seeded",
                            )
                            logger.info("✓ Fallback 4 succeeded (no recency filter, 20 tracks, pier artists allowed, min_gap=0)")
                        except ValueError as e5:
                            fallback_attempts.append(("no_recency", str(e5)))
                            logger.error("Fallback 4 failed: %s", e5)

                    # Restore original values
                    try:
                        self.config.config["playlists"]["ds_pipeline"]["pier_bridge"]["disallow_pier_artists_in_interiors"] = original_disallow
                        self.config.config["playlists"]["ds_pipeline"]["artist_spacing_min_gap"] = original_min_gap
                    except Exception:
                        pass

                # If all fallbacks failed, provide detailed error
                if ds_tracks is None:
                    fallback_summary = "\n".join([f"  - {mode}: {err[:100]}" for mode, err in fallback_attempts])
                    raise ValueError(
                        f"Unable to generate playlist even with progressive fallbacks.\n"
                        f"Seeds are too sonically/genre-isolated from your library.\n\n"
                        f"Attempts made:\n{fallback_summary}\n\n"
                        f"Suggestions:\n"
                        f"  - Try fewer seeds (2-3 instead of 4)\n"
                        f"  - Choose seeds that are more sonically similar to each other\n"
                        f"  - Add more music to your library in these genres"
                    )
            else:
                # Different error, re-raise
                raise

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
        # Use actual returned length for validation (may be shorter if fallback reduced target)
        actual_target_length = len(ds_tracks) if ds_tracks else track_count
        self._post_order_validate_ds_output(
            ordered_tracks=ds_tracks,
            expected_length=actual_target_length,
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

        _swap_log = (
            (self._last_ds_report.get("playlist_stats") or {})
            .get("playlist", {})
            .get("edge_repair_swap_log")
            if getattr(self, "_last_ds_report", None)
            else None
        )
        if _swap_log:
            from src.playlist.reporter import emit_edge_repair_log as _emit_repair_log
            _emit_repair_log(_swap_log)

        # Diagnostic: per-edge audit table (opt-in via emit_selected_edge_audit: true in config)
        if bool(getattr(self, "_last_ds_report", {}) and self._last_ds_report.get("emit_selected_edge_audit", False)):
            from src.playlist.reporter import emit_selected_edge_audit as _emit_edge_audit
            _beam_comps = (
                (self._last_ds_report.get("playlist_stats") or {})
                .get("playlist", {})
                .get("beam_edge_components")
            )
            _audit_rows = _build_edge_audit_rows(
                edge_scores_list=self._last_ds_report.get("edge_scores") or [],
                tracks=final_tracks,
                transition_floor=float(self._last_ds_report.get("transition_floor") or 0.20),
                beam_components=_beam_comps or None,
            )
            _emit_edge_audit(
                _audit_rows,
                transition_floor=float(self._last_ds_report.get("transition_floor") or 0.20),
            )

        # Add fallback info to report if fallbacks were used
        fallback_used = len(fallback_attempts) > 0
        if fallback_used:
            logger.info("✓ Playlist generated using fallback mode (relaxed constraints)")

        return {
            "title": title,
            "artists": tuple(sorted({t.get("artist") for t in resolved_seeds if t.get("artist")})),
            "genres": [],
            "tracks": final_tracks,
            "ds_report": getattr(self, "_last_ds_report", None),
            "fallback_mode_used": fallback_used,
            "fallback_attempts": fallback_attempts if fallback_used else None,
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
        logger.info("Top 10 most similar pairs:")
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
            last_cohesion_mode=getattr(self, "_last_cohesion_mode", None),
        )


# Example usage
if __name__ == "__main__":
    logger.info("Playlist Generator module loaded")
