"""AuditEmitter — encapsulates the optional run-audit state.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

The orchestrator emits a structured run-audit (preflight + final_failure
or final_success events, persisted as a markdown report) when
``audit_run`` is enabled in pier_bridge overrides. Previously the
orchestrator threaded three state variables — ``audit_events``,
``audit_context``, ``audit_path`` — and repeated the same
None-check + append + flush + log boilerplate at every emission site.

This module bundles the state into an ``AuditEmitter`` and collapses
the boilerplate into ``append()`` and ``flush()`` methods. The
behavior is preserved: pier-bridge builder still receives
``audit.events`` directly so it can append segment-level events from
inside the build.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.features.artifacts import ArtifactBundle
from src.playlist.run_audit import (
    RunAuditContext,
    RunAuditEvent,
    now_utc_iso,
    write_markdown_report,
)

logger = logging.getLogger(__name__)


class AuditEmitter:
    """Holds optional (events, context, path) state for the pipeline run."""

    def __init__(self, audit_cfg: Any) -> None:
        self.cfg = audit_cfg
        self._events: Optional[List[RunAuditEvent]] = (
            [] if bool(audit_cfg.enabled) else None
        )
        self._context: Optional[RunAuditContext] = None
        self._path: Optional[Path] = None

    @property
    def active(self) -> bool:
        """True if audit collection is enabled (events list exists)."""
        return self._events is not None

    @property
    def events(self) -> Optional[List[RunAuditEvent]]:
        """The events list (None if audit disabled).

        Returned by reference because pier_bridge_builder needs to
        append segment-level events directly.
        """
        return self._events

    @property
    def context(self) -> Optional[RunAuditContext]:
        return self._context

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def ensure_context(
        self,
        *,
        bundle: ArtifactBundle,
        seed_idx: int,
        seed_track_id: str,
        mode: str,
        dry_run: bool,
        artifact_path: Any,
        sonic_variant: Optional[str],
        allowed_ids_count: int,
        pool_source: Optional[str],
        artist_style_enabled: bool,
        artist_playlist: bool,
        audit_context_extra: Optional[Dict[str, Any]],
    ) -> None:
        """Lazily build the RunAuditContext and audit_path on first need.

        No-op if audit is inactive or already bootstrapped. Mirrors the
        legacy inline initialization byte-for-byte (timestamp format,
        run_id format, seed_artist fallback chain).
        """
        if not self.active or self._context is not None:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"ds_{mode}_{ts}_{uuid.uuid4().hex[:8]}"
        seed_artist_val: Optional[str] = None
        try:
            if bundle.track_artists is not None:
                seed_artist_val = str(bundle.track_artists[seed_idx])
            elif bundle.artist_keys is not None:
                seed_artist_val = str(bundle.artist_keys[seed_idx])
        except Exception:
            seed_artist_val = None
        self._context = RunAuditContext(
            timestamp_utc=now_utc_iso(),
            run_id=run_id,
            ds_mode=str(mode),
            seed_track_id=str(seed_track_id),
            seed_artist=seed_artist_val,
            dry_run=bool(dry_run),
            artifact_path=str(artifact_path),
            sonic_variant=str(sonic_variant) if sonic_variant else None,
            allowed_ids_count=int(allowed_ids_count),
            pool_source=str(pool_source) if pool_source is not None else None,
            artist_style_enabled=bool(artist_style_enabled),
            artist_playlist=bool(artist_playlist),
            extra=dict(audit_context_extra or {}),
        )
        self._path = Path(self.cfg.out_dir) / f"{self._context.run_id}.md"

    def has_kind(self, kind: str) -> bool:
        """Whether an event of the given ``kind`` has already been appended."""
        if not self.active:
            return False
        return any(e.kind == kind for e in (self._events or []))

    def append(self, kind: str, payload: Dict[str, Any]) -> None:
        """Append a RunAuditEvent (no-op when audit is inactive)."""
        if not self.active or self._events is None:
            return
        self._events.append(
            RunAuditEvent(kind=kind, ts_utc=now_utc_iso(), payload=payload)
        )

    def flush(self) -> None:
        """Write the markdown report if all three pieces of state are set.

        Logs and swallows any write error so the failure path that called
        ``flush()`` still gets to raise its own ValueError downstream.
        """
        if not self.active or self._context is None or self._path is None:
            return
        try:
            write_markdown_report(
                context=self._context,
                events=self._events,
                path=self._path,
                max_bytes=int(self.cfg.max_bytes),
            )
        except Exception as exc:
            logger.exception("Failed to write run audit report: %s", exc)

    def can_flush(self) -> bool:
        """All three pieces of state present (events + context + path)."""
        return self.active and self._context is not None and self._path is not None
