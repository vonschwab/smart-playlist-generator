"""Server-side Plex playlist export, wrapping the existing PlexExporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PlexNotConfigured(RuntimeError):
    """Raised when Plex settings are missing from config."""


def _load_plex_config(config_path: str) -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    import yaml

    with open(cfg_file, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("plex", {}) or {}


def run_plex_export(title: str, tracks: list[dict], config_path: str) -> str:
    """Export a playlist to Plex. Returns the Plex playlist key.

    Raises PlexNotConfigured if plex.enabled is false or base_url/token are missing,
    or RuntimeError on export failure.
    """
    plex = _load_plex_config(config_path)
    base_url = plex.get("base_url")
    token = plex.get("token")
    if not plex.get("enabled", False) or not base_url or not token:
        raise PlexNotConfigured(
            "Plex is not configured (set plex.enabled: true, plex.base_url and plex.token in config.yaml)."
        )

    from src.plex_exporter import PlexExporter

    exporter = PlexExporter(
        base_url,
        token,
        music_section=plex.get("music_section"),
        verify_ssl=plex.get("verify_ssl", True),
        path_map=plex.get("path_map"),
    )
    plex_tracks: list[dict[str, Any]] = [
        {
            "rating_key": t.get("rating_key") or t.get("track_id"),
            "title": t.get("title", ""),
            "artist": t.get("artist", ""),
            "file_path": t.get("file_path", ""),
        }
        for t in tracks
    ]
    key = exporter.export_playlist(title, plex_tracks)
    if not key:
        raise RuntimeError("Plex export returned no playlist key.")
    return str(key)
