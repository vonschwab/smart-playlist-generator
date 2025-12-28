"""
Plex Playlist Exporter - Creates Plex playlists from generated tracks.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)


class PlexExporter:
    """Exports playlists to a Plex server using its HTTP API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        music_section: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: int = 15,
        replace_existing: bool = True,
        path_map: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.music_section = music_section
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.replace_existing = replace_existing
        self.path_map = self._normalize_path_map(path_map)

        self.session = requests.Session()
        self._section_key: Optional[str] = None
        self._machine_id: Optional[str] = None
        self._path_cache: Dict[str, Optional[str]] = {}
        self._title_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._path_index: Optional[Dict[str, str]] = None

    def _normcase_path(self, path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    def _normalize_path_map(self, path_map: Optional[List[Dict[str, str]]]) -> List[Tuple[str, str]]:
        if not path_map:
            return []
        normalized = []
        for entry in path_map:
            src = entry.get("from")
            dst = entry.get("to")
            if src and dst:
                normalized.append((self._normcase_path(src), dst))
        return normalized

    def _join_mapped_path(self, dst: str, suffix: str) -> str:
        if not suffix:
            return dst
        sep = "\\" if ("\\" in dst and "/" not in dst) else "/"
        dst_clean = dst.rstrip("\\/")
        suffix_clean = suffix.lstrip("\\/")
        suffix_clean = suffix_clean.replace("\\", sep).replace("/", sep)
        return f"{dst_clean}{sep}{suffix_clean}"

    def _apply_path_map(self, path: str) -> str:
        if not self.path_map:
            return path
        norm = self._normcase_path(path)
        for src_norm, dst in self.path_map:
            if norm.startswith(src_norm):
                suffix = norm[len(src_norm):]
                return self._join_mapped_path(dst, suffix)
        return path

    def _normalize_local_path(self, path: str) -> str:
        mapped = self._apply_path_map(path)
        return self._normcase_path(mapped)

    def _normalize_plex_path(self, path: str) -> str:
        return self._normcase_path(path)

    def _paths_match(self, local_path: str, plex_path: str) -> bool:
        return self._normalize_local_path(local_path) == self._normalize_plex_path(plex_path)

    def _build_path_index(self) -> None:
        if self._path_index is not None:
            return
        index: Dict[str, str] = {}
        section_key = self._get_music_section_key()
        start = 0
        page_size = 1000
        total: Optional[int] = None
        while True:
            root = self._request(
                "GET",
                f"/library/sections/{section_key}/all",
                params={
                    "type": 10,
                    "X-Plex-Container-Start": start,
                    "X-Plex-Container-Size": page_size,
                },
            )
            tracks = list(root.findall(".//Track"))
            for track in tracks:
                rating_key = track.get("ratingKey")
                if not rating_key:
                    continue
                for part in track.findall(".//Part"):
                    part_path = part.get("file")
                    if not part_path:
                        continue
                    norm = self._normalize_plex_path(part_path)
                    if norm not in index:
                        index[norm] = rating_key

            if total is None:
                raw_total = root.get("totalSize") or root.get("total") or root.get("size")
                try:
                    total = int(raw_total) if raw_total is not None else None
                except (TypeError, ValueError):
                    total = None

            start += page_size
            if total is not None and start >= total:
                break
            if len(tracks) < page_size:
                break
        self._path_index = index
        logger.info("Plex path index ready (%d tracks)", len(index))

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        allow_empty: bool = False,
    ) -> Optional[ET.Element]:
        url = f"{self.base_url}{path}"
        headers = {"X-Plex-Token": self.token}
        resp = self.session.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Plex API request failed ({resp.status_code}) for {path}")
        # Handle empty responses (common for DELETE operations)
        if not resp.text or not resp.text.strip():
            if allow_empty:
                return None
            raise RuntimeError(f"Empty response from Plex API for {path}")
        try:
            return ET.fromstring(resp.text)
        except ET.ParseError as exc:
            # Log response content for debugging
            logger.debug("Plex response that failed to parse: %s", resp.text[:500])
            if allow_empty:
                return None
            raise RuntimeError(f"Failed to parse Plex XML response for {path}") from exc

    def _get_machine_id(self) -> str:
        if self._machine_id:
            return self._machine_id
        root = self._request("GET", "/")
        machine_id = root.get("machineIdentifier")
        if not machine_id:
            raise RuntimeError("Plex machineIdentifier not found")
        self._machine_id = machine_id
        return machine_id

    def _get_music_section_key(self) -> str:
        if self._section_key:
            return self._section_key
        root = self._request("GET", "/library/sections")
        section_key = None
        for directory in root.findall(".//Directory"):
            if directory.get("type") != "artist":
                continue
            title = directory.get("title") or ""
            key = directory.get("key")
            if self.music_section and title.lower() == self.music_section.lower():
                section_key = key
                break
            if not self.music_section and key:
                section_key = key
                break
        if not section_key:
            raise RuntimeError("Plex music library section not found")
        self._section_key = section_key
        return section_key

    def _search_tracks(self, query: str) -> List[ET.Element]:
        section_key = self._get_music_section_key()
        root = self._request(
            "GET",
            f"/library/sections/{section_key}/search",
            params={"type": 10, "query": query},
        )
        return list(root.findall(".//Track"))

    def _track_key_by_path(self, file_path: str) -> Optional[str]:
        normalized = self._normalize_local_path(file_path)
        if normalized in self._path_cache:
            return self._path_cache[normalized]

        try:
            self._build_path_index()
            if self._path_index:
                key = self._path_index.get(normalized)
                if key:
                    self._path_cache[normalized] = key
                    return key
        except Exception:
            logger.warning("Plex path index unavailable; falling back to search")

        query = Path(file_path).stem
        matches = self._search_tracks(query)
        for track in matches:
            for part in track.findall(".//Part"):
                part_path = part.get("file")
                if not part_path:
                    continue
                if self._paths_match(file_path, part_path):
                    key = track.get("ratingKey")
                    self._path_cache[normalized] = key
                    return key

        self._path_cache[normalized] = None
        return None

    def _track_key_by_title(self, artist: str, title: str) -> Optional[str]:
        key = (artist.casefold(), title.casefold())
        if key in self._title_cache:
            return self._title_cache[key]

        query = f"{artist} {title}".strip()
        matches = self._search_tracks(query)
        for track in matches:
            track_title = (track.get("title") or "").casefold()
            track_artist = (track.get("grandparentTitle") or "").casefold()
            if track_title == title.casefold() and track_artist == artist.casefold():
                rating_key = track.get("ratingKey")
                self._title_cache[key] = rating_key
                return rating_key

        self._title_cache[key] = None
        return None

    def _lookup_track_key(self, track: Dict[str, Any]) -> Optional[str]:
        file_path = track.get("file_path")
        if file_path:
            key = self._track_key_by_path(file_path)
            if key:
                return key
        artist = track.get("artist")
        title = track.get("title")
        if artist and title:
            return self._track_key_by_title(artist, title)
        return None

    def _find_playlist_key(self, title: str) -> Optional[str]:
        root = self._request("GET", "/playlists")
        for playlist in root.findall(".//Playlist"):
            if (playlist.get("title") or "") == title:
                return playlist.get("ratingKey")
        return None

    def _delete_playlist(self, playlist_key: str) -> None:
        self._request("DELETE", f"/playlists/{playlist_key}", allow_empty=True)
        logger.debug("Deleted existing Plex playlist: %s", playlist_key)

    def export_playlist(self, title: str, tracks: List[Dict[str, Any]]) -> Optional[str]:
        if not tracks:
            logger.warning("Skipping Plex export (no tracks in playlist)")
            return None

        rating_keys: List[str] = []
        missing = 0
        for track in tracks:
            key = self._lookup_track_key(track)
            if key:
                rating_keys.append(key)
            else:
                missing += 1

        if not rating_keys:
            logger.warning("Skipping Plex export (no tracks matched in Plex library)")
            return None

        if missing:
            logger.warning("Plex export skipped %d tracks with no match", missing)

        if self.replace_existing:
            existing_key = self._find_playlist_key(title)
            if existing_key:
                logger.info("Replacing existing Plex playlist '%s' (key=%s)", title, existing_key)
                self._delete_playlist(existing_key)

        machine_id = self._get_machine_id()
        uri = f"server://{machine_id}/com.plexapp.plugins.library/library/metadata/{','.join(rating_keys)}"
        params = {
            "type": "audio",
            "title": title,
            "smart": 0,
            "uri": uri,
        }
        root = self._request("POST", "/playlists", params=params)
        playlist = root.find(".//Playlist")
        if playlist is None:
            return None
        return playlist.get("ratingKey")
