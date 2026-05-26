"""Deterministic extraction helpers for authoritative release source pages."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class _BandcampTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_tag_link = False
        self._current_text: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        class_names = set((attr_map.get("class") or "").split())
        href = attr_map.get("href") or ""
        if "tag" in class_names and "bandcamp.com/discover/" in href:
            self._in_tag_link = True
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._in_tag_link:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_tag_link:
            return
        raw_tag = " ".join(part.strip() for part in self._current_text if part.strip())
        if raw_tag and raw_tag not in self.tags:
            self.tags.append(raw_tag)
        self._in_tag_link = False
        self._current_text = []


def extract_bandcamp_release_tags(html: str) -> list[str]:
    """Return visible Bandcamp release tags from a supplied release page HTML string."""
    parser = _BandcampTagParser()
    parser.feed(html)
    return parser.tags


def fetch_bandcamp_release_tags(
    source_url: str,
    *,
    fetch_html: Callable[[str], str] | None = None,
) -> list[str]:
    """Fetch exactly the supplied Bandcamp URL and extract its release tags."""
    html = fetch_html(source_url) if fetch_html else _fetch_html(source_url)
    return extract_bandcamp_release_tags(html)


def is_bandcamp_release_url(source_url: str) -> bool:
    """Return true only for Bandcamp album pages suitable for release tag extraction."""
    parsed = urlparse(source_url)
    host = parsed.netloc.casefold()
    path_parts = [part for part in parsed.path.split("/") if part]
    return host.endswith("bandcamp.com") and len(path_parts) >= 2 and path_parts[0] == "album"


def _fetch_html(source_url: str) -> str:
    request = Request(source_url, headers={"User-Agent": "playlist-generator-ai-genre-refinement/1.0"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_lastfm_tags_from_metadata(
    *,
    artist: str,
    album_id: str | None = None,
    metadata_db_path: str | Path = "data/metadata.db",
) -> list[str]:
    """Extract Last.fm-sourced genre tags from local metadata.db (read-only)."""
    from src.genre.normalize_unified import META_TAGS, DROP_TOKENS

    resolved = Path(metadata_db_path).resolve()
    if not resolved.exists():
        return []
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        raw_tags: list[str] = []
        try:
            for row in conn.execute(
                "SELECT DISTINCT genre FROM artist_genres WHERE artist = ? AND source LIKE '%lastfm%'",
                (artist,),
            ):
                if row["genre"]:
                    raw_tags.append(row["genre"])
        except sqlite3.OperationalError:
            pass
        if album_id:
            try:
                for row in conn.execute(
                    "SELECT DISTINCT genre FROM album_genres WHERE album_id = ? AND source LIKE '%lastfm%'",
                    (album_id,),
                ):
                    if row["genre"]:
                        raw_tags.append(row["genre"])
            except sqlite3.OperationalError:
                pass
            try:
                for row in conn.execute(
                    """
                    SELECT DISTINCT genre
                    FROM track_genres
                    WHERE track_id IN (SELECT track_id FROM tracks WHERE album_id = ?)
                      AND source LIKE '%lastfm%'
                    """,
                    (album_id,),
                ):
                    if row["genre"]:
                        raw_tags.append(row["genre"])
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()

    noise = META_TAGS | DROP_TOKENS
    seen: set[str] = set()
    filtered: list[str] = []
    for tag in raw_tags:
        key = tag.strip().casefold()
        if key and key not in noise and key not in seen:
            seen.add(key)
            filtered.append(tag)
    return filtered
