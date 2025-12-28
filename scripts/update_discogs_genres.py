#!/usr/bin/env python3
"""
Fetch Discogs genres/styles for library albums and upsert into album_genres.

Notes:
- Matching uses artist + normalized title (edition stripping) with optional strict-artist check.
- Tags are kept as-is (lowercased/trimmed), no alias expansion.
- Source labels: discogs_release, discogs_master.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import yaml

API_BASE = "https://api.discogs.com"
USER_AGENT = "PlaylistGenerator/DiscogsGenreImport"
DEFAULT_PAGE_SIZE = 15


class TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()

    def consume(self, tokens: float = 1.0) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.updated_at = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            need = tokens - self.tokens
            sleep_for = need / self.rate
            time.sleep(sleep_for)


# --- Normalization helpers ----------------------------------------------------

def strip_edition_suffix(title: str) -> str:
    """
    Remove edition/descriptive suffixes that hurt matching, but keep core title.
    """
    cleaned = title
    keyword_pattern = (
        r"(deluxe|remaster|remastered|expanded|anniversary|edition|studio masters|"
        r"box set|bonus tracks?|bonus disc|special edition|limited edition|"
        r"collector's edition|super deluxe|reissue|re-issue)"
    )
    # Remove parenthetical/bracketed chunks containing edition keywords
    cleaned = re.sub(r"[\[\(][^\]\)]*?" + keyword_pattern + r"[^\]\)]*?[\]\)]", "", cleaned, flags=re.IGNORECASE)
    # Remove anniversary phrases like ": 25th Anniversary Expanded Edition" or "- 30th Anniversary Remaster"
    cleaned = re.sub(
        r"[:\-\u2013\u2014]?\s*\d{1,3}th\s+anniversary(?:\s+(?:expanded|deluxe)?\s*edition|\s+remaster(?:ed)?|\s+edition)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing dash/colon descriptors containing edition keywords
    cleaned = re.sub(
        r"[:\-\u2013\u2014]\s*" + keyword_pattern + r"(?:\s+(?:edition|expanded edition|deluxe edition|remaster(?:ed)?|version))?\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove simple region markers like "(Japanese)", "(Japan)", "(UK)", "(US)"
    cleaned = re.sub(r"\((japanese|japan|uk|us|usa|uk version|us version|japan version)\)", "", cleaned, flags=re.IGNORECASE)
    # Clean stray trailing punctuation/whitespace left by removal
    cleaned = re.sub(r"[:\-\u2013\u2014]\s*$", "", cleaned).strip()
    # Collapse double spaces after removal
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def split_search_title(raw: str) -> Tuple[str, str]:
    """
    Discogs search `title` often looks like "Artist - Album".
    Return (artist_part, album_part).
    """
    parts = raw.split(" - ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", raw


def similarity(a: str, b: str) -> float:
    """Simple token overlap similarity."""
    if not a or not b:
        return 0.0
    a_tokens = set(re.findall(r"[a-z0-9]+", a.lower()))
    b_tokens = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), len(b_tokens))


# --- Data classes -------------------------------------------------------------

@dataclass
class AlbumRow:
    album_id: str
    title: str
    artist: str
    year: Optional[int]


@dataclass
class MatchResult:
    album: AlbumRow
    release_id: int
    master_id: Optional[int]
    score: float
    genres: List[str]
    styles: List[str]


# --- Discogs client -----------------------------------------------------------

class DiscogsClient:
    def __init__(self, token: Optional[str]) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Discogs token={token}"})
        self.token = token
        # Conservative rate: ~0.7 req/sec, no burst
        self.bucket = TokenBucket(rate_per_sec=0.7, capacity=1)

    def _get(self, path: str, params: Dict) -> Dict:
        backoff = [5, 10, 20]  # seconds
        for i, delay in enumerate([0] + backoff):
            if delay:
                time.sleep(delay)
            self.bucket.consume()
            resp = self.session.get(f"{API_BASE}{path}", params=params, timeout=20)
            if resp.status_code == 429:
                if i == len(backoff):
                    raise RuntimeError("Discogs rate limit hit (429). Slow down.")
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Discogs rate limit hit (429). Slow down.")

    def search_release(self, artist: Optional[str], title: str, year: Optional[int]) -> List[Dict]:
        params = {
            "type": "release",
            "release_title": title,
            "per_page": DEFAULT_PAGE_SIZE,
            "page": 1,
        }
        if artist:
            params["artist"] = artist
        if year:
            params["year"] = year
        return self._get("/database/search", params).get("results", [])

    def get_release(self, release_id: int) -> Dict:
        return self._get(f"/releases/{release_id}", {})

    def get_master(self, master_id: int) -> Dict:
        return self._get(f"/masters/{master_id}", {})


# --- Matching -----------------------------------------------------------------

def score_result(result: Dict, album: AlbumRow) -> float:
    raw_title = result.get("title", "") or ""
    artist_part, album_part = split_search_title(raw_title)
    album_score = similarity(album_part, strip_edition_suffix(album.title))
    artist_score = similarity(artist_part, album.artist)
    year_score = 0.1 if album.year and result.get("year") == album.year else 0.0
    return 0.6 * album_score + 0.3 * artist_score + year_score


def artist_similarity(result: Dict, album: AlbumRow) -> float:
    raw_title = result.get("title", "") or ""
    artist_part, _ = split_search_title(raw_title)
    return similarity(artist_part, album.artist)


def best_match(client: DiscogsClient, album: AlbumRow, threshold: float, strict_artist: bool = False) -> Optional[Dict]:
    search_title = strip_edition_suffix(album.title)
    anniversary_adjust = "anniversary" in album.title.lower() and search_title != album.title
    lowered_threshold = threshold * 0.8 if anniversary_adjust else threshold
    artist_min = 0.5 if strict_artist else 0.0

    def pick_best(results: List[Dict], score_cutoff: float) -> Optional[Dict]:
        if not results:
            return None
        scored = [(score_result(r, album), r) for r in results]
        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, r in scored:
            if sc >= score_cutoff and artist_similarity(r, album) >= artist_min:
                return r
        return None

    primary = client.search_release(album.artist, search_title, album.year)
    top = pick_best(primary, lowered_threshold)
    if top:
        return top

    alt_results = client.search_release(None, search_title, album.year)
    alt_threshold = lowered_threshold * 0.9 if anniversary_adjust else threshold * 0.9
    return pick_best(alt_results, alt_threshold)


def fetch_genres(client: DiscogsClient, release_id: int, master_id: Optional[int]) -> Tuple[List[str], List[str]]:
    rel = client.get_release(release_id)
    genres = rel.get("genres") or []
    styles = rel.get("styles") or []
    if master_id:
        try:
            master = client.get_master(master_id)
            genres = genres or master.get("genres") or []
            styles = styles or master.get("styles") or []
        except requests.HTTPError:
            pass
    return genres, styles


# --- DB helpers ---------------------------------------------------------------

def normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().lower())


def discogs_status(conn: sqlite3.Connection, album_id: str) -> Tuple[bool, bool]:
    """
    Return (has_data, has_empty) for discogs sources on this album.
    """
    rows = conn.execute(
        "SELECT genre FROM album_genres WHERE album_id=? AND source IN ('discogs_release','discogs_master')",
        (album_id,),
    ).fetchall()
    if not rows:
        return False, False
    has_empty = any(r[0] == "__EMPTY__" for r in rows)
    has_data = any(r[0] != "__EMPTY__" for r in rows)
    return has_data, has_empty


def _compute_album_id(artist: str, title: str, existing_id: Optional[str]) -> str:
    """Use stored album_id if present; otherwise derive deterministic id from artist|title."""
    if existing_id:
        return existing_id
    key = f"{artist}|{title}".lower().encode("utf-8", "ignore")
    return hashlib.md5(key).hexdigest()[:16]


def iter_albums(conn: sqlite3.Connection, artist_filter: Optional[str], limit: Optional[int], per_artist: Optional[int]) -> Iterable[AlbumRow]:
    """
    Iterate albums derived from tracks (covers new library additions).
    Orders by most recently modified track per album to surface new additions first.
    """
    params: List = []
    where = "WHERE album IS NOT NULL AND artist IS NOT NULL AND TRIM(album) != '' AND TRIM(artist) != ''"
    if artist_filter:
        where += " AND artist = ?"
        params.append(artist_filter)

    cur = conn.execute(
        f"""
        SELECT album_id, MIN(album) as album, MIN(artist) as artist, MAX(file_modified) as last_modified
        FROM tracks
        {where}
        GROUP BY album_id
        ORDER BY last_modified DESC
        """,
        params,
    )

    per_artist_counts: Dict[str, int] = {}
    remaining = limit if limit is not None else None

    for row in cur:
        if remaining is not None and remaining <= 0:
            break
        album_id = _compute_album_id(row[2] or "", row[1] or "", row[0])
        album = AlbumRow(album_id=album_id, title=row[1], artist=row[2], year=None)
        count = per_artist_counts.get(album.artist, 0)
        if per_artist is not None and count >= per_artist:
            continue
        per_artist_counts[album.artist] = count + 1
        if remaining is not None:
            remaining -= 1
        yield album


def upsert_album_genres(conn: sqlite3.Connection, album_id: str, genres: Sequence[str], source: str, dry_run: bool) -> None:
    if dry_run:
        return
    conn.execute("DELETE FROM album_genres WHERE album_id=? AND source=?", (album_id, source))
    conn.executemany(
        "INSERT INTO album_genres (album_id, genre, source) VALUES (?, ?, ?)",
        [(album_id, g, source) for g in genres],
    )


# --- Runner -------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    token = args.token or os.getenv("DISCOGS_TOKEN") or load_config_token(Path(args.config) if args.config else None)
    if not token:
        sys.exit("Discogs token required: set DISCOGS_TOKEN, pass --token, or add discogs.token to config.yaml")
    client = DiscogsClient(token)

    conn = sqlite3.connect(db_path)
    conn.isolation_level = None  # autocommit

    misses = []
    total_hits = 0
    total_albums = 0

    albums_list = list(iter_albums(conn, args.artist, args.limit, args.per_artist))
    total_to_process = len(albums_list)
    start_time = time.monotonic()
    for idx, album in enumerate(albums_list, start=1):
        total_albums += 1
        has_data, has_empty = discogs_status(conn, album.album_id)

        elapsed = time.monotonic() - start_time
        remaining = total_to_process - idx
        # Rough estimate: ~3 requests per album (search + release + optional master) at ~0.7 req/sec => ~4.3s/album.
        est_seconds = max(0.0, remaining * 4.3)
        est_minutes = int(est_seconds // 60)
        est_secs_rem = int(est_seconds % 60)
        eta_str = f"{est_minutes}m {est_secs_rem}s" if est_minutes else f"{est_secs_rem}s"
        prefix = f"[{idx}/{total_to_process}] [{eta_str} left]"
        if has_data:
            print(f"{prefix} [skip] {album.artist} - {album.title} (discogs already present)")
            continue
        if has_empty and not args.recheck_empty:
            print(f"{prefix} [skip] {album.artist} - {album.title} (previous discogs __EMPTY__)")
            continue
        try:
            match = best_match(client, album, args.threshold, strict_artist=args.strict_artist)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{prefix} [warn] search failed for {album.artist} - {album.title}: {exc}")
            misses.append((album, "search_error", str(exc)))
            continue
        if not match:
            print(f"{prefix} [miss] {album.artist} - {album.title}")
            upsert_album_genres(conn, album.album_id, ["__EMPTY__"], "discogs_release", args.dry_run)
            misses.append((album, "no_match", ""))
            continue
        release_id = match.get("id")
        master_id = match.get("master_id")
        score = score_result(match, album)
        try:
            genres, styles = fetch_genres(client, release_id, master_id)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{prefix} [warn] fetch failed for {album.artist} - {album.title}: {exc}")
            misses.append((album, "fetch_error", str(exc)))
            continue

        norm_genres = [normalize_tag(g) for g in genres if g]
        norm_styles = [normalize_tag(s) for s in styles if s]

        # Write album-level genres/styles with distinct sources
        upsert_album_genres(conn, album.album_id, norm_genres, "discogs_release", args.dry_run)
        if master_id:
            upsert_album_genres(conn, album.album_id, norm_styles, "discogs_master", args.dry_run)

        total_hits += 1
        genres_str = ", ".join(norm_genres) if norm_genres else "none"
        styles_str = ", ".join(norm_styles) if norm_styles else "none"
        print(f"{prefix} [hit] {album.artist} - {album.title} -> release {release_id} (score {score:.2f}) genres=[{genres_str}] styles=[{styles_str}]")

    print(f"\nProcessed albums: {total_albums}, hits: {total_hits}, misses: {len(misses)}")
    if misses and args.miss_log:
        with Path(args.miss_log).open("w", encoding="utf-8") as f:
            for album, reason, detail in misses:
                f.write(f"{album.artist}\t{album.title}\t{reason}\t{detail}\n")
        print(f"Misses logged to {args.miss_log}")

    conn.close()


def load_config_token(config_path: Optional[Path]) -> Optional[str]:
    if not config_path or not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    discogs = data.get("discogs") if isinstance(data, dict) else None
    if isinstance(discogs, dict):
        return discogs.get("token")
    return None


def default_db_path() -> Path:
    here = Path(__file__).resolve()
    options = [
        here.parent.parent / "data" / "metadata.db",
        here.parent.parent / "metadata.db",
    ]
    for path in options:
        if path.exists():
            return path
    return options[0]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Discogs genres/styles for library albums and upsert into album_genres.")
    parser.add_argument("--db", default=str(default_db_path()), help="Path to metadata.db (defaults to ../data/metadata.db if it exists).")
    parser.add_argument("--config", default="config.yaml", help="Config file to optionally read discogs.token from.")
    parser.add_argument("--token", help="Discogs user token (overrides env DISCOGS_TOKEN).")
    parser.add_argument("--artist", help="Only process this artist.")
    parser.add_argument("--limit", type=int, help="Max albums to process.")
    parser.add_argument("--per-artist", type=int, default=None, help="Max albums per artist to sample (default: all).")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum fuzzy match score to accept.")
    parser.add_argument("--strict-artist", action="store_true", help="Require decent artist similarity to accept a match (helps avoid wrong-artist hits).")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to the database.")
    parser.add_argument("--miss-log", help="Path to write miss records (tsv).")
    parser.add_argument("--recheck-empty", action="store_true", help="Retry albums previously marked __EMPTY__ for Discogs; otherwise skips them.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
