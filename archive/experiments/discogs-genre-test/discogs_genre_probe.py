"""
Discogs genre probe that only queries releases already in the local library.

Workflow:
- read albums from the SQLite library (no writes)
- search Discogs releases by artist + title
- fetch release/master genres/styles for matched releases
- aggregate genre/style counts
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import yaml


API_BASE = "https://api.discogs.com"
USER_AGENT = "PlaylistGenerator/DiscogsGenreTest"
DEFAULT_PAGE_SIZE = 15
SLEEP_SECONDS = 1.1  # stay under 60 req/min


def normalize(text: str) -> str:
    """Lowercase alphanumerics for simple fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


EDITION_KEYWORDS = [
    "deluxe",
    "remaster",
    "remastered",
    "expanded",
    "anniversary",
    "edition",
    "studio masters",
    "complete",
    "discography",
]


def strip_edition_suffix(title: str) -> str:
    """
    Remove parenthetical/bracketed edition/descriptive suffixes that hurt matching,
    e.g., "(Expanded & Remastered 2004)" or "[Dischord 40 1989]".
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
    """Cheap similarity without extra deps."""
    if not a or not b:
        return 0.0
    # token overlap score
    a_tokens = set(re.findall(r"[a-z0-9]+", a.lower()))
    b_tokens = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), len(b_tokens))


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


class DiscogsClient:
    def __init__(self, token: Optional[str]) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.token = token
        if token:
            self.session.headers.update({"Authorization": f"Discogs token={token}"})

    def _get(self, path: str, params: Dict) -> Dict:
        resp = self.session.get(f"{API_BASE}{path}", params=params, timeout=15)
        if resp.status_code == 429:
            raise RuntimeError("Discogs rate limit hit (429). Slow down.")
        resp.raise_for_status()
        return resp.json()

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

    # Primary search with artist filter
    primary = client.search_release(album.artist, search_title, album.year)
    top = pick_best(primary, lowered_threshold)
    if top:
        return top

    # Fallback: title-only search (no artist filter) for compilations/alt editions.
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


def load_config_token(config_path: Optional[Path]) -> Optional[str]:
    if not config_path or not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    discogs = data.get("discogs") if isinstance(data, dict) else None
    if isinstance(discogs, dict):
        return discogs.get("token")
    return None


def iter_albums(conn: sqlite3.Connection, artist_filter: Optional[str], limit: Optional[int], per_artist: Optional[int]) -> Iterable[AlbumRow]:
    query = "SELECT album_id, title, artist, release_year FROM albums"
    params: List = []
    if artist_filter:
        query += " WHERE artist = ?"
        params.append(artist_filter)
    query += " ORDER BY artist, title"
    cur = conn.execute(query, params)
    per_artist_counts: Dict[str, int] = defaultdict(int)
    for row in cur:
        if limit is not None and limit <= 0:
            break
        album = AlbumRow(album_id=row[0], title=row[1], artist=row[2], year=row[3])
        count = per_artist_counts[album.artist]
        if per_artist is not None and count >= per_artist:
            continue
        per_artist_counts[album.artist] += 1
        if limit is not None:
            limit -= 1
        yield album


def run(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    token = args.token or os.getenv("DISCOGS_TOKEN") or load_config_token(Path(args.config) if args.config else None)
    client = DiscogsClient(token)

    conn = sqlite3.connect(db_path)
    matches: List[MatchResult] = []
    genre_counter: Counter = Counter()
    style_counter: Counter = Counter()

    for album in iter_albums(conn, args.artist, args.limit, args.per_artist):
        try:
            match = best_match(client, album, args.threshold, strict_artist=args.strict_artist)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[warn] search failed for {album.artist} - {album.title}: {exc}")
            continue
        if not match:
            print(f"[miss] {album.artist} - {album.title}")
            continue
        release_id = match.get("id")
        master_id = match.get("master_id")
        score = score_result(match, album)
        try:
            genres, styles = fetch_genres(client, release_id, master_id)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[warn] fetch failed for {album.artist} - {album.title}: {exc}")
            continue
        matches.append(MatchResult(album=album, release_id=release_id, master_id=master_id, score=score, genres=genres, styles=styles))
        genre_counter.update(g.lower() for g in genres)
        style_counter.update(s.lower() for s in styles)
        genres_str = ", ".join(genres) if genres else "none"
        styles_str = ", ".join(styles) if styles else "none"
        print(f"[hit] {album.artist} - {album.title} -> release {release_id} (score {score:.2f}) genres=[{genres_str}] styles=[{styles_str}]")
        time.sleep(SLEEP_SECONDS)

    if not matches:
        print("No matches found.")
        return

    print("\nTop genres:")
    for genre, count in genre_counter.most_common(10):
        print(f"  {genre}: {count}")

    print("\nTop styles:")
    for style, count in style_counter.most_common(10):
        print(f"  {style}: {count}")


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
    parser = argparse.ArgumentParser(description="Probe Discogs genres for existing library releases only.")
    parser.add_argument("--db", default=str(default_db_path()), help="Path to metadata.db (defaults to ../data/metadata.db if it exists).")
    parser.add_argument("--config", default="config.yaml", help="Config file to optionally read discogs.token from.")
    parser.add_argument("--token", help="Discogs user token (overrides env DISCOGS_TOKEN).")
    parser.add_argument("--artist", help="Only process this artist.")
    parser.add_argument("--limit", type=int, help="Max albums to process.")
    parser.add_argument("--per-artist", type=int, default=3, help="Max albums per artist to sample.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum fuzzy match score to accept.")
    parser.add_argument("--strict-artist", action="store_true", help="Require decent artist similarity to accept a match (helps avoid wrong-artist hits).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
