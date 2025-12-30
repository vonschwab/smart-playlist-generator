"""
Fetch MusicBrainz recording IDs (MBIDs) for tracks in metadata.db without touching audio files.

This uses the public MusicBrainz API (rate-limited to ~1 req/sec) to search by
artist + title and optionally check duration tolerance.

Usage:
    python scripts/fetch_mbids_musicbrainz.py --db data/metadata.db --limit 500 --duration-tolerance-ms 3000

Options:
    --db                       Path to metadata.db (default: data/metadata.db)
    --limit                    Max tracks to process (default: 200) to avoid hammering the API
    --duration-tolerance-ms    Accept MBID match if abs(duration_ms - MB duration) <= tolerance (default: 3000)
    --force                    Overwrite existing MBIDs (default: skip tracks that already have mbid)
    --artist-like              Optional LIKE filter on artist to narrow scope (e.g., "%gilberto%")

Notes:
    - Requires network access to musicbrainz.org.
    - Respects 1 request/second rate limit with a small sleep.
    - Writes MBIDs into tracks.musicbrainz_id only; audio files are untouched.
"""

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import re
import difflib
import requests
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path as SysPath

# Ensure project root on sys.path for src imports
_ROOT = SysPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.string_utils import normalize_song_title

MBZ_BASE = "https://musicbrainz.org/ws/2"
NO_MATCH_MARKER = "__NO_MATCH__"
ERROR_MARKER = "__ERROR__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MusicBrainz MBIDs into metadata.db")
    parser.add_argument("--db", default="data/metadata.db", help="Path to metadata.db")
    parser.add_argument("--limit", type=int, default=200, help="Max tracks to process (default: 200)")
    parser.add_argument(
        "--duration-tolerance-ms",
        type=int,
        default=3000,
        help="Duration tolerance when comparing to MB result (default: 3000 ms)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process all tracks (alias for --force-all)",
    )
    parser.add_argument(
        "--force-no-match",
        action="store_true",
        help="Reprocess tracks previously marked __NO_MATCH__",
    )
    parser.add_argument(
        "--force-error",
        action="store_true",
        help="Reprocess tracks previously marked __ERROR__",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Process all tracks regardless of existing musicbrainz_id",
    )
    parser.add_argument(
        "--artist-like",
        help='Optional SQL LIKE filter on artist to narrow scope (e.g., "%gilberto%")',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per request on transient errors (default: 3)",
    )
    return parser.parse_args()


def get_session(timeout: float, max_retries: int) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "playlist-generator/mbid-fetch"})
    session.request = lambda *args, **kwargs: requests.Session.request(
        session,
        *args,
        timeout=timeout,
        **kwargs,
    )
    return session


def load_candidates(
    conn: sqlite3.Connection,
    limit: int,
    force: bool,
    force_no_match: bool,
    force_error: bool,
    force_all: bool,
    artist_like: Optional[str],
):
    cur = conn.cursor()
    sql = """
        SELECT track_id, title, artist, duration_ms
        FROM tracks
        WHERE title IS NOT NULL AND artist IS NOT NULL
    """
    params = []
    if not (force_all or force):
        clauses = ["musicbrainz_id IS NULL", "trim(musicbrainz_id) = ''"]
        if force_no_match:
            clauses.append("musicbrainz_id = ?")
            params.append(NO_MATCH_MARKER)
        if force_error:
            clauses.append("musicbrainz_id = ?")
            params.append(ERROR_MARKER)
        sql += " AND (" + " OR ".join(clauses) + ")"
    if artist_like:
        sql += " AND artist LIKE ?"
        params.append(artist_like)
    sql += " ORDER BY track_id LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    return cur.fetchall()


def search_recording(session: requests.Session, query: str, duration_ms: Optional[int], tolerance_ms: int) -> Optional[Dict[str, Any]]:
    params = {"query": query, "fmt": "json", "limit": 5}
    resp = session.get(f"{MBZ_BASE}/recording/", params=params)
    resp.raise_for_status()
    data = resp.json()
    recordings = data.get("recordings") or []
    if not recordings:
        return None

    # pick best by score/duration proximity
    best = None
    best_score = -1
    for rec in recordings:
        score = rec.get("score", 0)
        rec_len = rec.get("length")
        if duration_ms is not None and rec_len is not None:
            if abs(int(rec_len) - int(duration_ms)) > tolerance_ms:
                continue
        if score > best_score:
            best_score = score
            best = rec
    if best and best.get("id"):
        return best
    return None


def search_recording_relaxed(
    session: requests.Session,
    artist: str,
    title: str,
    duration_ms: Optional[int],
    tolerance_ms: int,
) -> Optional[Dict[str, Any]]:
    """
    Try a couple of query variants for better hit rate:
    1) fielded artist/title
    2) plain text artist title (no field qualifiers)
    3) fielded with stripped parentheticals in title
    4) fielded with punctuation-stripped title
    5) title with [feat/feat.] stripped
    6) recording-only queries with normalized titles (fallback)
    """
    variants = []
    variants.append(f'artist:"{artist}" AND recording:"{title}"')
    variants.append(f'"{artist}" "{title}"')
    clean_title = re.sub(r"\s*\([^)]*\)", "", title).strip()
    if clean_title and clean_title != title:
        variants.append(f'artist:"{artist}" AND recording:"{clean_title}"')
    punctless_title = re.sub(r"[^\w\s]", " ", title).strip()
    if punctless_title and punctless_title not in {title, clean_title}:
        variants.append(f'artist:"{artist}" AND recording:"{punctless_title}"')
    # Strip [feat. ...] or (feat. ...) segments from title
    no_feat_title = re.sub(r"\s*[\[(].*?feat\.?.*?[\])]", "", title, flags=re.IGNORECASE).strip()
    if no_feat_title and no_feat_title not in {title, clean_title, punctless_title}:
        variants.append(f'artist:"{artist}" AND recording:"{no_feat_title}"')
    # Try moving feature info out of artist/title splits for collaborations
    artist_clean = re.sub(r"\s*[\[(].*?feat\.?.*?[\])]", "", artist, flags=re.IGNORECASE).strip()
    if artist_clean and artist_clean != artist:
        variants.append(f'artist:"{artist_clean}" AND recording:"{title}"')
        if no_feat_title:
            variants.append(f'artist:"{artist_clean}" AND recording:"{no_feat_title}"')
    feat_parts = re.split(r"\s*(?:feat\.?|featuring)\s*", title, flags=re.IGNORECASE)
    if len(feat_parts) > 1:
        title_head = feat_parts[0].strip()
        title_feats = " ".join(feat_parts[1:]).strip()
        if title_head:
            variants.append(f'artist:"{artist}" AND recording:"{title_head}"')
            if title_feats:
                variants.append(f'artist:"{artist} {title_feats}" AND recording:"{title_head}"')
    norm_title = normalize_song_title(title) or ""
    if norm_title and norm_title not in {title, clean_title, punctless_title, no_feat_title}:
        variants.append(f'artist:"{artist}" AND recording:"{norm_title}"')
        variants.append(f'"{norm_title}"')
        variants.append(f'recording:"{norm_title}"')
    else:
        variants.append(f'recording:"{title}"')

    for q in variants:
        rec = search_recording(session, q, duration_ms, tolerance_ms)
        if rec:
            return rec
    return None


def normalize_simple(text: str) -> str:
    """Lowercase and remove non-alphanumerics for similarity checks."""
    text = normalize_song_title(text) or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_simple(a), normalize_simple(b)).ratio()


def best_artist_similarity(artist: str, mb_artist: str) -> float:
    """Allow reordered credits (e.g., A & B vs B & A)."""
    base = similarity(artist, mb_artist)
    split_re = re.compile(r"\s*(?:&|/|,|\+|;|feat\.?|featuring)\s*", re.IGNORECASE)
    artist_parts = [p for p in split_re.split(artist) if p]
    mb_parts = [p for p in split_re.split(mb_artist) if p]
    if artist_parts and mb_parts:
        artist_join = " & ".join(sorted(artist_parts))
        mb_join = " & ".join(sorted(mb_parts))
        base = max(base, similarity(artist_join, mb_join))
    return base


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    session = get_session(timeout=args.timeout, max_retries=args.max_retries)
    conn = sqlite3.connect(db_path)
    candidates = load_candidates(
        conn,
        args.limit,
        args.force,
        args.force_no_match,
        args.force_error,
        args.force_all,
        args.artist_like,
    )
    print(f"Loaded {len(candidates)} candidate tracks")

    cur = conn.cursor()
    updated = 0
    skipped = 0
    errors = 0
    processed = 0

    last_call = 0.0
    for track_id, title, artist, duration_ms in candidates:
        processed += 1
        # Respect 1 req/sec
        elapsed = time.perf_counter() - last_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        try:
            rec = search_recording_relaxed(session, artist, title, duration_ms, args.duration_tolerance_ms)
            last_call = time.perf_counter()
            if rec:
                mbid = rec.get("id")
                mb_title = rec.get("title", "")
                mb_score = rec.get("score", 0)
                artist_credit = rec.get("artist-credit") or []
                credit_names = []
                for ac in artist_credit:
                    if isinstance(ac, dict) and ac.get("name"):
                        credit_names.append(ac["name"])
                mb_artist = " & ".join(credit_names)
                mb_len = rec.get("length")
                dur_diff = None
                if duration_ms is not None and mb_len is not None:
                    dur_diff = abs(int(mb_len) - int(duration_ms))
                title_sim = similarity(title, mb_title)
                artist_sim = best_artist_similarity(artist, mb_artist or "")

                # Basic guard against false positives (loosened artist threshold; titles must still align)
                if artist_sim < 0.40 or title_sim < 0.55:
                    skipped += 1
                    print(f"[REJECT] {artist} - {title} vs {mb_artist} - {mb_title} (score={mb_score}, artist_sim={artist_sim:.2f}, title_sim={title_sim:.2f}, dur_diff={dur_diff})")
                    continue

                # If MB length is present and duration check fails, reject; if missing, accept on high score/sim
                if duration_ms is not None and mb_len is not None and dur_diff is not None:
                    if dur_diff > args.duration_tolerance_ms:
                        skipped += 1
                        print(f"[REJECT] {artist} - {title} vs {mb_artist} - {mb_title} (duration mismatch: {dur_diff} ms)")
                        continue
                elif mb_len is None:
                    # Accept missing duration only if confidence is high
                    if mb_score < 80 and (artist_sim < 0.7 or title_sim < 0.7):
                        skipped += 1
                        print(f"[REJECT] {artist} - {title} vs {mb_artist} - {mb_title} (no duration; low confidence score={mb_score}, artist_sim={artist_sim:.2f}, title_sim={title_sim:.2f})")
                        continue

                cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (mbid, track_id))
                updated += cur.rowcount
                print(f"[MATCH] {artist} - {title} -> {mbid} | MB: {mb_artist} - {mb_title} | score={mb_score}, artist_sim={artist_sim:.2f}, title_sim={title_sim:.2f}, dur_diff={dur_diff}")
                if updated % 50 == 0:
                    conn.commit()
            else:
                skipped += 1
                cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (NO_MATCH_MARKER, track_id))
                print(f"[NO MATCH] {artist} - {title}")
        except Exception as exc:
            errors += 1
            cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (ERROR_MARKER, track_id))
            print(f"Error on {artist} - {title}: {exc}")
            last_call = time.perf_counter()

        if processed % 200 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(f"Done. Updated={updated}, skipped(no match)={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
