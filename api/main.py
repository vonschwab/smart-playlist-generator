import os
import random
import re
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow importing existing generator modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config  # type: ignore
from src.local_library_client import LocalLibraryClient  # type: ignore
from src.playlist_generator import PlaylistGenerator  # type: ignore

from api.services.lastfm_service import fetch_artist_top_tracks  # type: ignore

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "metadata.db"
CONFIG_PATH = Path(os.getenv("PLAYLIST_CONFIG_PATH", ROOT_DIR / "config.yaml"))

app = FastAPI(title="Playlist Generator API")

config: Optional[Config] = None
library_client: Optional[LocalLibraryClient] = None
playlist_generator: Optional[PlaylistGenerator] = None
_cache_table_initialized = False
settings_cache: Dict[str, object] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_db_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection to the metadata database.
    Raises an HTTP error if the DB file is missing.
    """
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"metadata.db not found at {DB_PATH}",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _compute_library_status(conn: sqlite3.Connection) -> Dict[str, float]:
    """Return track totals and coverage statistics from the database."""
    cur = conn.cursor()

    total_tracks = cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    sonic_analyzed = cur.execute(
        """
        SELECT COUNT(*) FROM tracks
        WHERE sonic_analyzed_at IS NOT NULL
           OR sonic_features IS NOT NULL
        """
    ).fetchone()[0]

    genre_covered = cur.execute(
        """
        SELECT COUNT(*) FROM tracks t
        WHERE EXISTS (
            SELECT 1 FROM track_genres tg
            WHERE tg.track_id = t.track_id
        )
        OR EXISTS (
            SELECT 1 FROM album_genres ag
            WHERE ag.album_id = t.album_id
        )
        OR EXISTS (
            SELECT 1 FROM artist_genres ar
            WHERE ar.artist = t.artist
        )
        """
    ).fetchone()[0]

    def pct(count: int) -> float:
        return round((count / total_tracks) * 100, 2) if total_tracks else 0.0

    return {
        "total_tracks": total_tracks,
        "sonic_analyzed_tracks": sonic_analyzed,
        "sonic_coverage_pct": pct(sonic_analyzed),
        "genre_covered_tracks": genre_covered,
        "genre_coverage_pct": pct(genre_covered),
    }


def _ensure_cache_table() -> None:
    """Create Last.fm artist top tracks cache table if missing."""
    global _cache_table_initialized
    if _cache_table_initialized:
        return
    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lastfm_artist_top_tracks_cache (
                artist_name TEXT NOT NULL,
                track_name TEXT NOT NULL,
                playcount INTEGER NOT NULL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (artist_name, track_name)
            )
            """
        )
        conn.commit()
    _cache_table_initialized = True


def _get_cached_top_tracks(artist_name: str) -> List[Dict[str, object]]:
    _ensure_cache_table()
    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT track_name, playcount, fetched_at
            FROM lastfm_artist_top_tracks_cache
            WHERE LOWER(artist_name) = LOWER(?)
            ORDER BY playcount DESC
            """,
            (artist_name,),
        )
        rows = cur.fetchall()
        return [
            {"track_name": r["track_name"], "playcount": r["playcount"], "fetched_at": r["fetched_at"]}
            for r in rows
        ]


def _write_cache_top_tracks(artist_name: str, tracks: List[Dict[str, object]]) -> None:
    _ensure_cache_table()
    now = int(time.time())
    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM lastfm_artist_top_tracks_cache WHERE LOWER(artist_name) = LOWER(?)",
            (artist_name,),
        )
        cur.executemany(
            """
            INSERT INTO lastfm_artist_top_tracks_cache (artist_name, track_name, playcount, fetched_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    artist_name,
                    t.get("track_name", ""),
                    int(t.get("playcount", 0)),
                    now,
                )
                for t in tracks
            ],
        )
        conn.commit()


def _lastfm_available() -> bool:
    if not config:
        return False
    return bool(config.lastfm_api_key and config.lastfm_username)


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and remove remaster markers for loose matching."""
    t = title.lower()
    t = re.sub(r"\(.*remaster.*\)", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fetch_artist_tracks_from_lastfm(artist_name: str) -> tuple[List[Dict[str, object]], str]:
    if not _lastfm_available():
        return ([], "fallback")

    cached = _get_cached_top_tracks(artist_name)
    week_ago = int(time.time()) - 7 * 24 * 3600
    if cached and cached[0].get("fetched_at", 0) >= week_ago:
        return (cached, "lastfm_cache")

    if not config:
        return ([], "fallback")

    tracks = fetch_artist_top_tracks(
        username=config.lastfm_username,
        api_key=config.lastfm_api_key,
        artist_name=artist_name,
        max_pages=5,
        page_limit=200,
    )
    if tracks:
        _write_cache_top_tracks(artist_name, tracks)
        return (tracks, "lastfm_live")
    return ([], "fallback")


def _get_tracks_by_artist_local(artist_name: str) -> List[Dict[str, object]]:
    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT track_id, title, artist, album, duration_ms, file_path
            FROM tracks
            WHERE LOWER(artist) = LOWER(?)
        """,
            (artist_name,),
        )
        rows = cur.fetchall()
        return [
            {
                "track_id": r["track_id"],
                "title": r["title"] or "",
                "artist": r["artist"] or "",
                "album": r["album"] or "",
                "duration_ms": r["duration_ms"],
                "file_path": r["file_path"],
            }
            for r in rows
        ]


def _map_lastfm_to_local(artist_name: str, top_tracks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Map Last.fm track entries to local tracks by artist and title."""
    local_tracks = _get_tracks_by_artist_local(artist_name)
    title_map = {_normalize_title(t["title"]): t for t in local_tracks}
    mapped = []
    for entry in top_tracks:
        track_name = str(entry.get("track_name") or "")
        playcount = int(entry.get("playcount") or 0)
        norm = _normalize_title(track_name)
        if norm in title_map:
            mapped.append({**title_map[norm], "playcount": playcount})
    mapped.sort(key=lambda t: t.get("playcount", 0), reverse=True)
    return mapped


def _fallback_artist_seed(artist_name: str, random_seed: int) -> Dict[str, object]:
    """Pick a deterministic local track when Last.fm is unavailable."""
    tracks = _get_tracks_by_artist_local(artist_name)
    if not tracks:
        return {"source": "fallback", "primary_seed_track": None, "additional_seed_tracks": [], "top10": []}

    rng = random.Random(random_seed)
    tracks_sorted = sorted(tracks, key=lambda t: (t.get("title") or "").lower())
    primary = tracks_sorted[rng.randrange(len(tracks_sorted))]
    return {
        "source": "fallback",
        "artist_name": artist_name,
        "primary_seed_track": primary,
        "additional_seed_tracks": [],
        "top10": [],
    }


def _resolve_artist_seed(artist_name: str, random_seed: int) -> Dict[str, object]:
    """Resolve artist seed into primary and additional tracks."""
    top_tracks, source = _fetch_artist_tracks_from_lastfm(artist_name)
    rng = random.Random(random_seed)

    if top_tracks:
        mapped = _map_lastfm_to_local(artist_name, top_tracks)
        if mapped:
            primary = mapped[0]
            remaining = mapped[1:]
            additional_count = settings_cache.get("generation.additional_seed_count")
            if additional_count is None:
                additional_count = config.get("generation", "additional_seed_count", default=2) if config else 2
            sample_size = min(int(additional_count), len(remaining))
            additional = rng.sample(remaining, sample_size) if sample_size else []
            return {
                "artist_name": artist_name,
                "source": source,
                "top10": [
                    {"track_name": t["track_name"], "playcount": t["playcount"]}
                    for t in top_tracks[:10]
                ],
                "primary_seed_track": primary,
                "additional_seed_tracks": additional,
            }

    return _fallback_artist_seed(artist_name, random_seed)


def _sanitize_filename(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("Filename required")
    if "/" in raw or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError("Invalid filename")
    clean = re.sub(r'[<>:"/\\|*\x00-\x1F]', "_", raw)
    if not clean.lower().endswith(".m3u"):
        clean += ".m3u"
    return clean


def _resolve_paths(track_ids: List[str]) -> List[str]:
    if not library_client:
        raise HTTPException(status_code=500, detail="Library client unavailable")
    paths: List[str] = []
    for tid in track_ids:
        fp = library_client.get_track_file_path(tid)
        if not fp:
            raise HTTPException(status_code=400, detail=f"Missing file path for track {tid}")
        paths.append(fp)
    return paths


def _write_m3u(paths: List[str], output_dir: str, filename: str) -> str:
    safe_name = _sanitize_filename(filename)
    out_dir = Path(output_dir) if output_dir else Path(
        settings_cache.get("export.default_output_dir")
        or (config.get("playlists", "m3u_export_path", default="E:\\PLAYLISTS") if config else "E:\\PLAYLISTS")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / safe_name
    with dest.open("w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for p in paths:
            f.write(f"{p}\n")
    return str(dest.resolve())


class Seed(BaseModel):
    type: Literal["track"]
    track_id: str = Field(..., description="Track ID from the metadata database")


class ArtistSeed(BaseModel):
    type: Literal["artist"]
    artist_name: str = Field(..., description="Artist name for seed resolution")


class GenerateRequest(BaseModel):
    seed: Annotated[Union[Seed, ArtistSeed], Field(discriminator="type")]
    mode: Optional[Literal["narrow", "dynamic", "discover"]] = None
    length: Optional[int] = Field(None, gt=0, le=200)
    random_seed: Optional[int] = Field(
        None, description="Used for deterministic variations"
    )


class PlaylistTrack(BaseModel):
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    score: Optional[float] = None


class PlaylistResponse(BaseModel):
    playlist_id: str
    seed: Annotated[Union[Seed, ArtistSeed], Field(discriminator="type")]
    resolved_seed: Optional[Dict[str, object]] = None
    mode: str
    length: int
    random_seed: Optional[int]
    tracks: List[PlaylistTrack]


class ExportRequest(BaseModel):
    track_ids: List[str]
    output_dir: Optional[str] = None
    filename: str


class ExportResponse(BaseModel):
    ok: bool
    path: Optional[str] = None
    error: Optional[str] = None


def _init_services() -> None:
    """Initialize shared services once for the API process."""
    global config, library_client, playlist_generator
    if config and library_client and playlist_generator:
        return

    config = Config(str(CONFIG_PATH))
    library_client = LocalLibraryClient(db_path=str(DB_PATH))
    playlist_generator = PlaylistGenerator(
        library_client,
        config,
        lastfm_client=None,
        track_matcher=None,
        metadata_client=None,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _init_services()
    yield
    if library_client:
        library_client.close()


app.router.lifespan_context = lifespan


@app.get("/api/library/status")
def get_library_status() -> Dict[str, float]:
    """Report total tracks, sonic-analysis coverage, and genre coverage."""
    with _get_db_connection() as conn:
        return _compute_library_status(conn)


@app.post("/api/playlist/generate", response_model=PlaylistResponse)
def generate_playlist(request: GenerateRequest) -> PlaylistResponse:
    """
    Generate a playlist from a track or artist seed using existing similarity logic.
    """
    _init_services()
    assert library_client is not None
    assert playlist_generator is not None

    # Apply defaults from settings
    mode = request.mode or settings_cache.get("generation.default_mode") or "narrow"
    length = request.length or settings_cache.get("generation.default_length") or 30

    deterministic = settings_cache.get("generation.deterministic_by_default")
    if deterministic is None:
        deterministic = True
    seed_for_rng = request.random_seed
    if seed_for_rng is None and deterministic:
        seed_for_rng = abs(hash(str(request.seed)))

    rng = random.Random(seed_for_rng or 0)

    if request.seed.type == "track":
        seed_track = library_client.get_track_by_key(request.seed.track_id)
        if not seed_track:
            raise HTTPException(status_code=404, detail="Seed track not found")
        resolution = None
        seeds_for_generator = [seed_track]
        additional_seeds: List[Dict[str, object]] = []
    else:
        resolution = _resolve_artist_seed(request.seed.artist_name, seed_for_rng or 0)
        primary_seed = resolution.get("primary_seed_track")
        if not primary_seed:
            raise HTTPException(status_code=404, detail="No seed tracks found for artist")
        seed_track = library_client.get_track_by_key(primary_seed.get("track_id", ""))
        if not seed_track:
            raise HTTPException(status_code=404, detail="Primary seed track not found locally")

        seeds_for_generator = [seed_track]
        additional_seeds = []
        for add in resolution.get("additional_seed_tracks", []):
            tid = add.get("track_id") or ""
            enriched = library_client.get_track_by_key(tid) or add
            additional_seeds.append(enriched)

    dynamic_mode = mode == "dynamic"
    similar_tracks = playlist_generator.generate_similar_tracks(
        seeds_for_generator, dynamic=dynamic_mode
    )

    def to_playlist_track(track: Dict[str, object], forced_score: Optional[float] = None) -> PlaylistTrack:
        return PlaylistTrack(
            track_id=str(track.get("rating_key") or track.get("track_id") or ""),
            title=str(track.get("title") or "Unknown Title"),
            artist=str(track.get("artist") or "Unknown Artist"),
            album=track.get("album") if track.get("album") is not None else None,
            duration_ms=track.get("duration") if track.get("duration") is not None else track.get("duration_ms"),
            score=forced_score if forced_score is not None else track.get("hybrid_score") or track.get("similarity_score"),
        )

    combined: List[PlaylistTrack] = []
    seen: set[str] = set()

    def add_track(track_dict: Dict[str, object], forced_score: Optional[float] = None) -> None:
        tid = str(track_dict.get("rating_key") or track_dict.get("track_id") or "")
        if not tid or tid in seen:
            return
        seen.add(tid)
        combined.append(to_playlist_track(track_dict, forced_score))

    add_track(seed_track, forced_score=1.0)
    for add in additional_seeds:
        add_track(add)
    for t in similar_tracks:
        add_track(t)

    if len(combined) > 1:
        head, rest = combined[0], combined[1:]
        rng.shuffle(rest)
        combined = [head] + rest

    playlist_id = str(uuid.uuid4())
    resolved_seed_payload = None
    if request.seed.type == "artist":
        resolved_seed_payload = {
            "primary_seed_track_id": seed_track.get("rating_key") or seed_track.get("track_id"),
            "additional_seed_track_ids": [
                t.get("rating_key") or t.get("track_id") for t in additional_seeds
            ],
            "source": resolution.get("source") if resolution else "fallback",
        }

    return PlaylistResponse(
        playlist_id=playlist_id,
        seed=request.seed,
        resolved_seed=resolved_seed_payload,
        mode=mode,
        length=length,
        random_seed=seed_for_rng,
        tracks=combined[: length],
    )


@app.get("/api/search/tracks")
def search_tracks(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> Dict[str, List[Dict[str, str]]]:
    """
    Search tracks by title/artist/album (contains match).
    """
    with _get_db_connection() as conn:
        cur = conn.cursor()
        like = f"%{q}%"
        cur.execute(
            """
            SELECT track_id, title, artist, album
            FROM tracks
            WHERE title LIKE ?
               OR artist LIKE ?
               OR album LIKE ?
            ORDER BY artist, title
            LIMIT ?
            """,
            (like, like, like, limit),
        )
        rows = [
            {
                "track_id": r["track_id"],
                "title": r["title"] or "",
                "artist": r["artist"] or "",
                "album": r["album"] or "",
            }
            for r in cur.fetchall()
        ]
        return {"results": rows}


@app.get("/api/search/artists")
def search_artists(
    q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)
) -> Dict[str, List[Dict[str, object]]]:
    """
    Search artists by name (contains match) and return track counts.
    """
    with _get_db_connection() as conn:
        cur = conn.cursor()
        like = f"%{q}%"
        cur.execute(
            """
            SELECT artist as artist_name, COUNT(*) AS track_count
            FROM tracks
            WHERE artist LIKE ?
            GROUP BY artist
            ORDER BY track_count DESC
            LIMIT ?
            """,
            (like, limit),
        )
        rows = [
            {"artist_name": r["artist_name"] or "", "track_count": r["track_count"]}
            for r in cur.fetchall()
        ]
        return {"results": rows}


@app.get("/api/seed/artist")
def resolve_artist_seed(
    artist_name: str = Query(..., min_length=1),
    random_seed: int = Query(0, ge=0),
) -> Dict[str, object]:
    """Resolve an artist into primary and additional seed tracks."""
    resolution = _resolve_artist_seed(artist_name, random_seed)
    if not resolution.get("primary_seed_track"):
        raise HTTPException(status_code=404, detail="No seed tracks found for artist")
    return resolution


@app.post("/api/playlist/export", response_model=ExportResponse)
def export_playlist(req: ExportRequest) -> ExportResponse:
    if not req.track_ids:
        return ExportResponse(ok=False, error="track_ids is required")
    try:
        paths = _resolve_paths(req.track_ids)
        out_path = _write_m3u(paths, req.output_dir or "", req.filename)
        return ExportResponse(ok=True, path=out_path)
    except ValueError as exc:
        return ExportResponse(ok=False, error=str(exc))
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        return ExportResponse(ok=False, error=str(exc))


@app.get("/api/settings/export")
def get_export_settings() -> Dict[str, str]:
    _init_services()
    if not config:
        raise HTTPException(status_code=500, detail="Config unavailable")
    default_dir = settings_cache.get("export.default_output_dir") or config.get(
        "playlists", "m3u_export_path", default="E:\\PLAYLISTS"
    )
    return {"default_output_dir": default_dir}


def _load_settings() -> Dict[str, object]:
    _init_services()
    if not config:
        raise HTTPException(status_code=500, detail="Config unavailable")
    gen_default_length = config.get("generation", "default_length", default=30)
    gen_default_mode = config.get("generation", "default_mode", default="narrow")
    deterministic = config.get("generation", "deterministic_by_default", default=True)
    additional_seed_count = config.get("generation", "additional_seed_count", default=2)
    export_dir = config.get("playlists", "m3u_export_path", default="E:\\PLAYLISTS")
    settings = {
        "export": {"default_output_dir": export_dir},
        "lastfm": {
            "username": config.lastfm_username or "",
            "api_key": "****" if config.lastfm_api_key else "",
        },
        "generation": {
            "default_mode": gen_default_mode,
            "default_length": gen_default_length,
            "deterministic_by_default": deterministic,
            "additional_seed_count": additional_seed_count,
        },
        "advanced": {},
    }
    settings_cache.update(
        {
            "export.default_output_dir": export_dir,
            "generation.default_mode": gen_default_mode,
            "generation.default_length": gen_default_length,
            "generation.deterministic_by_default": deterministic,
            "generation.additional_seed_count": additional_seed_count,
        }
    )
    return settings


def _write_config(updated: Dict[str, object]) -> None:
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        existing = yaml.safe_load(f)
    merged = existing or {}

    def deep_set(path: List[str], value: object):
        ref = merged
        for key in path[:-1]:
            if key not in ref or not isinstance(ref[key], dict):
                ref[key] = {}
            ref = ref[key]
        ref[path[-1]] = value

    # Export
    export_dir = updated["export"]["default_output_dir"]
    deep_set(["playlists", "m3u_export_path"], export_dir)

    # Last.fm
    lastfm = updated.get("lastfm", {})
    if "username" in lastfm:
        deep_set(["lastfm", "username"], lastfm.get("username") or "")
    if "api_key" in lastfm:
        deep_set(["lastfm", "api_key"], lastfm.get("api_key") or "")

    # Generation
    gen = updated.get("generation", {})
    deep_set(["generation", "default_mode"], gen.get("default_mode"))
    deep_set(["generation", "default_length"], gen.get("default_length"))
    deep_set(["generation", "deterministic_by_default"], gen.get("deterministic_by_default"))
    deep_set(["generation", "additional_seed_count"], gen.get("additional_seed_count"))

    tmp_path = CONFIG_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f)
    tmp_path.replace(CONFIG_PATH)


@app.get("/api/settings")
def get_settings() -> Dict[str, object]:
    return _load_settings()


class SettingsPayload(BaseModel):
    export: Dict[str, str]
    lastfm: Dict[str, Optional[str]]
    generation: Dict[str, object]
    advanced: Optional[Dict[str, object]] = None


@app.put("/api/settings")
def update_settings(payload: SettingsPayload) -> Dict[str, object]:
    global config
    _init_services()
    # Validation
    gen = payload.generation
    mode = gen.get("default_mode", "narrow")
    if mode not in ("narrow", "dynamic", "discover"):
        raise HTTPException(status_code=400, detail="Invalid default_mode")
    length = int(gen.get("default_length", 30))
    if length < 5 or length > 500:
        raise HTTPException(status_code=400, detail="default_length out of range")
    add_seed = int(gen.get("additional_seed_count", 2))
    if add_seed < 0 or add_seed > 5:
        raise HTTPException(status_code=400, detail="additional_seed_count out of range")

    lastfm_api_key = payload.lastfm.get("api_key", "")
    if lastfm_api_key == "****":
        # keep existing
        lastfm_api_key = config.lastfm_api_key if config else ""

    cleaned = {
        "export": {"default_output_dir": payload.export.get("default_output_dir", "")},
        "lastfm": {
            "username": payload.lastfm.get("username") or "",
            "api_key": lastfm_api_key or "",
        },
        "generation": {
            "default_mode": mode,
            "default_length": length,
            "deterministic_by_default": bool(gen.get("deterministic_by_default", True)),
            "additional_seed_count": add_seed,
        },
        "advanced": payload.advanced or {},
    }

    _write_config(cleaned)
    # reload config
    config = Config(str(CONFIG_PATH))
    return _load_settings()
