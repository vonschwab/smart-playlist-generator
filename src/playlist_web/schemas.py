"""Pydantic schemas for web API request/response serialization."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.playlist.request_models import GeneratePlaylistRequest


class GenerateRequestBody(BaseModel):
    """API request body for playlist generation."""

    mode: str = "artist"
    tracks: int = 30
    artist: Optional[str] = None
    genre: Optional[str] = None
    seed_tracks: list[str] = Field(default_factory=list)
    seed_track_ids: list[str] = Field(default_factory=list)
    genre_mode: Optional[str] = None
    sonic_mode: Optional[str] = None
    pace_mode: Optional[str] = None
    include_collaborations: bool = False
    exclude_seed_tracks_from_recency: bool = False

    def to_request(self) -> GeneratePlaylistRequest:
        """Convert to internal GeneratePlaylistRequest for worker processing."""
        return GeneratePlaylistRequest(
            mode=self.mode,
            tracks=self.tracks,
            artist=self.artist,
            genre=self.genre,
            seed_tracks=list(self.seed_tracks),
            seed_track_ids=list(self.seed_track_ids),
            genre_mode=self.genre_mode,
            sonic_mode=self.sonic_mode,
            pace_mode=self.pace_mode,
            include_collaborations=self.include_collaborations,
            exclude_seed_tracks_from_recency=self.exclude_seed_tracks_from_recency,
        )


class TrackOut(BaseModel):
    """Response model for a single track in a playlist."""

    position: int
    rating_key: Optional[str] = None
    artist: str = "Unknown"
    title: str = "Unknown"
    album: str = ""
    duration_ms: int = 0
    file_path: str = ""
    sonic_similarity: Optional[float] = None
    genre_similarity: Optional[float] = None
    genres: list[str] = Field(default_factory=list)


class MetricsOut(BaseModel):
    """Response model for playlist quality metrics."""

    mean_transition: Optional[float] = None
    min_transition: Optional[float] = None
    p10_transition: Optional[float] = None
    p90_transition: Optional[float] = None
    distinct_artists: Optional[int] = None


class PlaylistOut(BaseModel):
    """Response model for a generated playlist."""

    name: str = "Generated Playlist"
    track_count: int = 0
    tracks: list[TrackOut] = Field(default_factory=list)
    metrics: MetricsOut = Field(default_factory=MetricsOut)

    @classmethod
    def from_worker(cls, raw: dict[str, Any]) -> "PlaylistOut":
        """Parse a worker result dict into a PlaylistOut response."""
        # Extract tracks, handling missing fields gracefully
        tracks_raw = raw.get("tracks", [])
        tracks = [
            TrackOut(
                position=t.get("position", 0),
                rating_key=t.get("rating_key"),
                artist=t.get("artist", "Unknown"),
                title=t.get("title", "Unknown"),
                album=t.get("album", ""),
                duration_ms=t.get("duration_ms", 0),
                file_path=t.get("file_path", ""),
                sonic_similarity=t.get("sonic_similarity"),
                genre_similarity=t.get("genre_similarity"),
                genres=t.get("genres", []),
            )
            for t in tracks_raw
        ]

        # Extract metrics
        metrics_raw = raw.get("metrics") or {}
        metrics = MetricsOut(
            mean_transition=metrics_raw.get("mean_transition"),
            min_transition=metrics_raw.get("min_transition"),
            p10_transition=metrics_raw.get("p10_transition"),
            p90_transition=metrics_raw.get("p90_transition"),
            distinct_artists=metrics_raw.get("distinct_artists"),
        )

        return cls(
            name=raw.get("name", "Generated Playlist"),
            track_count=raw.get("track_count", len(tracks_raw)),
            tracks=tracks,
            metrics=metrics,
        )


class JobOut(BaseModel):
    """Response model for a job status query or completion."""

    job_id: str
    status: str
    stage: str = ""
    error: Optional[str] = None
    playlist: Optional[PlaylistOut] = None
