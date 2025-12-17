# REST API Reference

FastAPI backend for Playlist Generator. Provides REST endpoints for playlist generation, library querying, and track search.

## Getting Started

### Start the API Server

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Or using Python module:

```bash
python -m api
```

### Interactive Documentation

Visit: **http://localhost:8000/docs**

Swagger UI with live testing interface for all endpoints.

### Test Connectivity

```bash
curl http://127.0.0.1:8000/api/library/status
```

## Base URL

All endpoints are prefixed with `/api/` (e.g., `/api/library/status`).

## Authentication

Currently no authentication required. Configure CORS for frontend:

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Library Endpoints

### Library Status

Get library statistics and analysis progress.

**Endpoint**: `GET /api/library/status`

**Response** (200 OK):
```json
{
  "total_tracks": 34100,
  "total_duration_hours": 1200,
  "genres_count": 287,
  "coverage": {
    "genres_analyzed": 34000,
    "genres_percent": 99.7,
    "sonic_analyzed": 31000,
    "sonic_percent": 90.9
  }
}
```

**Use Case**: Display library health in UI, show analysis progress.

### Library Search

Search tracks by title, artist, or album.

**Endpoint**: `GET /api/library/search`

**Query Parameters**:
- `q` (required): Search query string
- `limit` (optional, default: 50): Maximum results to return

**Response** (200 OK):
```json
{
  "query": "fela",
  "results": [
    {
      "track_id": "abc123def456",
      "title": "Zombie",
      "artist": "Fela Kuti",
      "album": "Zombie",
      "duration": 243,
      "genres": ["afrobeat", "funk", "world"]
    },
    {
      "track_id": "xyz789uvw012",
      "title": "Shakara",
      "artist": "Fela Kuti",
      "album": "Shakara and Overseas",
      "duration": 301,
      "genres": ["afrobeat", "funk"]
    }
  ],
  "count": 2
}
```

**Use Case**: Autocomplete in track search, build seed track selector.

### List Artists

Get all artists in library with track counts.

**Endpoint**: `GET /api/library/artists`

**Query Parameters**:
- `limit` (optional, default: 100): Maximum artists to return
- `sort` (optional, default: "name"): "name" or "count"

**Response** (200 OK):
```json
{
  "artists": [
    {
      "name": "Fela Kuti",
      "track_count": 42,
      "genres": ["afrobeat", "funk"]
    },
    {
      "name": "Tony Allen",
      "track_count": 15,
      "genres": ["afrobeat", "jazz"]
    }
  ],
  "total": 2
}
```

**Use Case**: Artist browser, seed selection.

### List Genres

Get all genres in library with track counts.

**Endpoint**: `GET /api/library/genres`

**Query Parameters**:
- `limit` (optional, default: 50): Maximum genres to return
- `sort` (optional, default: "count"): "name" or "count"

**Response** (200 OK):
```json
{
  "genres": [
    {
      "name": "rock",
      "track_count": 8500,
      "coverage": 24.9
    },
    {
      "name": "electronic",
      "track_count": 6200,
      "coverage": 18.2
    }
  ],
  "total": 287
}
```

**Use Case**: Genre browser, filter options.

## Playlist Generation Endpoints

### Generate Single Playlist

Create a playlist from a seed track.

**Endpoint**: `POST /api/playlists/generate`

**Request Body**:
```json
{
  "seed_track_id": "abc123def456",
  "count": 50,
  "mode": "ds",
  "constraints": {
    "min_genre_similarity": 0.30,
    "max_per_artist": 3
  }
}
```

**Parameters**:
- `seed_track_id` (required): Track ID to base playlist on
- `count` (optional, default: 50): Number of tracks
- `mode` (optional, default: "ds"): Generation mode
  - `ds`: Balanced hybrid (sonic + genre)
  - `dynamic`: Progressive emphasis on transitions
  - `narrow`: Strict genre coherence
  - `discover`: Genre exploration
  - `legacy`: Pure sonic similarity
- `constraints` (optional): Generation constraints
  - `min_genre_similarity` (0.0-1.0): Minimum genre similarity
  - `max_per_artist` (1-10): Maximum tracks per artist
  - `transition_floor` (0.0-1.0): Minimum transition quality

**Response** (200 OK):
```json
{
  "playlist": {
    "id": "playlist_20251216_123456",
    "seed_track": {
      "track_id": "abc123",
      "title": "Zombie",
      "artist": "Fela Kuti",
      "duration": 243
    },
    "tracks": [
      {
        "index": 0,
        "track_id": "xyz789",
        "title": "Progress",
        "artist": "Tony Allen",
        "duration": 301,
        "genres": ["afrobeat", "jazz"],
        "similarity": 0.87
      },
      {
        "index": 1,
        "track_id": "uvw012",
        "title": "Water No Get Enemy",
        "artist": "Fela Kuti",
        "duration": 256,
        "genres": ["afrobeat"],
        "similarity": 0.82
      }
    ],
    "metadata": {
      "mode": "ds",
      "total_duration": 12543,
      "avg_similarity": 0.87,
      "artist_count": 24,
      "genre_count": 8
    }
  }
}
```

**Errors**:
- `400 Bad Request`: Invalid seed track ID or parameters
- `404 Not Found`: Seed track not found
- `422 Unprocessable Entity`: Validation error in request

**Use Case**: Generate single playlist from UI track selector.

### Generate Multiple Playlists

Create several playlists from multiple seed tracks.

**Endpoint**: `POST /api/playlists/generate-multiple`

**Request Body**:
```json
{
  "seed_track_ids": ["abc123", "xyz789", "uvw012"],
  "count": 30,
  "mode": "dynamic"
}
```

**Parameters**:
- `seed_track_ids` (required): Array of seed track IDs
- `count` (optional, default: 30): Tracks per playlist
- `mode` (optional, default: "ds"): Generation mode (same as single)

**Response** (200 OK):
```json
{
  "playlists": [
    {
      "seed_track": {...},
      "tracks": [...],
      "metadata": {...}
    },
    {
      "seed_track": {...},
      "tracks": [...],
      "metadata": {...}
    }
  ],
  "summary": {
    "total_playlists": 3,
    "total_tracks": 90,
    "avg_duration_hours": 5.2
  }
}
```

**Use Case**: Batch generation, multi-seed exploration.

### Export Playlist to M3U

Download playlist as M3U file.

**Endpoint**: `GET /api/playlists/{playlist_id}/export`

**Query Parameters**:
- `format` (optional, default: "m3u"): Export format
  - `m3u`: M3U playlist format
  - `json`: JSON metadata
  - `pls`: PLS format

**Response** (200 OK): M3U file download

```
#EXTM3U
#EXTINF:243,Fela Kuti - Zombie
/path/to/track1.mp3
#EXTINF:301,Tony Allen - Progress
/path/to/track2.mp3
...
```

**Use Case**: Save playlist to file, import into music player.

### Get Playlist History

List recently generated playlists.

**Endpoint**: `GET /api/playlists/history`

**Query Parameters**:
- `limit` (optional, default: 20): Maximum playlists
- `days` (optional, default: 30): Days to look back

**Response** (200 OK):
```json
{
  "playlists": [
    {
      "id": "playlist_20251216_123456",
      "seed_track": {...},
      "created_at": "2025-12-16T15:30:45Z",
      "track_count": 50,
      "duration_minutes": 208
    }
  ],
  "total": 5
}
```

**Use Case**: Recently generated list, playlist recovery.

## Similarity Endpoints

### Calculate Similarity

Compute similarity between two tracks.

**Endpoint**: `POST /api/similarity`

**Request Body**:
```json
{
  "track1_id": "abc123",
  "track2_id": "xyz789",
  "method": "hybrid"
}
```

**Parameters**:
- `track1_id` (required): First track ID
- `track2_id` (required): Second track ID
- `method` (optional, default: "hybrid"): Calculation method
  - `sonic`: Audio features only
  - `genre`: Genre only
  - `hybrid`: Combined 60% sonic + 40% genre

**Response** (200 OK):
```json
{
  "track1": {
    "track_id": "abc123",
    "title": "Zombie",
    "artist": "Fela Kuti"
  },
  "track2": {
    "track_id": "xyz789",
    "title": "Progress",
    "artist": "Tony Allen"
  },
  "similarities": {
    "sonic": 0.82,
    "genre": 0.85,
    "hybrid": 0.83
  }
}
```

**Use Case**: Debug similarity calculations, understand playlist decisions.

### Find Similar Tracks

Find N most similar tracks to a given track.

**Endpoint**: `GET /api/similarity/{track_id}/similar`

**Query Parameters**:
- `limit` (optional, default: 20): Number of similar tracks
- `method` (optional, default: "hybrid"): Similarity method

**Response** (200 OK):
```json
{
  "seed_track": {
    "track_id": "abc123",
    "title": "Zombie",
    "artist": "Fela Kuti"
  },
  "similar_tracks": [
    {
      "track_id": "xyz789",
      "title": "Progress",
      "artist": "Tony Allen",
      "similarity": 0.87
    },
    {
      "track_id": "uvw012",
      "title": "Water No Get Enemy",
      "artist": "Fela Kuti",
      "similarity": 0.85
    }
  ]
}
```

**Use Case**: "More like this" functionality, explore neighbors.

## Error Responses

All errors follow standard HTTP status codes:

### 400 Bad Request
Invalid request parameters or malformed JSON.

```json
{
  "error": "Invalid count: must be between 1 and 500",
  "status": 400
}
```

### 404 Not Found
Resource not found (track, playlist, etc.).

```json
{
  "error": "Track abc123 not found",
  "status": 404
}
```

### 422 Unprocessable Entity
Validation error in request body.

```json
{
  "error": "Validation error",
  "details": [
    {
      "field": "count",
      "message": "must be a positive integer"
    }
  ],
  "status": 422
}
```

### 500 Internal Server Error
Server error during processing.

```json
{
  "error": "Database connection failed",
  "status": 500
}
```

## Rate Limiting

No rate limiting by default. For production, add in `api/main.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/playlists/generate", dependencies=[Depends(limiter.limit("10/minute"))])
async def generate_playlist(...):
    ...
```

## CORS Configuration

Modify in `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Search for a track
response = requests.get(f"{BASE_URL}/library/search", params={"q": "fela"})
tracks = response.json()["results"]
seed_track_id = tracks[0]["track_id"]

# Generate playlist
payload = {
    "seed_track_id": seed_track_id,
    "count": 50,
    "mode": "ds"
}
response = requests.post(f"{BASE_URL}/playlists/generate", json=payload)
playlist = response.json()["playlist"]

# Print tracks
for i, track in enumerate(playlist["tracks"]):
    print(f"{i+1}. {track['artist']} - {track['title']}")
```

## JavaScript/TypeScript Client Example

```typescript
const BASE_URL = "http://localhost:8000/api";

// Search for a track
const searchResponse = await fetch(`${BASE_URL}/library/search?q=fela`);
const { results } = await searchResponse.json();
const seedTrackId = results[0].track_id;

// Generate playlist
const playlistResponse = await fetch(`${BASE_URL}/playlists/generate`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    seed_track_id: seedTrackId,
    count: 50,
    mode: "ds"
  })
});
const { playlist } = await playlistResponse.json();

// Display tracks
playlist.tracks.forEach((track, i) => {
  console.log(`${i+1}. ${track.artist} - ${track.title}`);
});
```

## Next Steps

- [Quick Start](quickstart.md) - Get started
- [Architecture](architecture.md) - Understand the system
- [Configuration](configuration.md) - Customize settings

