from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai_genre_enrichment.client import OpenAIEnrichmentClient
from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
from src.ai_genre_enrichment.model_prior import (
    MODEL_PRIOR_INSTRUCTIONS,
    build_model_prior_payload,
    build_model_prior_prompt,
    map_model_prior_terms,
    model_prior_response_format,
    validate_model_prior_response,
)
from src.ai_genre_enrichment.normalization import (
    make_release_key,
    normalize_release_artist,
    normalize_release_name,
)


@dataclass(frozen=True)
class AdHocRelease:
    artist: str
    album: str
    year: int
    track_titles: list[str]
    existing_genres_by_source: dict[str, list[str]]
    album_id: str | None = None
    identifiers: dict[str, str] | None = None

    @property
    def normalized_artist(self) -> str:
        return normalize_release_artist(self.artist)

    @property
    def normalized_album(self) -> str:
        return normalize_release_name(self.album)

    @property
    def release_key(self) -> str:
        return make_release_key(self.artist, self.album)


ALBUMS = [
    AdHocRelease("FRITZ", "Pastel", 2021, [], {}),
    AdHocRelease("Miyauchi Yuri", "Beta", 2022, [], {}),
    AdHocRelease("Melkbelly", "Pith", 2020, [], {}),
    AdHocRelease("gena", "fluid flows & spirals glow", 2023, [], {}),
    AdHocRelease("Donna Hafford", "Porch Swing Poppy", 2008, [], {}),
]


def main() -> int:
    client = OpenAIEnrichmentClient(model="gpt-4o-mini", web_mode="off")
    vocabulary = GenreVocabulary()
    results: list[dict[str, Any]] = []
    for release in ALBUMS:
        payload = build_model_prior_payload(release)
        result = client.request_structured(
            payload=payload,
            prompt=build_model_prior_prompt(payload),
            response_format=model_prior_response_format(),
            validator=validate_model_prior_response,
            instructions=MODEL_PRIOR_INSTRUCTIONS,
            estimated_output_tokens=700,
        )
        mapped_terms = []
        if result.status == "complete":
            mapped_terms = map_model_prior_terms(result.response_json.get("genres", []), vocabulary, payload=payload)
        results.append(
            {
                "artist": release.artist,
                "album": release.album,
                "year": release.year,
                "status": result.status,
                "genres": result.response_json.get("genres", []),
                "mapped_terms": mapped_terms,
                "warnings": result.response_json.get("warnings", []),
                "token_usage": result.token_usage,
                "estimated_cost_usd": result.estimated_cost_usd,
                "error_message": result.error_message,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
