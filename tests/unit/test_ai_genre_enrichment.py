import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

import scripts.ai_genre_enrich as ai_genre_cli
from src.ai_genre_enrichment.client import OpenAIEnrichmentClient
from src.ai_genre_enrichment.discovery import (
    ReleasePayload,
    compute_input_hash,
    discover_releases,
    is_generic_only_release,
)
from src.ai_genre_enrichment.models import validate_ai_response
from src.ai_genre_enrichment.normalization import make_release_key, normalize_release_name
from src.ai_genre_enrichment.prompt import SYSTEM_INSTRUCTIONS
from src.ai_genre_enrichment.pricing import estimate_cost_usd
from src.ai_genre_enrichment.routing import EnrichmentLane, WebMode, route_release
from src.ai_genre_enrichment.storage import SidecarStore
from scripts.ai_genre_enrich import _effective_web_mode, _request_payload, main as ai_genre_main


def _metadata_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            year INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?)",
        ("t1", "The Bill Evans Trio", "Waltz for Debby", "Waltz For Debby", "a1", 1961),
    )
    conn.execute(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?)",
        ("t2", "The Bill Evans Trio", "My Foolish Heart", "Waltz For Debby", "a1", 1961),
    )
    conn.execute(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?)",
        ("t3", "Slowdive", "Alison", "Souvlaki", "a2", 1993),
    )
    conn.execute("INSERT INTO artist_genres VALUES (?, ?, ?)", ("The Bill Evans Trio", "jazz", "musicbrainz_artist"))
    conn.execute("INSERT INTO album_genres VALUES (?, ?, ?)", ("a1", "cool jazz", "discogs_release"))
    conn.execute("INSERT INTO track_genres VALUES (?, ?, ?)", ("t1", "live", "file"))
    conn.execute("INSERT INTO track_genres VALUES (?, ?, ?)", ("t3", "shoegaze", "file"))
    conn.commit()
    conn.close()
    return db_path


def test_release_key_normalization_uses_existing_artist_identity_rules():
    assert normalize_release_name("  太陽風～オーロラの神秘～  ") == "太陽風 オーロラの神秘"
    assert make_release_key("The Bill Evans Trio", "  Waltz   For Debby  ") == "bill evans::waltz for debby"


def test_input_hash_is_stable_across_ordering_noise():
    payload_a = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={"musicbrainz_release_mbid": "mbid"},
        year=2000,
        track_titles=["B", "A"],
        existing_genres_by_source={"track:file": ["rock", "shoegaze"], "artist:mb": ["indie rock"]},
        genre_counts={"rock": 1, "shoegaze": 1, "indie rock": 1},
    )
    payload_b = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={"musicbrainz_release_mbid": "mbid"},
        year=2000,
        track_titles=["A", "B"],
        existing_genres_by_source={"artist:mb": ["indie rock"], "track:file": ["shoegaze", "rock"]},
        genre_counts={"indie rock": 1, "shoegaze": 1, "rock": 1},
    )

    assert compute_input_hash(payload_a, "ai-genre-v1", "taxonomy-v1") == compute_input_hash(
        payload_b, "ai-genre-v1", "taxonomy-v1"
    )


def test_input_hash_changes_for_web_mode_and_source_evidence():
    payload = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={},
        year=None,
        track_titles=["A"],
        existing_genres_by_source={"album:discogs": ["rock"]},
        genre_counts={"rock": 1},
    )

    no_web = compute_input_hash(
        payload,
        "prompt-v2",
        "taxonomy-v1",
        web_mode="off",
        source_evidence_hash="none",
        response_schema_version="schema-v2",
    )
    with_web = compute_input_hash(
        payload,
        "prompt-v2",
        "taxonomy-v1",
        web_mode="auto",
        source_evidence_hash="source-a",
        response_schema_version="schema-v2",
    )

    assert no_web != with_web


def test_cache_skips_complete_matching_check(tmp_path: Path):
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    store.record_complete_check(
        release_key="artist::album",
        normalized_artist="artist",
        normalized_album="album",
        album_id="a1",
        identifiers={},
        input_hash="hash1",
        prompt_version="prompt-v1",
        taxonomy_version="taxonomy-v1",
        model="gpt-test",
        response_json={"canonical_artist": "Artist"},
        overall_confidence=0.8,
        evidence_quality="high",
        auto_apply_eligible=False,
        token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        estimated_cost_usd=0.0001,
    )

    assert store.has_complete_check("artist::album", "hash1", "prompt-v1", "taxonomy-v1", "gpt-test")
    assert not store.has_complete_check("artist::album", "hash1", "prompt-v1", "taxonomy-v1", "other-model")


def test_cache_identity_includes_web_mode_and_schema_version(tmp_path: Path):
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    kwargs = {
        "release_key": "artist::album",
        "normalized_artist": "artist",
        "normalized_album": "album",
        "album_id": "a1",
        "identifiers": {},
        "input_hash": "hash1",
        "prompt_version": "prompt-v2",
        "taxonomy_version": "taxonomy-v1",
        "model": "gpt-test",
        "response_json": {"canonical_artist": "Artist"},
        "overall_confidence": 0.8,
        "evidence_quality": "high",
        "auto_apply_eligible": False,
    }
    store.record_complete_check(**kwargs, web_mode="off", source_evidence_hash="none", response_schema_version="schema-v2")

    assert store.has_complete_check(
        "artist::album", "hash1", "prompt-v2", "taxonomy-v1", "gpt-test",
        web_mode="off", source_evidence_hash="none", response_schema_version="schema-v2"
    )
    assert not store.has_complete_check(
        "artist::album", "hash1", "prompt-v2", "taxonomy-v1", "gpt-test",
        web_mode="auto", source_evidence_hash="none", response_schema_version="schema-v2"
    )


def test_cache_identity_unique_rows_include_web_mode_source_hash_and_schema_version(tmp_path: Path):
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    kwargs = {
        "release_key": "artist::album",
        "normalized_artist": "artist",
        "normalized_album": "album",
        "album_id": "a1",
        "identifiers": {},
        "input_hash": "hash1",
        "prompt_version": "prompt-v2",
        "taxonomy_version": "taxonomy-v1",
        "model": "gpt-test",
        "response_json": {"canonical_artist": "Artist"},
        "overall_confidence": 0.8,
        "evidence_quality": "high",
        "auto_apply_eligible": False,
    }

    store.record_complete_check(
        **kwargs,
        web_mode="off",
        source_evidence_hash="none",
        response_schema_version="schema-v2",
    )
    store.record_complete_check(
        **kwargs,
        web_mode="required",
        source_evidence_hash="source-a",
        response_schema_version="schema-v3",
    )

    rows = sqlite3.connect(db_path).execute(
        """
        SELECT web_mode, source_evidence_hash, response_schema_version
        FROM ai_genre_release_checks
        WHERE release_key = ?
        ORDER BY web_mode
        """,
        ("artist::album",),
    ).fetchall()

    assert rows == [
        ("off", "none", "schema-v2"),
        ("required", "source-a", "schema-v3"),
    ]


def test_storage_demotes_low_confidence_auto_apply_suggestions(tmp_path: Path):
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()

    store.record_complete_check(
        release_key="artist::album",
        normalized_artist="artist",
        normalized_album="album",
        album_id="a1",
        identifiers={},
        input_hash="hash1",
        prompt_version="prompt-v1",
        taxonomy_version="taxonomy-v1",
        model="gpt-test",
        response_json={
            "should_escalate": False,
            "release_level_confidence": 0.9,
            "evidence_quality": "high",
            "source_evidence": [
                {
                    "source_name": "Official release page",
                    "source_url": "https://artist.example/releases/album",
                    "source_type": "official_release",
                    "reliability": "high",
                    "release_specific": True,
                    "extracted_genres_or_styles": ["cool jazz"],
                    "evidence_summary": "Official notes identify the release as cool jazz.",
                }
            ],
            "new_genres_to_add": [
                {
                    "genre": "ambient",
                    "confidence": 0.7,
                    "reason": "Track titles suggest an atmospheric sound.",
                    "recommendation_basis": "model_knowledge",
                    "supporting_source_indexes": [],
                    "auto_apply_eligible": True,
                },
                {
                    "genre": "cool jazz",
                    "confidence": 0.9,
                    "reason": "Official release notes support it.",
                    "recommendation_basis": "authoritative_source",
                    "supporting_source_indexes": [0],
                    "auto_apply_eligible": True,
                },
            ],
        },
        overall_confidence=0.9,
        evidence_quality="high",
        auto_apply_eligible=True,
    )

    rows = sqlite3.connect(db_path).execute(
        """
        SELECT genre, auto_apply_eligible
        FROM ai_genre_suggestions
        WHERE suggestion_type = 'add'
        ORDER BY genre
        """
    ).fetchall()
    assert rows == [("ambient", 0), ("cool jazz", 1)]


def test_response_schema_validation_rejects_invalid_enums():
    valid = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Official release page",
                "source_url": "https://artist.example/releases/album",
                "source_type": "official_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["jazz"],
                "evidence_summary": "Local metadata contains jazz.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.7,
        "evidence_quality": "medium",
        "web_search_used": False,
        "web_search_quality": "none",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "jazz",
                "confidence": 0.9,
                "reason": "Present in sources.",
                "recommendation_basis": "local_metadata",
                "supporting_source_indexes": [0],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "cool jazz",
                "confidence": 0.8,
                "reason": "Fits release.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": False,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [
            {
                "tag": "live",
                "confidence": 0.8,
                "reason": "Release recording context.",
                "recommendation_basis": "local_metadata",
                "supporting_source_indexes": [0],
                "descriptor_or_genre": "descriptor",
            }
        ],
        "review_only_suggestions": [
            {
                "tag": "rare microgenre",
                "confidence": 0.4,
                "reason": "Weak evidence.",
                "recommendation_basis": "model_knowledge",
                "supporting_source_indexes": [],
                "descriptor_or_genre": "genre",
            }
        ],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }
    assert validate_ai_response(valid)["evidence_quality"] == "medium"

    invalid = dict(valid)
    invalid["evidence_quality"] = "certain"
    with pytest.raises(ValueError):
        validate_ai_response(invalid)


def test_schema_rejects_bad_supporting_source_index():
    invalid = {
        "release_identity": {"status": "confirmed", "canonical_artist": "A", "canonical_album": "B", "notes": ""},
        "source_evidence": [],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": False,
        "web_search_quality": "none",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "ambient",
                "confidence": 0.9,
                "reason": "Source backed.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": False,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }
    with pytest.raises(ValueError):
        validate_ai_response(invalid)


def test_generic_only_detection_distinguishes_specific_genres():
    generic_payload = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id=None,
        identifiers={},
        year=None,
        track_titles=[],
        existing_genres_by_source={"artist:source": ["rock", "indie rock"], "track:file": ["live"]},
        genre_counts={"rock": 1, "indie rock": 1, "live": 1},
    )
    specific_payload = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id=None,
        identifiers={},
        year=None,
        track_titles=[],
        existing_genres_by_source={"track:file": ["shoegaze"]},
        genre_counts={"shoegaze": 1},
    )

    assert is_generic_only_release(generic_payload)
    assert not is_generic_only_release(specific_payload)


def test_metadata_quality_routing_for_well_tagged_generic_and_descriptor_only_releases():
    well_tagged = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={},
        year=2000,
        track_titles=["A", "B", "C", "D", "E"],
        existing_genres_by_source={"album:source": ["shoegaze", "dream pop", "slowcore"]},
        genre_counts={"shoegaze": 1, "dream pop": 1, "slowcore": 1},
    )
    generic = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={},
        year=2000,
        track_titles=["A", "B", "C", "D", "E"],
        existing_genres_by_source={"album:source": ["rock", "indie rock"]},
        genre_counts={"rock": 1, "indie rock": 1},
    )
    descriptor_only = ReleasePayload(
        artist="Artist",
        album="Album",
        normalized_artist="artist",
        normalized_album="album",
        release_key="artist::album",
        album_id="a1",
        identifiers={},
        year=2000,
        track_titles=["A"],
        existing_genres_by_source={"album:source": ["live", "Japanese"]},
        genre_counts={"live": 1, "japanese": 1},
    )

    assert route_release(well_tagged, WebMode.AUTO).lane == EnrichmentLane.SKIP_WELL_TAGGED
    assert route_release(generic, WebMode.AUTO).lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT
    assert route_release(generic, WebMode.OFF).lane == EnrichmentLane.NO_WEB_ADJUDICATION
    assert route_release(descriptor_only, WebMode.AUTO).lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT


def test_dry_run_client_does_not_call_openai(monkeypatch):
    client = OpenAIEnrichmentClient(model="gpt-4o-mini", dry_run=True, web_mode="required")

    def explode(*_args, **_kwargs):
        raise AssertionError("OpenAI should not be called during dry-run")

    monkeypatch.setattr(client, "_call_openai", explode)
    result = client.enrich({"artist": "Artist", "album": "Album"}, "prompt", {"type": "object"})

    assert result.status == "skipped"
    assert result.response_json["dry_run"] is True
    assert result.response_json["web_mode"] == "required"
    assert result.token_usage["estimated_prompt_chars"] > 0
    assert result.estimated_cost_usd is not None


def test_prompt_is_source_grounded_and_not_general_knowledge_only():
    assert "Use local metadata first" in SYSTEM_INSTRUCTIONS
    assert "Primary authoritative sources" in SYSTEM_INSTRUCTIONS
    assert "Never claim a source says something" in SYSTEM_INSTRUCTIONS
    assert "Do not use web search to rediscover MusicBrainz or Discogs" in SYSTEM_INSTRUCTIONS
    assert "supplied source tags" in SYSTEM_INSTRUCTIONS
    assert "Last.fm" in SYSTEM_INSTRUCTIONS
    assert "Discogs release/master styles" not in SYSTEM_INSTRUCTIONS
    assert "MusicBrainz release/artist tags" not in SYSTEM_INSTRUCTIONS
    assert "general music knowledge only" not in SYSTEM_INSTRUCTIONS


def test_web_allowlist_excludes_baseline_and_secondary_context_domains():
    from src.ai_genre_enrichment.client import DEFAULT_ALLOWED_WEB_DOMAINS

    disallowed = {
        "discogs.com",
        "musicbrainz.org",
        "last.fm",
        "allmusic.com",
        "wikipedia.org",
        "wikidata.org",
        "pitchfork.com",
        "boomkat.com",
        "bandcampdaily.com",
    }
    assert disallowed.isdisjoint(DEFAULT_ALLOWED_WEB_DOMAINS)


def test_web_search_tool_omits_domain_filter_by_default_and_accepts_override(monkeypatch):
    captured: list[dict[str, object]] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class _Responses:
        @staticmethod
        def create(**kwargs):
            captured.append(kwargs)
            raise RuntimeError("stop before network")

    class _OpenAI:
        def __init__(self):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))

    OpenAIEnrichmentClient(model="gpt-test", web_mode="auto", max_retries=0).enrich({}, "prompt", {})
    OpenAIEnrichmentClient(
        model="gpt-test",
        web_mode="auto",
        allowed_web_domains=["bandcamp.com"],
        max_retries=0,
    ).enrich({}, "prompt", {})

    assert captured[0]["tools"] == [{"type": "web_search"}]
    assert captured[1]["tools"] == [{"type": "web_search", "filters": {"allowed_domains": ["bandcamp.com"]}}]


def test_gpt_4o_mini_domain_filter_fails_fast_without_api_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = OpenAIEnrichmentClient(
        model="gpt-4o-mini",
        web_mode="required",
        allowed_web_domains=["bandcamp.com"],
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("domain filter incompatibility should be caught before API call")

    monkeypatch.setattr(client, "_call_openai", explode)
    result = client.enrich({}, "prompt", {})

    assert result.status == "failed"
    assert "does not support web_search domain filters" in (result.error_message or "")


def test_client_retries_once_after_response_validation_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    calls: list[str] = []
    invalid_response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Official Bandcamp",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["drone"],
                "evidence_summary": "Bandcamp lists drone.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "ambient",
                "confidence": 0.9,
                "reason": "Supported by Bandcamp.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }
    valid_response = dict(invalid_response)
    valid_response["source_evidence"] = [
        {
            "source_name": "Official Bandcamp",
            "source_url": "https://artist.bandcamp.com/album/release",
            "source_type": "bandcamp_release",
            "reliability": "high",
            "release_specific": True,
            "extracted_genres_or_styles": ["ambient"],
            "evidence_summary": "Bandcamp lists ambient.",
        }
    ]
    valid_response["existing_genres_to_keep"] = [
        {
            "genre": "ambient",
            "confidence": 0.9,
            "reason": "Supported by Bandcamp.",
            "recommendation_basis": "authoritative_source",
            "supporting_source_indexes": [0],
        }
    ]

    class _Response:
        def __init__(self, payload):
            self.output_text = json.dumps(payload)
            self.usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    client = OpenAIEnrichmentClient(model="gpt-test", web_mode="required", max_retries=0)

    def fake_call(_prompt, _response_format, *, instructions):
        calls.append(instructions)
        return _Response(invalid_response if len(calls) == 1 else valid_response)

    monkeypatch.setattr(client, "_call_openai", fake_call)

    result = client.enrich({}, "prompt", {})

    assert result.status == "complete", result.error_message
    assert len(calls) == 2
    assert "Previous response failed validation" in calls[1]
    assert result.response_json["existing_genres_to_keep"][0]["supporting_source_indexes"] == [0]


def test_authoritative_source_lane_uses_required_web_tool_even_when_cli_mode_is_auto():
    assert _effective_web_mode(EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT, WebMode.AUTO) == WebMode.REQUIRED
    assert _effective_web_mode(EnrichmentLane.NO_WEB_ADJUDICATION, WebMode.AUTO) == WebMode.AUTO


def test_request_payload_accepts_user_supplied_source_urls():
    release = ReleasePayload(
        artist="Brijean",
        album="Macro",
        normalized_artist="brijean",
        normalized_album="macro",
        release_key="brijean::macro",
        album_id="a1",
        identifiers={},
        year=None,
        track_titles=[],
        existing_genres_by_source={},
        genre_counts={},
    )
    route = route_release(release, WebMode.AUTO)

    payload = _request_payload(release, route, source_urls=["https://brijean.bandcamp.com/album/macro"])

    supplied = payload["supplied_source_evidence"]
    assert supplied[1]["source_type"] == "bandcamp_release"
    assert supplied[1]["source_url"] == "https://brijean.bandcamp.com/album/macro"
    assert supplied[1]["reliability"] == "high"
    assert supplied[1]["release_specific"] is True


def test_request_payload_attaches_user_supplied_source_tags_to_source_url():
    release = ReleasePayload(
        artist="Cole Pulice",
        album="Gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        release_key="cole pulice::gloam",
        album_id="a1",
        identifiers={},
        year=None,
        track_titles=[],
        existing_genres_by_source={},
        genre_counts={},
    )
    route = route_release(release, WebMode.AUTO)

    payload = _request_payload(
        release,
        route,
        source_urls=["https://colepulice.bandcamp.com/album/gloam-2"],
        source_tags=["ambient", "ambient jazz", "electroacoustic", "electronica", "fourth world"],
    )

    supplied = payload["supplied_source_evidence"]
    assert supplied[1]["source_type"] == "bandcamp_release"
    assert supplied[1]["extracted_genres_or_styles"] == [
        "ambient",
        "ambient jazz",
        "electroacoustic",
        "electronica",
        "fourth world",
    ]
    assert "User-supplied source tags" in supplied[1]["evidence_summary"]


def test_source_url_domains_are_normalized_before_validation():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Official Bandcamp Release",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["alternative rock"],
                "evidence_summary": "Bandcamp page lists alternative rock.",
            },
            {
                "source_name": "Wikipedia Article",
                "source_url": "https://en.wikipedia.org/wiki/Album",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["alternative rock"],
                "evidence_summary": "Wikipedia lists alternative rock.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "alternative rock",
                "confidence": 0.9,
                "reason": "Official release source supports it.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["source_evidence"][0]["source_type"] == "bandcamp_release"
    assert normalized["source_evidence"][1]["source_type"] == "review_context"
    assert normalized["source_evidence"][1]["reliability"] == "low"
    assert normalized["source_evidence"][1]["release_specific"] is False


def test_authoritative_recommendations_require_authoritative_supporting_source():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Wikipedia Article",
                "source_url": "https://en.wikipedia.org/wiki/Album",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["alternative rock"],
                "evidence_summary": "Wikipedia lists alternative rock.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "alternative rock",
                "confidence": 0.9,
                "reason": "Wikipedia supports it.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    addition = normalized["new_genres_to_add"][0]
    assert normalized["source_evidence"][0]["source_type"] == "review_context"
    assert addition["recommendation_basis"] == "review_context"
    assert addition["auto_apply_eligible"] is False


def test_authoritative_recommendations_cannot_mix_review_context_sources():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Official Bandcamp Release",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["indie rock"],
                "evidence_summary": "Bandcamp lists indie rock.",
            },
            {
                "source_name": "Pitchfork Review",
                "source_url": "https://pitchfork.com/reviews/albums/release/",
                "source_type": "official_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["rock"],
                "evidence_summary": "Pitchfork calls it rock.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "indie rock",
                "confidence": 0.9,
                "reason": "Bandcamp and Pitchfork support it.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
                "auto_apply_eligible": False,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    addition = normalized["new_genres_to_add"][0]
    assert normalized["source_evidence"][1]["source_type"] == "review_context"
    assert addition["recommendation_basis"] == "authoritative_source"
    assert addition["supporting_source_indexes"] == [0]


def test_allaboutjazz_is_review_context_not_official_release():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "All About Jazz",
                "source_url": "https://www.allaboutjazz.com/album/example",
                "source_type": "official_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["afrobeat"],
                "evidence_summary": "All About Jazz describes the album.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["source_evidence"][0]["source_type"] == "review_context"
    assert normalized["source_evidence"][0]["reliability"] == "low"
    assert normalized["source_evidence"][0]["release_specific"] is False


def test_amoeba_is_review_context_not_official_label():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Amoeba Music Album Description",
                "source_url": "https://www.amoeba.com/macro-cd-brijean/albums/4385001/",
                "source_type": "official_label",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["rock"],
                "evidence_summary": "Amoeba describes the album.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["source_evidence"][0]["source_type"] == "review_context"
    assert normalized["source_evidence"][0]["reliability"] == "low"
    assert normalized["source_evidence"][0]["release_specific"] is False


def test_streaming_domains_are_review_context_not_official_sources():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Gloam by Cole Pulice: Listen on Audiomack",
                "source_url": "https://audiomack.com/cole-pulice/album/gloam",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic"],
                "evidence_summary": "Audiomack lists the album as electronic.",
            },
            {
                "source_name": "Qobuz Album Page",
                "source_url": "https://www.qobuz.com/us-en/album/example/abc",
                "source_type": "official_distributor",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["folk"],
                "evidence_summary": "Qobuz lists the album as folk.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.8,
        "evidence_quality": "medium",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert [source["source_type"] for source in normalized["source_evidence"]] == [
        "review_context",
        "review_context",
    ]
    assert all(source["reliability"] == "low" for source in normalized["source_evidence"])
    assert all(source["release_specific"] is False for source in normalized["source_evidence"])


def test_review_context_source_is_always_low_reliability_context():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Amoeba Music",
                "source_url": "https://www.amoeba.com/example/albums/1/",
                "source_type": "review_context",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["dream pop"],
                "evidence_summary": "Shop copy describes the album.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.8,
        "evidence_quality": "medium",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [
            {
                "tag": "dream pop",
                "confidence": 0.8,
                "reason": "Review context only.",
                "recommendation_basis": "review_context",
                "supporting_source_indexes": [0],
                "descriptor_or_genre": "genre",
            }
        ],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": True,
    }

    normalized = validate_ai_response(response)

    assert normalized["source_evidence"][0]["reliability"] == "low"
    assert normalized["source_evidence"][0]["release_specific"] is False


def test_authoritative_keep_recommendations_cannot_cite_review_context_sources():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "The Skinny",
                "source_url": "https://www.theskinny.co.uk/music/reviews/albums/example",
                "source_type": "review_context",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic"],
                "evidence_summary": "Review describes the album.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.8,
        "evidence_quality": "medium",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "electronic",
                "confidence": 0.8,
                "reason": "Supported by review context only.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    keep = normalized["existing_genres_to_keep"][0]
    assert keep["recommendation_basis"] == "review_context"
    assert keep["supporting_source_indexes"] == [0]


def test_authoritative_keep_reason_with_baseline_source_is_normalized_to_local_metadata():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Artist Bandcamp",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["indie pop"],
                "evidence_summary": "Bandcamp lists indie pop.",
            },
            {
                "source_name": "Local metadata payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "medium",
                "release_specific": True,
                "extracted_genres_or_styles": ["indie rock"],
                "evidence_summary": "Existing local genre metadata grouped by source.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "indie rock",
                "confidence": 0.8,
                "reason": "Existing genre from Discogs release metadata.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    keep = normalized["existing_genres_to_keep"][0]
    assert keep["recommendation_basis"] == "local_metadata"
    assert keep["supporting_source_indexes"] == [1]


def test_authoritative_keep_can_infer_matching_source_tag_index():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient", "ambient jazz", "electroacoustic"],
                "evidence_summary": "Bandcamp lists ambient, ambient jazz, and electroacoustic.",
            },
            {
                "source_name": "Local metadata payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "medium",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic", "jazz"],
                "evidence_summary": "Existing local genre metadata grouped by source.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "ambient jazz",
                "confidence": 0.9,
                "reason": "Supported by Bandcamp tags.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["existing_genres_to_keep"][0]["supporting_source_indexes"] == [0]


def test_authoritative_recommendations_drop_review_context_indexes_when_authoritative_source_remains():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient"],
                "evidence_summary": "Bandcamp lists ambient.",
            },
            {
                "source_name": "Audiomack",
                "source_url": "https://audiomack.com/cole-pulice/album/gloam",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient"],
                "evidence_summary": "Streaming page also lists ambient.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "ambient",
                "confidence": 0.9,
                "reason": "Supported by Bandcamp and streaming metadata.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["source_evidence"][1]["source_type"] == "review_context"
    assert normalized["existing_genres_to_keep"][0]["supporting_source_indexes"] == [0]


def test_authoritative_addition_with_only_streaming_source_is_demoted_to_review_context():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Audiomack",
                "source_url": "https://audiomack.com/cole-pulice/album/gloam",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic"],
                "evidence_summary": "Streaming page lists electronic.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "electronic",
                "confidence": 0.9,
                "reason": "Supported by streaming metadata.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    addition = normalized["new_genres_to_add"][0]
    assert normalized["source_evidence"][0]["source_type"] == "review_context"
    assert addition["recommendation_basis"] == "review_context"
    assert addition["auto_apply_eligible"] is False


def test_dropping_streaming_index_does_not_leave_unrelated_authoritative_source():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Forced Exposure",
                "source_url": "https://www.forcedexposure.com/Artists/PULICE.COLE.html",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["experimental"],
                "evidence_summary": "Forced Exposure lists experimental.",
            },
            {
                "source_name": "Audiomack",
                "source_url": "https://audiomack.com/cole-pulice/album/gloam",
                "source_type": "official_artist",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic"],
                "evidence_summary": "Streaming page lists electronic.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "electronic",
                "confidence": 0.9,
                "reason": "Supported by Forced Exposure and Audiomack.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
            },
            {
                "genre": "experimental",
                "confidence": 0.9,
                "reason": "Supported by Forced Exposure and Audiomack.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
            },
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    electronic, experimental = normalized["existing_genres_to_keep"]
    assert normalized["source_evidence"][1]["source_type"] == "review_context"
    assert electronic["recommendation_basis"] == "review_context"
    assert electronic["supporting_source_indexes"] == [1]
    assert experimental["recommendation_basis"] == "authoritative_source"
    assert experimental["supporting_source_indexes"] == [0]


def test_authoritative_keep_without_matching_source_tag_is_downgraded():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient jazz", "electronica"],
                "evidence_summary": "Bandcamp lists ambient jazz and electronica.",
            },
            {
                "source_name": "Forced Exposure",
                "source_url": "https://www.forcedexposure.com/Artists/PULICE.COLE.html",
                "source_type": "official_label",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["experimental"],
                "evidence_summary": "Forced Exposure lists experimental.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "electronic",
                "confidence": 0.9,
                "reason": "Supported by multiple sources.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
            },
            {
                "genre": "jazz",
                "confidence": 0.9,
                "reason": "Supported by multiple sources.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0, 1],
            },
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    electronic, jazz = normalized["existing_genres_to_keep"]
    assert electronic["recommendation_basis"] == "model_knowledge"
    assert electronic["supporting_source_indexes"] == []
    assert jazz["recommendation_basis"] == "model_knowledge"
    assert jazz["supporting_source_indexes"] == []


def test_unsupported_source_based_existing_keep_uses_local_metadata_when_local_payload_matches():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronica"],
                "evidence_summary": "Bandcamp lists electronica.",
            },
            {
                "source_name": "Local metadata payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "medium",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic", "experimental", "jazz"],
                "evidence_summary": "Existing local metadata.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [
            {
                "genre": "electronic",
                "confidence": 0.9,
                "reason": "Supported by local metadata and Bandcamp tags.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
            }
        ],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    keep = normalized["existing_genres_to_keep"][0]
    assert keep["recommendation_basis"] == "local_metadata"
    assert keep["supporting_source_indexes"] == [1]


def test_pruned_authoritative_source_tag_not_in_local_payload_is_rescued_as_addition():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient", "ambient jazz", "Oakland"],
                "evidence_summary": "Bandcamp lists ambient, ambient jazz, and Oakland.",
            },
            {
                "source_name": "Local metadata payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "medium",
                "release_specific": True,
                "extracted_genres_or_styles": ["electronic", "jazz"],
                "evidence_summary": "Existing local metadata.",
            },
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [
            {
                "genre": "ambient",
                "confidence": 0.8,
                "reason": "Too broad and overlaps with electronic.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
                "prune_type": "too_broad",
                "descriptor_or_genre": "genre",
            },
            {
                "genre": "ambient jazz",
                "confidence": 0.8,
                "reason": "Overlaps with jazz.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
                "prune_type": "too_broad",
                "descriptor_or_genre": "genre",
            },
            {
                "genre": "Oakland",
                "confidence": 0.8,
                "reason": "A location; not a genre.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
                "prune_type": "descriptor",
                "descriptor_or_genre": "descriptor",
            },
        ],
        "new_genres_to_add": [
            {
                "genre": "ambient jazz",
                "confidence": 0.8,
                "reason": "Supported by Bandcamp tags.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": False,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["existing_genres_to_prune"] == []
    additions = {item["genre"]: item for item in normalized["new_genres_to_add"]}
    assert set(additions) == {"ambient", "ambient jazz"}
    assert additions["ambient"]["auto_apply_eligible"] is False
    assert additions["ambient"]["supporting_source_indexes"] == [0]


def test_authoritative_add_reason_cannot_invent_baseline_source_authority():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Artist Bandcamp",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["indie pop"],
                "evidence_summary": "Bandcamp lists indie pop.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "adequate",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "indie rock",
                "confidence": 0.8,
                "reason": "Existing genre from Discogs release metadata.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": False,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    with pytest.raises(ValueError, match="baseline source names cannot justify authoritative_source"):
        validate_ai_response(response)


def test_schema_uses_authoritative_source_basis_not_source_backed():
    from src.ai_genre_enrichment.models import RECOMMENDATION_BASES, SOURCE_TYPES

    assert "authoritative_source" in RECOMMENDATION_BASES
    assert "review_context" in RECOMMENDATION_BASES
    assert "source_backed" not in RECOMMENDATION_BASES
    assert "discogs" not in SOURCE_TYPES
    assert "musicbrainz" not in SOURCE_TYPES
    assert "lastfm" not in SOURCE_TYPES


def test_non_genre_source_tags_are_demoted_from_genre_additions():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Cole Pulice",
            "canonical_album": "Gloam",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Bandcamp",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["saxophone", "Oakland"],
                "evidence_summary": "Bandcamp tags include saxophone and Oakland.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.9,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "Oakland",
                "confidence": 0.9,
                "reason": "Supported by Bandcamp tags.",
                "recommendation_basis": "hybrid",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    normalized = validate_ai_response(response)

    assert normalized["new_genres_to_add"] == []
    assert normalized["descriptor_tags"][0]["tag"] == "Oakland"
    assert normalized["descriptor_tags"][0]["descriptor_or_genre"] == "descriptor"
    assert normalized["descriptor_tags"][0]["supporting_source_indexes"] == [0]


def test_auto_apply_requires_authoritative_or_hybrid_source(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    local_response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Local payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["ambient"],
                "evidence_summary": "Existing metadata says ambient.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.95,
        "evidence_quality": "high",
        "web_search_used": False,
        "web_search_quality": "none",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "ambient",
                "confidence": 0.95,
                "reason": "Strong local metadata.",
                "recommendation_basis": "local_metadata",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            }
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }
    authoritative_response = dict(local_response)
    authoritative_response["source_evidence"] = [
        {
            "source_name": "Official release page",
            "source_url": "https://artist.example/releases/album",
            "source_type": "official_release",
            "reliability": "high",
            "release_specific": True,
            "extracted_genres_or_styles": ["ambient"],
            "evidence_summary": "Official release notes describe ambient composition.",
        }
    ]
    authoritative_response["new_genres_to_add"] = [
        {
            "genre": "ambient",
            "confidence": 0.95,
            "reason": "Official release notes use this style.",
            "recommendation_basis": "authoritative_source",
            "supporting_source_indexes": [0],
            "auto_apply_eligible": True,
            "descriptor_or_genre": "genre",
        }
    ]

    store.record_complete_check(
        release_key="artist::local",
        normalized_artist="artist",
        normalized_album="local",
        album_id=None,
        identifiers={},
        input_hash="local",
        prompt_version="p",
        taxonomy_version="t",
        model="m",
        response_json=local_response,
        overall_confidence=0.95,
        evidence_quality="high",
        auto_apply_eligible=True,
    )
    store.record_complete_check(
        release_key="artist::official",
        normalized_artist="artist",
        normalized_album="official",
        album_id=None,
        identifiers={},
        input_hash="official",
        prompt_version="p",
        taxonomy_version="t",
        model="m",
        response_json=authoritative_response,
        overall_confidence=0.95,
        evidence_quality="high",
        auto_apply_eligible=True,
    )

    rows = sqlite3.connect(db_path).execute(
        """
        SELECT c.release_key, s.auto_apply_eligible
        FROM ai_genre_suggestions s
        JOIN ai_genre_release_checks c ON c.check_id = s.check_id
        WHERE s.suggestion_type = 'add'
        ORDER BY c.release_key
        """
    ).fetchall()
    assert rows == [("artist::local", 0), ("artist::official", 1)]


def test_auto_apply_blocks_broad_parent_genres(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Official Bandcamp",
                "source_url": "https://artist.bandcamp.com/album/release",
                "source_type": "bandcamp_release",
                "reliability": "high",
                "release_specific": True,
                "extracted_genres_or_styles": ["alternative rock", "dream pop"],
                "evidence_summary": "Bandcamp lists alternative rock and dream pop.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.95,
        "evidence_quality": "high",
        "web_search_used": True,
        "web_search_quality": "strong",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [],
        "new_genres_to_add": [
            {
                "genre": "alternative rock",
                "confidence": 0.95,
                "reason": "Official Bandcamp lists it.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            },
            {
                "genre": "dream pop",
                "confidence": 0.95,
                "reason": "Official Bandcamp lists it.",
                "recommendation_basis": "authoritative_source",
                "supporting_source_indexes": [0],
                "auto_apply_eligible": True,
                "descriptor_or_genre": "genre",
            },
        ],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    store.record_complete_check(
        release_key="artist::album",
        normalized_artist="artist",
        normalized_album="album",
        album_id=None,
        identifiers={},
        input_hash="hash",
        prompt_version="p",
        taxonomy_version="t",
        model="m",
        response_json=response,
        overall_confidence=0.95,
        evidence_quality="high",
        auto_apply_eligible=True,
    )

    rows = sqlite3.connect(db_path).execute(
        """
        SELECT genre, auto_apply_eligible
        FROM ai_genre_suggestions
        WHERE suggestion_type = 'add'
        ORDER BY genre
        """
    ).fetchall()
    assert rows == [("alternative rock", 0), ("dream pop", 1)]


def test_prune_cannot_remove_broad_parent_because_specific_child_exists():
    response = {
        "release_identity": {
            "status": "confirmed",
            "canonical_artist": "Artist",
            "canonical_album": "Album",
            "notes": "",
        },
        "source_evidence": [
            {
                "source_name": "Local payload",
                "source_url": None,
                "source_type": "local_payload",
                "reliability": "medium",
                "release_specific": True,
                "extracted_genres_or_styles": ["rock", "indie rock"],
                "evidence_summary": "Local metadata includes rock and indie rock.",
            }
        ],
        "source_conflicts": [],
        "release_level_confidence": 0.8,
        "evidence_quality": "medium",
        "web_search_used": False,
        "web_search_quality": "none",
        "model_knowledge_used": False,
        "existing_genres_to_keep": [],
        "existing_genres_to_prune": [
            {
                "genre": "rock",
                "confidence": 0.9,
                "reason": "Too broad; indie rock is more specific.",
                "recommendation_basis": "local_metadata",
                "supporting_source_indexes": [0],
                "prune_type": "too_broad",
                "descriptor_or_genre": "genre",
            }
        ],
        "new_genres_to_add": [],
        "descriptor_tags": [],
        "review_only_suggestions": [],
        "warnings": [],
        "uncertainty_notes": [],
        "should_escalate": False,
    }

    with pytest.raises(ValueError, match="cannot prune broad parent genres merely because"):
        validate_ai_response(response)


def test_cost_estimate_uses_model_token_rates():
    assert estimate_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=500) == pytest.approx(0.00045)


def test_sidecar_initializes_enriched_genre_authority_tables(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    assert "ai_genre_source_pages" in tables
    assert "ai_genre_source_tags" in tables
    assert "ai_genre_tag_classifications" in tables
    assert "enriched_genres" in tables
    assert "enriched_genre_signatures" in tables


def test_extract_bandcamp_tags_from_release_html():
    from src.ai_genre_enrichment.source_extraction import extract_bandcamp_release_tags

    html = """
    <div class="tralbumData tralbum-tags tralbum-tags-nu">
      <a class="tag" href="https://bandcamp.com/discover/ambient">ambient</a>
      <a class="tag" href="https://bandcamp.com/discover/ambient-jazz">ambient jazz</a>
      <a class="tag" href="https://bandcamp.com/discover/fourth-world">fourth world</a>
      <a class="tag" href="https://bandcamp.com/discover/oakland">Oakland</a>
    </div>
    """

    assert extract_bandcamp_release_tags(html) == ["ambient", "ambient jazz", "fourth world", "Oakland"]


def test_fetch_bandcamp_release_tags_uses_supplied_url_only():
    from src.ai_genre_enrichment.source_extraction import fetch_bandcamp_release_tags

    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return """
        <div class="tralbumData tralbum-tags tralbum-tags-nu">
          <a class="tag" href="https://bandcamp.com/discover/electronica">electronica</a>
        </div>
        """

    tags = fetch_bandcamp_release_tags("https://artist.bandcamp.com/album/release", fetch_html=fake_fetch)

    assert calls == ["https://artist.bandcamp.com/album/release"]
    assert tags == ["electronica"]


def test_source_tag_classification_preserves_specific_bandcamp_tags():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    cases = {
        "ambient": ("genre_style", "ambient"),
        "ambient jazz": ("genre_style", "ambient jazz"),
        "electroacoustic": ("genre_style", "electroacoustic"),
        "electronica": ("genre_style", "electronica"),
        "fourth world": ("genre_style", "fourth world"),
        "saxophone": ("instrument", "saxophone"),
        "Oakland": ("place", "oakland"),
        "meditation": ("mood_function", "meditation"),
    }

    for raw, expected in cases.items():
        result = classify_source_tag(raw)
        assert (result.classification, result.normalized_tag) == expected


def test_source_tag_classification_does_not_demote_niche_subgenres():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    result = classify_source_tag("fourth world")

    assert result.classification == "genre_style"
    assert result.normalized_tag == "fourth world"
    assert result.confidence >= 0.9
    assert "niche" not in result.reason.casefold()


def test_source_tag_classification_accepts_observed_niche_styles():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    expected = {
        "slowcore": "slowcore",
        "spiritual jazz": "spiritual jazz",
        "experimental jazz": "experimental jazz",
        "electronic jazz": "electronic jazz",
        "devotional": "devotional",
    }

    for raw, normalized in expected.items():
        result = classify_source_tag(raw)
        assert result.classification == "genre_style"
        assert result.normalized_tag == normalized

    soundscape = classify_source_tag("soundscape")
    assert soundscape.classification == "descriptor"
    assert soundscape.normalized_tag == "soundscapes"


def test_source_tag_classification_handles_indie_dance_batch_tags():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    genre_tags = {
        "psychedelic": "psychedelic",
        "rnb": "r&b",
    }
    non_genre_tags = {
        "dfa": "descriptor",
        "dfa records": "descriptor",
        "james murphy": "descriptor",
        "lcd soundsystem": "descriptor",
        "of montreal": "descriptor",
        "polyvinyl": "descriptor",
        "athens": "place",
        "perry": "place",
    }

    for raw, normalized in genre_tags.items():
        result = classify_source_tag(raw)
        assert result.classification == "genre_style"
        assert result.normalized_tag == normalized

    for raw, classification in non_genre_tags.items():
        result = classify_source_tag(raw)
        assert result.classification == classification


def test_source_tag_classification_handles_scaled_bandcamp_tags():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    genre_tags = [
        "alternative",
        "experimental",
        "neoclassical",
        "new age",
        "avant-folk",
        "metal",
        "rock",
        "pop",
    ]
    non_genre_tags = {
        "songs": "descriptor",
        "soundtrack": "format",
        "Montréal": "place",
        "Indianapolis": "place",
        "Washington": "place",
        "washington d.c.": "place",
        "baltimore": "place",
    }

    for tag in genre_tags:
        result = classify_source_tag(tag)
        assert (result.normalized_tag, result.classification) == (tag, "genre_style")

    for tag, classification in non_genre_tags.items():
        result = classify_source_tag(tag)
        assert result.classification == classification


def test_source_locator_schema_returns_only_candidate_sources():
    from src.ai_genre_enrichment.source_locator import source_locator_response_format

    schema = source_locator_response_format()["schema"]
    props = schema["properties"]

    assert "candidate_sources" in props
    assert "new_genres_to_add" not in props
    assert "existing_genres_to_keep" not in props


def test_source_locator_prompt_excludes_baseline_and_streaming_sources():
    from src.ai_genre_enrichment.source_locator import SOURCE_LOCATOR_INSTRUCTIONS

    text = SOURCE_LOCATOR_INSTRUCTIONS.casefold()

    assert "bandcamp" in text
    assert "musicbrainz" in text
    assert "discogs" in text
    assert "last.fm" in text
    assert "spotify" in text
    assert "qobuz" in text
    assert "do not return" in text


def test_tag_adjudicator_schema_classifies_tags_without_web_fields():
    from src.ai_genre_enrichment.tag_adjudicator import tag_adjudicator_response_format

    schema = tag_adjudicator_response_format()["schema"]
    item = schema["properties"]["tag_classifications"]["items"]

    assert "tag_classifications" in schema["properties"]
    assert "source_url" not in item["properties"]
    assert item["properties"]["classification"]["enum"] == [
        "genre_style",
        "descriptor",
        "instrument",
        "place",
        "format",
        "mood_function",
        "review_only",
    ]


def test_tag_adjudicator_instructions_preserve_narrow_source_backed_genres():
    from src.ai_genre_enrichment.tag_adjudicator import TAG_ADJUDICATOR_INSTRUCTIONS

    text = TAG_ADJUDICATOR_INSTRUCTIONS.casefold()

    assert "fourth world" in text
    assert "niche" in text
    assert "not a reason to demote" in text
    assert "do not collapse" in text


def test_build_enriched_genres_from_source_tag_classifications(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://colepulice.bandcamp.com/album/gloam-2",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(page_id, ["ambient", "ambient jazz", "saxophone"])
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("cole pulice::gloam")

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT genre, basis, status
        FROM enriched_genres
        WHERE release_key = ?
        ORDER BY genre
        """,
        ("cole pulice::gloam",),
    ).fetchall()
    signature_json = conn.execute(
        """
        SELECT signature_json
        FROM enriched_genre_signatures
        WHERE release_key = ?
        """,
        ("cole pulice::gloam",),
    ).fetchone()[0]

    assert rows == [
        ("ambient", "authoritative_source", "accepted"),
        ("ambient jazz", "authoritative_source", "accepted"),
    ]
    assert json.loads(signature_json) == {
        "genres": ["ambient", "ambient jazz"],
        "sources": [
            {
                "source_type": "bandcamp_release",
                "source_url": "https://colepulice.bandcamp.com/album/gloam-2",
            }
        ],
    }


def test_build_enriched_genres_preserves_all_specific_genre_style_tags(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://colepulice.bandcamp.com/album/gloam-2",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(
        page_id,
        [
            "ambient",
            "ambient jazz",
            "electroacoustic",
            "electronica",
            "fourth world",
            "improvisation",
            "meditation",
            "saxophone",
            "Oakland",
        ],
    )
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("cole pulice::gloam")

    rows = [
        row[0]
        for row in sqlite3.connect(db_path).execute(
            """
            SELECT genre
            FROM enriched_genres
            WHERE release_key = ?
            ORDER BY genre
            """,
            ("cole pulice::gloam",),
        )
    ]

    assert rows == [
        "ambient",
        "ambient jazz",
        "electroacoustic",
        "electronica",
        "fourth world",
    ]


def test_build_enriched_genres_does_not_write_empty_signature_for_weak_bandcamp_tags(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="allegra krieger::art of the unseen infinity machine",
        normalized_artist="allegra krieger",
        normalized_album="art of the unseen infinity machine",
        album_id="a1",
        source_url="https://allegrakrieger.bandcamp.com/album/art-of-the-unseen-infinity-machine",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(page_id, ["new york", "songs", "sharp"])
    store.classify_source_tags(page_id)

    store.rebuild_enriched_genres_for_release("allegra krieger::art of the unseen infinity machine")

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM enriched_genres").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM enriched_genre_signatures").fetchone()[0] == 0


def test_report_summarizes_review_only_source_tag_gaps(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_a = store.upsert_source_page(
        release_key="patricia brennan::maquishti",
        normalized_artist="patricia brennan",
        normalized_album="maquishti",
        album_id="a1",
        source_url="https://patriciabrennan.bandcamp.com/album/maquishti",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    page_b = store.upsert_source_page(
        release_key="unwound::6 30 1999 reykjavik iceland limited edition",
        normalized_artist="unwound",
        normalized_album="6 30 1999 reykjavik iceland limited edition",
        album_id="a2",
        source_url="https://unwound.bandcamp.com/album/6-30-1999-reykjavik-iceland",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(page_a, ["creative-music", "jazz-vibraphone", "unknown style"])
    store.replace_source_tags(page_b, ["unknown style", "math-rock-ish"])
    store.classify_source_tags(page_a)
    store.classify_source_tags(page_b)

    report = store.report()

    assert report["classification_counts"]["genre_style"] == 1
    assert report["classification_counts"]["review_only"] == 3
    assert report["review_only_tag_gaps"] == [
        {
            "tag": "unknown style",
            "count": 2,
            "source_types": ["bandcamp_release"],
            "example_releases": [
                "patricia brennan::maquishti",
                "unwound::6 30 1999 reykjavik iceland limited edition",
            ],
        },
        {
            "tag": "math-rock-ish",
            "count": 1,
            "source_types": ["bandcamp_release"],
            "example_releases": ["unwound::6 30 1999 reykjavik iceland limited edition"],
        },
    ]


def test_classify_tags_refreshes_canonical_normalized_tag(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="feeble little horse::girl with fish",
        normalized_artist="feeble little horse",
        normalized_album="girl with fish",
        album_id="a1",
        source_url="https://feeblelittlehorse.bandcamp.com/album/girl-with-fish",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(page_id, ["post punk"])
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ai_genre_source_tags
            SET normalized_tag = 'post punk'
            WHERE source_page_id = ?
            """,
            (page_id,),
        )

    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("feeble little horse::girl with fish")

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT normalized_tag FROM ai_genre_source_tags").fetchone()[0] == "post-punk"
    assert conn.execute("SELECT genre FROM enriched_genres").fetchone()[0] == "post-punk"


def test_classify_tags_merges_duplicate_canonicalized_tags(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="sonic youth::live in dallas 2006",
        normalized_artist="sonic youth",
        normalized_album="live in dallas 2006",
        album_id="a1",
        source_url="https://sonicyouth.bandcamp.com/album/live-in-dallas-2006",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO ai_genre_source_tags (
                source_page_id, raw_tag, normalized_tag, tag_position, extracted_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (page_id, "experiemental", "experiemental", 0, "2026-01-01T00:00:00Z"),
                (page_id, "experimental", "experimental", 1, "2026-01-01T00:00:00Z"),
            ],
        )

    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("sonic youth::live in dallas 2006")

    conn = sqlite3.connect(db_path)
    tags = conn.execute(
        """
        SELECT normalized_tag
        FROM ai_genre_source_tags
        ORDER BY normalized_tag
        """
    ).fetchall()
    genres = conn.execute("SELECT genre FROM enriched_genres").fetchall()

    assert tags == [("experimental",)]
    assert genres == [("experimental",)]


def test_build_enriched_genres_ignores_non_authoritative_or_ambiguous_source_pages(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    review_page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://example.com/review/gloam",
        source_type="review_context",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Review context, not authority.",
    )
    ambiguous_page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://colepulice.bandcamp.com/album/gloam-2",
        source_type="bandcamp_release",
        identity_status="ambiguous",
        identity_confidence=0.4,
        evidence_summary="Ambiguous identity.",
    )
    store.replace_source_tags(review_page_id, ["ambient"])
    store.replace_source_tags(ambiguous_page_id, ["ambient jazz"])
    store.classify_source_tags(review_page_id)
    store.classify_source_tags(ambiguous_page_id)

    store.rebuild_enriched_genres_for_release("cole pulice::gloam")

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM enriched_genres").fetchone()[0] == 0
    signature_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM enriched_genre_signatures
        WHERE release_key = ?
        """,
        ("cole pulice::gloam",),
    ).fetchone()[0]
    assert signature_count == 0


@pytest.mark.parametrize(
    ("artist", "album", "url", "source_tags", "expected_genres", "excluded_tags"),
    [
        (
            "Ada Lea",
            "one hand on the steering wheel the other sewing a garden",
            "https://adaleamusic.bandcamp.com/album/one-hand-on-the-steering-wheel-the-other-sewing-a-garden",
            ["alternative rock", "indie rock", "folk-pop", "montreal"],
            ["alternative rock", "folk-pop", "indie rock"],
            ["montreal"],
        ),
        (
            "Allegra Krieger",
            "Art of the Unseen Infinity Machine",
            "https://allegrakrieger.bandcamp.com/album/art-of-the-unseen-infinity-machine",
            ["singer-songwriter", "indie folk", "folk", "new york"],
            ["folk", "indie folk", "singer-songwriter"],
            ["new york"],
        ),
        (
            "Amor de Dias",
            "The House in the Sea",
            "https://amordedias.bandcamp.com/album/the-house-at-sea",
            ["indie pop", "chamber pop", "bossa nova", "london"],
            ["bossa nova", "chamber pop", "indie pop"],
            ["london"],
        ),
        (
            "Art Feynman",
            "Half Price at 3:30",
            "https://artfeynman.bandcamp.com/album/half-price-at-3-30",
            ["art pop", "krautrock", "post-punk", "california"],
            ["art pop", "krautrock", "post-punk"],
            ["california"],
        ),
        (
            "Bachelor",
            "Doomin' Sun",
            "https://bachelortheband.bandcamp.com/album/doomin-sun",
            ["indie rock", "dream pop", "slacker rock", "demo"],
            ["dream pop", "indie rock", "slacker rock"],
            ["demo"],
        ),
        (
            "Aunt Katrina",
            "Hot",
            "https://auntkatrina.bandcamp.com/album/hot",
            ["power pop", "indie rock", "garage rock", "los angeles"],
            ["garage rock", "indie rock", "power pop"],
            ["los angeles"],
        ),
        (
            "ari solus",
            "i've hurt people (compilation)",
            "https://arisolus.bandcamp.com/album/ive-hurt-people",
            ["bedroom pop", "emo", "lo-fi", "compilation"],
            ["bedroom pop", "emo", "lo-fi"],
            ["compilation"],
        ),
        (
            "airport people",
            "ednes",
            "https://airportpeople.bandcamp.com/album/ednes",
            ["ambient", "downtempo", "electronica", "instrumental"],
            ["ambient", "downtempo", "electronica"],
            ["instrumental"],
        ),
        (
            "Candy Claws",
            "Ceres & Calypso in the Deep Time (10th Anniversary)",
            "https://candyclaws.bandcamp.com/album/ceres-calypso-in-the-deep-time",
            ["pop", "dream pop", "indie rock", "shoegaze", "new york"],
            ["dream pop", "indie rock", "pop", "shoegaze"],
            ["new york"],
        ),
        (
            "Cap'n Jazz",
            "Analphabetapolothology",
            "https://capnjazz.bandcamp.com/album/analphabetapolothology",
            ["alternative", "indie", "punk", "american football", "chicago"],
            ["alternative", "indie", "punk"],
            ["american football", "chicago"],
        ),
        (
            "Caribou",
            "Andorra",
            "https://caribouband.bandcamp.com/album/andorra",
            ["electronic", "london"],
            ["electronic"],
            ["london"],
        ),
        (
            "Cantoma",
            "out of town",
            "https://cantoma.bandcamp.com/album/out-of-town",
            ["electronic", "balearic", "house", "london"],
            ["balearic", "electronic", "house"],
            ["london"],
        ),
        (
            "Fabiano do Nascimento & Sam Gendel",
            "The Room",
            "https://samgendel.bandcamp.com/album/the-room",
            ["7 string guitar", "brazilian", "jazz", "acoustic", "experimental", "folk", "world", "los angeles"],
            ["experimental", "folk", "jazz", "world"],
            ["7 string guitar", "brazilian", "acoustic", "los angeles"],
        ),
        (
            "Fabulous Diamonds",
            "Commercial Music",
            "https://fabulousdiamonds.bandcamp.com/album/commercial-music",
            ["alternative", "ambient", "drone", "experimental", "krautrock", "melbourne"],
            ["alternative", "ambient", "drone", "experimental", "krautrock"],
            ["melbourne"],
        ),
        (
            "Far Caspian",
            "Autofiction",
            "https://farcaspian.bandcamp.com/album/autofiction",
            ["alternative", "dream pop", "indie", "lo-fi bedroom", "lo-fi bedroom pop", "leeds"],
            ["alternative", "dream pop", "indie", "lo-fi bedroom pop"],
            ["lo-fi bedroom", "leeds"],
        ),
        (
            "Faye Webster",
            "I Know I'm Funny haha",
            "https://fayewebster.bandcamp.com/album/i-know-im-funny-haha",
            ["alternative", "alt-country", "folk", "indie", "atlanta"],
            ["alt-country", "alternative", "folk", "indie"],
            ["atlanta"],
        ),
        (
            "feeble little horse",
            "Girl with Fish",
            "https://feeblelittlehorse.bandcamp.com/album/girl-with-fish",
            ["alternative", "fuzz", "indie", "post punk", "shoegaze", "noise-pop", "pittsburgh"],
            ["alternative", "indie", "noise-pop", "post-punk", "shoegaze"],
            ["fuzz", "pittsburgh"],
        ),
        (
            "Felbm",
            "Elements of Nature",
            "https://felbm.bandcamp.com/album/elements-of-nature",
            ["jazz", "soundway records", "acoustic", "alternative", "ambient", "library music", "minimal", "utrecht"],
            ["alternative", "ambient", "jazz", "library music", "minimal"],
            ["soundway records", "acoustic", "utrecht"],
        ),
        (
            "Felt",
            "Forever Breathes the Lonely Word: Remastered Edition",
            "https://feltband.bandcamp.com/album/forever-breathes-the-lonely-word",
            ["alternative", "indie", "indie pop", "jangle pop", "united kingdom"],
            ["alternative", "indie", "indie pop", "jangle pop"],
            ["united kingdom"],
        ),
        (
            "Patricia Brennan",
            "Maquishti",
            "https://patriciabrennan.bandcamp.com/album/maquishti",
            [
                "jazz",
                "avant garde",
                "brooklyn",
                "creative-music",
                "electronic-music",
                "experimental",
                "free improvisation",
                "jazz-and-improvised-music",
                "jazz-vibraphone",
                "marimba",
                "mexico",
                "percussion",
                "vibes",
            ],
            [
                "avant-garde",
                "creative music",
                "electronic",
                "experimental",
                "free improvisation",
                "jazz",
                "jazz and improvised music",
            ],
            ["brooklyn", "jazz vibraphone", "marimba", "mexico", "percussion", "vibes"],
        ),
        (
            "Patricia Wolf",
            "See-Through",
            "https://patriciawolf.bandcamp.com/album/see-through",
            [
                "ambient",
                "portland",
                "balearic",
                "dream pop",
                "drone",
                "electronic",
                "experimental",
                "field recordings",
                "new age",
                "soundscapes",
            ],
            [
                "ambient",
                "balearic",
                "dream pop",
                "drone",
                "electronic",
                "experimental",
                "field recordings",
                "new age",
            ],
            ["portland", "soundscapes"],
        ),
        (
            "Florian T M Zeisig",
            "Planet Inc",
            "https://stroomtv.bandcamp.com/album/planet-inc",
            [
                "ambient",
                "dub",
                "dub techno",
                "electronic",
                "experimental pop",
                "folk",
                "new wave",
                "space ambient",
                "space music",
                "belgium",
            ],
            [
                "ambient",
                "dub",
                "dub techno",
                "electronic",
                "experimental pop",
                "folk",
                "new wave",
                "space ambient",
                "space music",
            ],
            ["belgium"],
        ),
        (
            "MF Doom",
            "Operation: Doomsday",
            "https://mfdoom.bandcamp.com/album/operation-doomsday",
            ["hip-hop/rap", "united states"],
            ["hip hop"],
            ["united states"],
        ),
        (
            "Michael O'Mahony",
            "Talkbox",
            "https://michaelomahony.bandcamp.com/album/talkbox",
            ["electronic", "disco", "italo", "pop", "synthpop", "united kingdom"],
            ["disco", "electronic", "italo", "pop", "synthpop"],
            ["united kingdom"],
        ),
        (
            "Mount Kimbie",
            "The Sunset Violent",
            "https://mountkimbie.bandcamp.com/album/the-sunset-violent",
            ["alternative", "electronic", "drone", "electronica", "idm", "uk dubstep", "london"],
            ["alternative", "drone", "electronic", "electronica", "idm", "uk dubstep"],
            ["london"],
        ),
        (
            "Sonic Youth",
            "Live In Dallas 2006",
            "https://sonicyouth.bandcamp.com/album/live-in-dallas-2006",
            ["rock", "alternative rock", "experiemental", "experimental", "indie rock", "noise rock", "new york"],
            ["alternative rock", "experimental", "indie rock", "noise rock", "rock"],
            ["new york"],
        ),
        (
            "Thundercat",
            "It Is What It Is",
            "https://thundercat.bandcamp.com/album/it-is-what-it-is",
            ["electronic", "r&b", "soul", "bass", "beats", "cosmic", "los angeles"],
            ["electronic", "r&b", "soul"],
            ["bass", "beats", "cosmic", "los angeles"],
        ),
        (
            "Unwound",
            "6/30/1999: Reykjavik, Iceland (Limited Edition)",
            "https://unwound.bandcamp.com/album/6-30-1999-reykjavik-iceland",
            ["rock", "indie rock", "math rock", "noise rock", "post-hardcore", "post-rock", "washington"],
            ["indie rock", "math rock", "noise rock", "post-hardcore", "post-rock", "rock"],
            ["washington"],
        ),
    ],
)
def test_candidate_artist_source_tags_build_expected_enriched_genres(
    tmp_path: Path,
    artist: str,
    album: str,
    url: str,
    source_tags: list[str],
    expected_genres: list[str],
    excluded_tags: list[str],
):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    release_key = make_release_key(artist, album)

    page_id = store.upsert_source_page(
        release_key=release_key,
        normalized_artist=normalize_release_name(artist),
        normalized_album=normalize_release_name(album),
        album_id=None,
        source_url=url,
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Fixture Bandcamp release page.",
    )
    store.replace_source_tags(page_id, source_tags)
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release(release_key)

    conn = sqlite3.connect(db_path)
    genres = [
        row[0]
        for row in conn.execute(
            """
            SELECT genre
            FROM enriched_genres
            WHERE release_key = ?
            ORDER BY genre
            """,
            (release_key,),
        )
    ]
    classifications = {
        row[0]: row[1]
        for row in conn.execute(
            """
            SELECT t.normalized_tag, c.classification
            FROM ai_genre_source_tags t
            JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
            ORDER BY t.normalized_tag
            """
        )
    }

    assert genres == sorted(expected_genres)
    for tag in excluded_tags:
        assert tag not in genres
        assert classifications[tag] != "genre_style"


def test_discovery_uses_read_only_metadata_db_and_does_not_create_side_effect_tables(tmp_path: Path):
    db_path = _metadata_db(tmp_path)
    before_tables = sqlite3.connect(db_path).execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    releases = discover_releases(db_path, limit=10)

    after_tables = sqlite3.connect(db_path).execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert before_tables == after_tables
    assert [release.release_key for release in releases][:2] == ["bill evans::waltz for debby", "slowdive::souvlaki"]
    bill = releases[0]
    assert bill.album_id == "a1"
    assert bill.track_titles == ["Waltz for Debby", "My Foolish Heart"]
    assert bill.existing_genres_by_source["album:discogs_release"] == ["cool jazz"]


def test_cli_dry_run_does_not_write_sidecar_run_log(tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    result = ai_genre_main(
        [
            "--metadata-db",
            str(metadata_db),
            "--sidecar-db",
            str(sidecar_db),
            "run",
            "--limit",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    conn = sqlite3.connect(sidecar_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_run_log").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_release_checks").fetchone()[0] == 0


def test_cli_extract_tags_dry_run_does_not_write_sidecar(tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    result = ai_genre_main(
        [
            "--metadata-db",
            str(metadata_db),
            "--sidecar-db",
            str(sidecar_db),
            "extract-tags",
            "--artist",
            "The Bill Evans Trio",
            "--album",
            "Waltz For Debby",
            "--source-url",
            "https://example.bandcamp.com/album/waltz",
            "--dry-run",
        ]
    )

    assert result == 0
    conn = sqlite3.connect(sidecar_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_source_pages").fetchone()[0] == 0


def test_cli_extract_tags_rejects_single_source_url_for_multiple_releases(tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    result = ai_genre_main(
        [
            "--metadata-db",
            str(metadata_db),
            "--sidecar-db",
            str(sidecar_db),
            "extract-tags",
            "--limit",
            "2",
            "--source-url",
            "https://example.bandcamp.com/album/waltz",
            "--dry-run",
        ]
    )

    assert result == 2
    conn = sqlite3.connect(sidecar_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_source_pages").fetchone()[0] == 0


def test_cli_extract_tags_rejects_bandcamp_artist_page_as_release_source(tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    result = ai_genre_main(
        [
            "--metadata-db",
            str(metadata_db),
            "--sidecar-db",
            str(sidecar_db),
            "extract-tags",
            "--artist",
            "The Bill Evans Trio",
            "--album",
            "Waltz For Debby",
            "--source-url",
            "https://artist.bandcamp.com/",
            "--dry-run",
        ]
    )

    assert result == 2
    conn = sqlite3.connect(sidecar_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_source_pages").fetchone()[0] == 0


def test_cli_extract_tags_reports_fetch_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    def failing_fetch(_url: str) -> list[str]:
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(ai_genre_cli, "fetch_bandcamp_release_tags", failing_fetch)

    result = ai_genre_main(
        [
            "--metadata-db",
            str(metadata_db),
            "--sidecar-db",
            str(sidecar_db),
            "extract-tags",
            "--artist",
            "The Bill Evans Trio",
            "--album",
            "Waltz For Debby",
            "--source-url",
            "https://artist.bandcamp.com/album/missing",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "failed extract" in output
    assert "HTTP Error 404: Not Found" in output
    assert "Traceback" not in output
    conn = sqlite3.connect(sidecar_db)
    source_page = conn.execute(
        """
        SELECT source_type, identity_status, extraction_status, evidence_summary
        FROM ai_genre_source_pages
        """
    ).fetchone()
    assert source_page == (
        "bandcamp_release",
        "confirmed",
        "failed",
        "User-supplied source URL. Extraction failed: HTTP Error 404: Not Found",
    )
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_source_tags").fetchone()[0] == 0


def test_ai_genre_docs_describe_enriched_genres_authority():
    text = Path("docs/AI_GENRE_ENRICHMENT.md").read_text(encoding="utf-8")

    assert "enriched_genres" in text
    assert "deterministic Bandcamp" in text
    assert "does not modify artist_genres, album_genres, or track_genres" in text
    assert "source discovery" in text
    assert "tag adjudication" in text
    assert "Niche subgenres are not automatically review-only" in text
    assert "does not write empty enriched genre signatures" in text


def test_no_direct_bandcamp_scraping_provider_is_added():
    assert not Path("src/ai_genre_enrichment/bandcamp.py").exists()


def test_classify_source_tag_uses_vocabulary_tiers(tmp_path: Path) -> None:
    """Tags recognized by the engine vocabulary (Tier 2) should classify as genre_style."""
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    # "psychedelic rock" is a SYNONYM_MAP target in normalize_unified.py
    # It should be recognized even if not in the curated YAML Tier 1
    result = classify_source_tag("psychedelic rock")
    # After vocabulary integration, this should NOT be review_only
    # (it may be genre_style at 0.85 from Tier 2, or 0.95 if already in Tier 1)
    assert result.classification == "genre_style"


def test_ingest_local_creates_source_pages_and_enriched_genres(tmp_path: Path) -> None:
    # Create a minimal metadata DB with genres for Slowdive
    metadata_db = tmp_path / "metadata.db"
    conn = sqlite3.connect(metadata_db)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            year INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?)", ("t1", "Slowdive", "Alison", "Souvlaki", "a2", 1993))
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'shoegaze', 'musicbrainz_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'dream pop', 'musicbrainz_artist')")
    conn.commit()
    conn.close()

    sidecar_db = tmp_path / "ai_genre_enriched_test.db"

    result = ai_genre_main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "ingest-local",
        "--artist", "Slowdive",
        "--album", "Souvlaki",
    ])
    assert result == 0

    store = SidecarStore(sidecar_db)
    with store.connect() as conn:
        pages = list(conn.execute(
            "SELECT * FROM ai_genre_source_pages WHERE release_key LIKE '%slowdive%'"
        ))
        assert len(pages) >= 1
        assert pages[0]["source_type"] == "local_metadata"
        assert pages[0]["source_url"] == "local://metadata.db"

        enriched = list(conn.execute(
            "SELECT * FROM enriched_genres WHERE release_key LIKE '%slowdive%'"
        ))
        assert len(enriched) >= 1
        genres = {row["genre"] for row in enriched}
        assert "shoegaze" in genres


def test_extract_lastfm_tags_from_metadata(tmp_path: Path) -> None:
    from src.ai_genre_enrichment.source_extraction import extract_lastfm_tags_from_metadata

    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, album TEXT, album_id TEXT, year INTEGER)")
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO tracks VALUES ('t1', 'Slowdive', 'Alison', 'Souvlaki', 'a1', 1993)")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'shoegaze', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'dream pop', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'seen live', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'rock', 'musicbrainz_artist')")
    conn.execute("INSERT INTO album_genres VALUES ('a1', 'indie', 'lastfm_album')")
    conn.commit()
    conn.close()

    tags = extract_lastfm_tags_from_metadata(
        artist="Slowdive",
        album_id="a1",
        metadata_db_path=db_path,
    )
    # Should only return lastfm-sourced tags, not musicbrainz
    assert "shoegaze" in tags
    assert "dream pop" in tags
    assert "indie" in tags
    assert "rock" not in tags
    # Meta-tags should be pre-filtered
    assert "seen live" not in tags


def test_extract_lastfm_command(tmp_path: Path) -> None:
    metadata_db = tmp_path / "metadata.db"
    conn = sqlite3.connect(metadata_db)
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, album TEXT, album_id TEXT, year INTEGER)")
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO tracks VALUES ('t1', 'Slowdive', 'Alison', 'Souvlaki', 'a1', 1993)")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'shoegaze', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'dream pop', 'lastfm_artist')")
    conn.commit()
    conn.close()

    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    result = ai_genre_main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "extract-lastfm",
        "--artist", "Slowdive",
        "--album", "Souvlaki",
    ])
    assert result == 0

    store = SidecarStore(sidecar_db)
    with store.connect() as conn:
        pages = list(conn.execute(
            "SELECT * FROM ai_genre_source_pages WHERE source_type = 'lastfm_tags'"
        ))
        assert len(pages) == 1
        tags = list(conn.execute(
            "SELECT normalized_tag FROM ai_genre_source_tags WHERE source_page_id = ?",
            (pages[0]["source_page_id"],),
        ))
        tag_set = {row["normalized_tag"] for row in tags}
        assert "shoegaze" in tag_set
        assert "dream pop" in tag_set


def test_review_queue_returns_unreviewed_tags(tmp_path: Path) -> None:
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    store = SidecarStore(sidecar_db)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="a1",
        source_url="https://slowdive.bandcamp.com/album/souvlaki",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="Test source.",
    )
    store.replace_source_tags(page_id, ["shoegaze", "noise pop", "xyzzy"])
    store.classify_source_tags(page_id)

    queue = store.get_review_queue()
    # "xyzzy" should be review_only
    review_tags = {item["normalized_tag"] for item in queue}
    assert "xyzzy" in review_tags
    # Tags already classified as genre_style at high confidence should NOT be in the queue
    assert "shoegaze" not in review_tags


def test_record_review_decision_removes_from_queue(tmp_path: Path) -> None:
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    store = SidecarStore(sidecar_db)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="a1",
        source_url="https://slowdive.bandcamp.com/album/souvlaki",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="Test source.",
    )
    store.replace_source_tags(page_id, ["xyzzy"])
    store.classify_source_tags(page_id)

    queue = store.get_review_queue()
    assert len(queue) == 1
    item = queue[0]

    store.record_review_decision(
        source_tag_id=item["source_tag_id"],
        release_key="slowdive::souvlaki",
        raw_tag="xyzzy",
        normalized_tag="xyzzy",
        original_classification="review_only",
        reviewed_classification="rejected",
    )

    queue_after = store.get_review_queue()
    assert len(queue_after) == 0
