from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_genre_enrichment.client import OpenAIEnrichmentClient
from src.ai_genre_enrichment.discovery import ReleasePayload, compute_input_hash, discover_releases
from src.ai_genre_enrichment.models import RESPONSE_SCHEMA_VERSION, response_format_schema, validate_ai_response
from src.ai_genre_enrichment.prompt import PROMPT_VERSION, SYSTEM_INSTRUCTIONS, TAXONOMY_VERSION, build_batch_request, build_prompt
from src.ai_genre_enrichment.routing import EnrichmentLane, WebMode, route_release
from src.ai_genre_enrichment.source_extraction import fetch_bandcamp_release_tags, is_bandcamp_release_url
from src.ai_genre_enrichment.storage import SidecarStore

DEFAULT_METADATA_DB = ROOT / "data" / "metadata.db"
DEFAULT_SIDECAR_DB = ROOT / "data" / "ai_genre_enrichment.db"
DEFAULT_MODEL = "gpt-4o-mini"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "discover":
        return cmd_discover(args)
    if args.command == "run-one":
        return cmd_run_one(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "prepare-batch":
        return cmd_prepare_batch(args)
    if args.command == "collect-batch":
        print("collect-batch is reserved for a later Batch API collection pass.")
        return 0
    if args.command == "extract-tags":
        return cmd_extract_tags(args)
    if args.command == "classify-tags":
        return cmd_classify_tags(args)
    if args.command == "build-enriched":
        return cmd_build_enriched(args)
    if args.command == "show-enriched":
        return cmd_show_enriched(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "ingest-local":
        return cmd_ingest_local(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-assisted album-level genre enrichment")
    parser.add_argument("--metadata-db", type=Path, default=DEFAULT_METADATA_DB)
    parser.add_argument("--sidecar-db", type=Path, default=DEFAULT_SIDECAR_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL)

    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Discover canonical artist+album releases")
    add_release_filters(discover)
    discover.add_argument("--dry-run", action="store_true", help="Print compact request payloads")
    discover.add_argument("--web-mode", choices=[mode.value for mode in WebMode], default=WebMode.OFF.value)

    run_one = sub.add_parser("run-one", help="Run or dry-run one artist+album release")
    run_one.add_argument("--artist", required=True)
    run_one.add_argument("--album", required=True)
    run_one.add_argument("--dry-run", action="store_true")
    run_one.add_argument("--force", action="store_true")
    run_one.add_argument("--web-mode", choices=[mode.value for mode in WebMode], default=WebMode.OFF.value)
    run_one.add_argument("--max-web-enrichment", type=int)
    run_one.add_argument(
        "--source-url",
        action="append",
        dest="source_urls",
        help="Authoritative release/artist/label/Bandcamp URL to supply as evidence target; repeat for multiple URLs.",
    )
    run_one.add_argument(
        "--source-tag",
        action="append",
        dest="source_tags",
        help="Known authoritative source tag from a supplied release page; repeat for multiple tags.",
    )
    run_one.add_argument(
        "--allowed-web-domain",
        action="append",
        dest="allowed_web_domains",
        help="Limit web search to this domain; repeat for multiple domains. Defaults to broad authoritative-source search.",
    )

    run = sub.add_parser("run", help="Run enrichment over discovered releases")
    add_release_filters(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--only-unchecked", action="store_true")
    run.add_argument("--web-mode", choices=[mode.value for mode in WebMode], default=WebMode.OFF.value)
    run.add_argument("--max-web-enrichment", type=int)
    run.add_argument(
        "--source-url",
        action="append",
        dest="source_urls",
        help="Authoritative release/artist/label/Bandcamp URL to supply as evidence target; repeat for multiple URLs.",
    )
    run.add_argument(
        "--source-tag",
        action="append",
        dest="source_tags",
        help="Known authoritative source tag from a supplied release page; repeat for multiple tags.",
    )
    run.add_argument(
        "--allowed-web-domain",
        action="append",
        dest="allowed_web_domains",
        help="Limit web search to this domain; repeat for multiple domains. Defaults to broad authoritative-source search.",
    )

    batch = sub.add_parser("prepare-batch", help="Write Responses API JSONL requests for later batch processing")
    add_release_filters(batch)
    batch.add_argument("--out", type=Path)
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--only-unchecked", action="store_true")

    sub.add_parser("collect-batch", help="Stub for future Batch API result collection")
    extract = sub.add_parser("extract-tags", help="Extract deterministic tags from confirmed source URLs")
    add_release_filters(extract)
    extract.add_argument("--source-url", action="append", dest="source_urls", required=True)
    extract.add_argument("--dry-run", action="store_true")

    classify = sub.add_parser("classify-tags", help="Classify extracted source tags")
    add_release_filters(classify)
    classify.add_argument("--dry-run", action="store_true")

    build = sub.add_parser("build-enriched", help="Build enriched_genres from classified source tags")
    add_release_filters(build)
    build.add_argument("--dry-run", action="store_true")

    show = sub.add_parser("show-enriched", help="Show enriched genre signature for a release")
    add_release_filters(show)

    sub.add_parser("report", help="Summarize sidecar recommendations and run counters")

    ingest_local = sub.add_parser("ingest-local", help="Ingest genres from local metadata.db genre tables as a confirmed source")
    add_release_filters(ingest_local)
    ingest_local.add_argument("--dry-run", action="store_true")

    return parser


def add_release_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--artist")
    parser.add_argument("--album")
    parser.add_argument("--generic-only", action="store_true")
    parser.add_argument("--min-existing-specific-genres", type=int)


def cmd_discover(args: argparse.Namespace) -> int:
    releases = _discover(args)
    print(f"Discovered {len(releases)} release(s).")
    for release in releases:
        route = route_release(release, getattr(args, "web_mode", WebMode.OFF.value))
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "release_key": release.release_key,
                        "lane": route.lane.value,
                        "web_mode": route.web_mode.value,
                        "reasons": route.reasons,
                        "payload": release.to_request_payload(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(
                f"{release.release_key} lane={route.lane.value} tracks={len(release.track_titles)} "
                f"genres={sum(len(v) for v in release.existing_genres_by_source.values())}"
            )
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    releases = discover_releases(
        args.metadata_db,
        artist=args.artist,
        album=args.album,
        limit=1,
    )
    if not releases:
        print("No matching release found.")
        return 1
    return _run_releases(args, releases)


def cmd_run(args: argparse.Namespace) -> int:
    return _run_releases(args, _discover(args))


def cmd_prepare_batch(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    out = args.out or ROOT / "data" / "ai_genre_batches" / f"batch_{_timestamp()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for release in releases:
            route = route_release(release, WebMode.OFF)
            payload = _request_payload(
                release,
                route,
                source_urls=getattr(args, "source_urls", None),
                source_tags=getattr(args, "source_tags", None),
            )
            source_hash = _source_evidence_hash(payload)
            input_hash = compute_input_hash(
                release,
                PROMPT_VERSION,
                TAXONOMY_VERSION,
                web_mode=WebMode.OFF.value,
                source_evidence_hash=source_hash,
                response_schema_version=RESPONSE_SCHEMA_VERSION,
            )
            if _should_skip_cached(
                store,
                release,
                input_hash,
                args.model,
                force=args.force,
                only_unchecked=args.only_unchecked,
                web_mode=WebMode.OFF.value,
                source_evidence_hash=source_hash,
            ):
                continue
            prompt = build_prompt(payload)
            request = build_batch_request(
                custom_id=f"{release.release_key}:{input_hash[:12]}",
                model=args.model,
                prompt=prompt,
                response_format=response_format_schema(),
                instructions=SYSTEM_INSTRUCTIONS,
            )
            fh.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    print(f"Wrote {written} batch request(s) to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    report = store.report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_extract_tags(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1
    if len(releases) != 1:
        print("extract-tags requires filters that resolve exactly one release when source URLs are supplied.")
        return 2
    invalid_bandcamp_urls = [
        url for url in args.source_urls or [] if "bandcamp.com" in url.lower() and not is_bandcamp_release_url(url)
    ]
    if invalid_bandcamp_urls:
        print("extract-tags requires Bandcamp URLs to point to release album pages under /album/.")
        return 2

    for release in releases:
        urls = list(args.source_urls or [])
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "release_key": release.release_key,
                        "source_urls": urls,
                        "dry_run": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            continue
        failed = False
        for url in urls:
            page_id = store.upsert_source_page(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                source_url=url,
                source_type=_source_type_for_url(url),
                identity_status="confirmed",
                identity_confidence=1.0,
                evidence_summary="User-supplied source URL.",
            )
            try:
                tags = fetch_bandcamp_release_tags(url) if "bandcamp.com" in url.lower() else []
            except OSError as exc:
                failed = True
                store.mark_source_page_extraction_failed(page_id, str(exc))
                print(f"failed extract {release.release_key} {url}: {exc}")
                continue
            store.replace_source_tags(page_id, tags)
            print(f"extracted {release.release_key} {url} tags={len(tags)}")
        if failed:
            return 1
    return 0


def cmd_classify_tags(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1
    for release in releases:
        page_ids = _source_page_ids_for_release(store, release.release_key)
        if args.dry_run:
            print(
                json.dumps(
                    {"release_key": release.release_key, "source_page_ids": page_ids, "dry_run": True},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            continue
        for page_id in page_ids:
            store.classify_source_tags(page_id)
        print(f"classified {release.release_key} pages={len(page_ids)}")
    return 0


def cmd_build_enriched(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1
    for release in releases:
        if args.dry_run:
            print(json.dumps({"release_key": release.release_key, "dry_run": True}, ensure_ascii=False, sort_keys=True))
            continue
        store.rebuild_enriched_genres_for_release(release.release_key)
        print(f"built-enriched {release.release_key}")
    return 0


def cmd_ingest_local(args: argparse.Namespace) -> int:
    from src.genre.normalize_unified import DROP_TOKENS, META_TAGS
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    vocab = GenreVocabulary(library_db_path=args.metadata_db)
    set_vocabulary(vocab)

    noise = META_TAGS | DROP_TOKENS
    resolved_db = Path(args.metadata_db).resolve()
    uri = f"file:{resolved_db.as_posix()}?mode=ro"
    ingested = 0
    for release in releases:
        try:
            mconn = sqlite3.connect(uri, uri=True)
            mconn.row_factory = sqlite3.Row
        except Exception as exc:
            print(f"Warning: cannot open metadata DB: {exc}")
            continue
        try:
            raw_genres: list[str] = []
            for row in mconn.execute(
                "SELECT DISTINCT genre FROM artist_genres WHERE artist = ?",
                (release.artist,),
            ):
                if row["genre"]:
                    raw_genres.append(row["genre"])
            if release.album_id:
                try:
                    for row in mconn.execute(
                        "SELECT DISTINCT genre FROM album_genres WHERE album_id = ?",
                        (release.album_id,),
                    ):
                        if row["genre"]:
                            raw_genres.append(row["genre"])
                except sqlite3.OperationalError as exc:
                    print(f"Warning: could not read metadata genres for {release.release_key}: {exc}", file=sys.stderr)
                    raw_genres = []
        except sqlite3.OperationalError:
            raw_genres = []
        finally:
            mconn.close()

        filtered = [g for g in raw_genres if g.strip() and g.strip().casefold() not in noise]
        seen: set[str] = set()
        deduped: list[str] = []
        for g in filtered:
            k = g.strip().casefold()
            if k not in seen:
                seen.add(k)
                deduped.append(g)

        if not deduped:
            continue

        if args.dry_run:
            print(json.dumps(
                {
                    "release_key": release.release_key,
                    "local_genres": deduped,
                    "dry_run": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ))
            continue

        page_id = store.upsert_source_page(
            release_key=release.release_key,
            normalized_artist=release.normalized_artist,
            normalized_album=release.normalized_album,
            album_id=release.album_id,
            source_url="local://metadata.db",
            source_type="local_metadata",
            identity_status="confirmed",
            identity_confidence=1.0,
            evidence_summary="Genres from local metadata.db genre tables.",
        )
        store.replace_source_tags(page_id, deduped)
        store.classify_source_tags(page_id)
        store.rebuild_enriched_genres_for_release(release.release_key)
        ingested += 1
        print(f"ingested {release.release_key} tags={len(deduped)}")

    print(f"Ingested {ingested} release(s).")
    return 0


def cmd_show_enriched(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1
    with store.connect() as conn:
        for release in releases:
            row = conn.execute(
                """
                SELECT signature_json
                FROM enriched_genre_signatures
                WHERE release_key = ?
                """,
                (release.release_key,),
            ).fetchone()
            payload = json.loads(row["signature_json"]) if row else {"genres": [], "sources": []}
            print(json.dumps({"release_key": release.release_key, **payload}, ensure_ascii=False, sort_keys=True))
    return 0


def _discover(args: argparse.Namespace) -> list[ReleasePayload]:
    return discover_releases(
        args.metadata_db,
        limit=args.limit,
        artist=args.artist,
        album=args.album,
        generic_only=args.generic_only,
        min_existing_specific_genres=args.min_existing_specific_genres,
    )


def _run_releases(args: argparse.Namespace, releases: list[ReleasePayload]) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    called = 0
    skipped = 0
    failed = 0
    cache_hits = 0
    skipped_well_tagged = 0
    no_web_checks = 0
    authoritative_source_checks = 0
    needs_review_count = 0
    web_enrichment_used = 0
    tokens: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for release in releases:
        route = route_release(release, getattr(args, "web_mode", WebMode.OFF.value))
        effective_web_mode = _effective_web_mode(route.lane, route.web_mode)
        if route.lane == EnrichmentLane.SKIP_WELL_TAGGED:
            skipped += 1
            skipped_well_tagged += 1
            print(f"skip_well_tagged {release.release_key}: {'; '.join(route.reasons)}")
            continue
        if (
            route.lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT
            and getattr(args, "max_web_enrichment", None) is not None
            and web_enrichment_used >= args.max_web_enrichment
        ):
            skipped += 1
            needs_review_count += 1
            print(f"needs_review {release.release_key}: max web enrichment reached")
            continue

        payload = _request_payload(
            release,
            route,
            source_urls=getattr(args, "source_urls", None),
            source_tags=getattr(args, "source_tags", None),
        )
        source_hash = _source_evidence_hash(payload)
        input_hash = compute_input_hash(
            release,
            PROMPT_VERSION,
            TAXONOMY_VERSION,
            web_mode=effective_web_mode.value,
            source_evidence_hash=source_hash,
            response_schema_version=RESPONSE_SCHEMA_VERSION,
        )
        if _should_skip_cached(
            store,
            release,
            input_hash,
            args.model,
            force=getattr(args, "force", False),
            only_unchecked=getattr(args, "only_unchecked", False),
            web_mode=effective_web_mode.value,
            source_evidence_hash=source_hash,
        ):
            skipped += 1
            cache_hits += 1
            print(f"cached {release.release_key}")
            continue

        prompt = build_prompt(payload)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "release_key": release.release_key,
                        "lane": route.lane.value,
                        "web_mode": effective_web_mode.value,
                        "reasons": route.reasons,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            store.record_pending_check(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                identifiers=release.identifiers,
                input_hash=input_hash,
                prompt_version=PROMPT_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
                model=args.model,
                web_mode=effective_web_mode.value,
                source_evidence_hash=source_hash,
                response_schema_version=RESPONSE_SCHEMA_VERSION,
            )

        client = OpenAIEnrichmentClient(
            model=args.model,
            dry_run=args.dry_run,
            web_mode=effective_web_mode,
            allowed_web_domains=getattr(args, "allowed_web_domains", None),
        )
        result = client.enrich(payload, prompt, response_format_schema(), instructions=SYSTEM_INSTRUCTIONS)
        if result.status == "failed":
            failed += 1
            if not args.dry_run:
                store.record_failed_check(
                    release_key=release.release_key,
                    normalized_artist=release.normalized_artist,
                    normalized_album=release.normalized_album,
                    album_id=release.album_id,
                    identifiers=release.identifiers,
                    input_hash=input_hash,
                    prompt_version=PROMPT_VERSION,
                    taxonomy_version=TAXONOMY_VERSION,
                    model=args.model,
                    web_mode=effective_web_mode.value,
                    source_evidence_hash=source_hash,
                    response_schema_version=RESPONSE_SCHEMA_VERSION,
                    error_message=result.error_message or "unknown OpenAI error",
                )
            print(f"failed {release.release_key}: {result.error_message}")
            continue

        for key in tokens:
            tokens[key] += result.token_usage.get(key, 0)
        if args.dry_run and result.estimated_cost_usd is not None:
            print(
                "estimate "
                f"{release.release_key} "
                f"input_tokens~{result.token_usage.get('estimated_prompt_tokens', 0)} "
                f"output_tokens~{result.token_usage.get('estimated_output_tokens', 0)} "
                f"cost~${result.estimated_cost_usd:.6f}"
            )
        if not args.dry_run:
            response_json = validate_ai_response(result.response_json)
            additions = response_json.get("new_genres_to_add", [])
            store.record_complete_check(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                identifiers=release.identifiers,
                input_hash=input_hash,
                prompt_version=PROMPT_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
                model=args.model,
                web_mode=effective_web_mode.value,
                source_evidence_hash=source_hash,
                response_schema_version=RESPONSE_SCHEMA_VERSION,
                response_json=response_json,
                overall_confidence=response_json.get("release_level_confidence"),
                evidence_quality=response_json.get("evidence_quality"),
                auto_apply_eligible=bool(additions) and all(item.get("auto_apply_eligible") for item in additions),
                token_usage=result.token_usage,
                estimated_cost_usd=result.estimated_cost_usd,
            )
        called += 1
        if route.lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT:
            web_enrichment_used += 1
            authoritative_source_checks += 1
        else:
            no_web_checks += 1
        if not args.dry_run and result.response_json.get("should_escalate"):
            needs_review_count += 1
        print(f"{'dry-run' if args.dry_run else 'checked'} {route.lane.value} {release.release_key}")

    if not args.dry_run:
        store.record_run_log(
            command=args.command,
            status="failed" if failed else "complete",
            releases_seen=len(releases),
            releases_called=called,
            releases_skipped=skipped,
            releases_failed=failed,
            cache_hits=cache_hits,
            skipped_well_tagged=skipped_well_tagged,
            no_web_checks=no_web_checks,
            authoritative_source_checks=authoritative_source_checks,
            needs_review=needs_review_count,
            token_usage=tokens,
        )
    return 1 if failed else 0


def _effective_web_mode(lane: EnrichmentLane, requested_mode: WebMode | str) -> WebMode:
    mode = WebMode(requested_mode)
    if lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT and mode == WebMode.AUTO:
        return WebMode.REQUIRED
    return mode


def _should_skip_cached(
    store: SidecarStore,
    release: ReleasePayload,
    input_hash: str,
    model: str,
    *,
    force: bool,
    only_unchecked: bool,
    web_mode: str,
    source_evidence_hash: str,
) -> bool:
    if force:
        return False
    cached = store.has_complete_check(
        release.release_key,
        input_hash,
        PROMPT_VERSION,
        TAXONOMY_VERSION,
        model,
        web_mode=web_mode,
        source_evidence_hash=source_evidence_hash,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
    )
    return cached or (only_unchecked and cached)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _request_payload(
    release: ReleasePayload,
    route: object,
    source_urls: list[str] | None = None,
    source_tags: list[str] | None = None,
) -> dict[str, object]:
    payload = release.to_request_payload()
    payload["routing"] = {
        "lane": route.lane.value,
        "web_mode": route.web_mode.value,
        "reasons": route.reasons,
    }
    supplied_source_evidence = [
        {
            "source_name": "Local metadata payload",
            "source_url": None,
            "source_type": "local_payload",
            "reliability": "medium",
            "release_specific": True,
            "extracted_genres_or_styles": sorted(release.genre_counts),
            "evidence_summary": "Existing local artist, album, and track genre metadata grouped by source.",
        }
    ]
    urls = source_urls or []
    tags = _clean_source_tags(source_tags)
    if urls:
        for index, url in enumerate(urls):
            supplied_source_evidence.append(_source_url_evidence(url, source_tags=tags if index == 0 else []))
    elif tags:
        supplied_source_evidence.append(_source_tag_evidence(tags))
    payload["supplied_source_evidence"] = supplied_source_evidence
    return payload


def _source_url_evidence(url: str, source_tags: list[str] | None = None) -> dict[str, object]:
    lowered = url.lower()
    if "bandcamp.com" in lowered:
        source_type = "bandcamp_release"
        source_name = "User-supplied Bandcamp release URL"
    else:
        source_type = "official_release"
        source_name = "User-supplied authoritative source URL"
    tags = source_tags or []
    evidence_summary = (
        "User supplied this URL as an authoritative release/artist/label evidence target. "
        "Use web search/tool evidence to inspect it; do not infer genres from the URL alone."
    )
    if tags:
        evidence_summary += " User-supplied source tags from this page: " + ", ".join(tags) + "."
    return {
        "source_name": source_name,
        "source_url": url,
        "source_type": source_type,
        "reliability": "high",
        "release_specific": True,
        "extracted_genres_or_styles": tags,
        "evidence_summary": evidence_summary,
    }


def _source_tag_evidence(source_tags: list[str]) -> dict[str, object]:
    return {
        "source_name": "User-supplied authoritative source tags",
        "source_url": None,
        "source_type": "official_release",
        "reliability": "high",
        "release_specific": True,
        "extracted_genres_or_styles": source_tags,
        "evidence_summary": "User-supplied source tags from an authoritative release page: "
        + ", ".join(source_tags)
        + ".",
    }


def _clean_source_tags(source_tags: list[str] | None) -> list[str]:
    seen = set()
    cleaned = []
    for tag in source_tags or []:
        normalized = " ".join(tag.strip().split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            cleaned.append(normalized)
    return cleaned


def _source_evidence_hash(payload: dict[str, object]) -> str:
    source_payload = payload.get("supplied_source_evidence", [])
    blob = json.dumps(source_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _source_type_for_url(url: str) -> str:
    return "bandcamp_release" if "bandcamp.com" in url.lower() else "official_release"


def _source_page_ids_for_release(store: SidecarStore, release_key: str) -> list[int]:
    with store.connect() as conn:
        return [
            int(row["source_page_id"])
            for row in conn.execute(
                """
                SELECT source_page_id
                FROM ai_genre_source_pages
                WHERE release_key = ?
                ORDER BY source_page_id
                """,
                (release_key,),
            )
        ]


if __name__ == "__main__":
    raise SystemExit(main())
