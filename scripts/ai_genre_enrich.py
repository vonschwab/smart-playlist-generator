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

from src.ai_genre_enrichment.discovery import ReleasePayload, compute_input_hash, discover_releases
from src.ai_genre_enrichment.provider import (
    create_enrichment_client,
    get_enrichment_provider,
    resolve_enrichment_model,
)
from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, collect_hybrid_evidence, fuse_hybrid_evidence
from src.ai_genre_enrichment.layered_assignment import build_layered_release_diagnostics, materialize_layered_assignments
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.model_prior import (
    MODEL_PRIOR_INSTRUCTIONS,
    MODEL_PRIOR_PROMPT_VERSION,
    MODEL_PRIOR_SCHEMA_VERSION,
    MODEL_PRIOR_TAXONOMY_VERSION,
    build_model_prior_payload,
    build_model_prior_prompt,
    map_model_prior_terms,
    model_prior_response_format,
    stable_input_hash,
    validate_model_prior_response,
)
from src.ai_genre_enrichment.models import RESPONSE_SCHEMA_VERSION, response_format_schema, validate_ai_response
from src.ai_genre_enrichment.policy import STABILIZED_POLICY_VERSION
from src.ai_genre_enrichment.prompt import PROMPT_VERSION, SYSTEM_INSTRUCTIONS, TAXONOMY_VERSION, build_batch_request, build_prompt
from src.ai_genre_enrichment.routing import EnrichmentLane, RouteDecision, WebMode, route_release
from src.ai_genre_enrichment.source_extraction import fetch_bandcamp_release_tags, is_bandcamp_release_url
from src.ai_genre_enrichment.storage import SidecarStore

DEFAULT_METADATA_DB = ROOT / "data" / "metadata.db"
DEFAULT_SIDECAR_DB = ROOT / "data" / "ai_genre_enrichment.db"
DEFAULT_LAYERED_FIXTURES = ROOT / "data" / "layered_genre_smoke_fixtures.yaml"
DEFAULT_MODEL = "gpt-4o-mini"


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; release keys carry non-Latin text
    # (e.g. Japanese city-pop titles). Force UTF-8 so progress prints never
    # crash the run, replacing any truly unencodable char rather than raising.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    # Resolve --model once: explicit flag wins, else the active provider's
    # default. Downstream store records and cache lookups need a concrete name.
    if hasattr(args, "model") and args.model is None:
        args.model = resolve_enrichment_model(None)
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
    if args.command == "extract-lastfm":
        return cmd_extract_lastfm(args)
    if args.command == "extract-bandcamp":
        return cmd_extract_bandcamp(args)
    if args.command == "review":
        return cmd_review(args)
    if args.command == "review-escalated":
        return cmd_review_escalated(args)
    if args.command == "graduate-reviewed":
        return cmd_graduate_reviewed(args)
    if args.command == "graduate-ai":
        return cmd_graduate_ai(args)
    if args.command == "rebuild-artifacts":
        return cmd_rebuild_artifacts(args)
    if args.command == "model-prior-one":
        return cmd_model_prior_one(args)
    if args.command == "model-prior":
        return cmd_model_prior(args)
    if args.command == "model-prior-report":
        return cmd_model_prior_report(args)
    if args.command == "hybrid-enrich-one":
        return cmd_hybrid_enrich_one(args)
    if args.command == "graph-init":
        return cmd_graph_init(args)
    if args.command == "graph-report":
        return cmd_graph_report(args)
    if args.command == "graph-build-assignments":
        return cmd_graph_build_assignments(args)
    if args.command == "graph-show-release":
        return cmd_graph_show_release(args)
    if args.command == "graph-fixture-report":
        return cmd_graph_fixture_report(args)
    if args.command == "graph-propose-growth":
        return cmd_graph_propose_growth(args)
    if args.command == "graph-ingest-growth":
        return cmd_graph_ingest_growth(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-assisted album-level genre enrichment")
    parser.add_argument("--metadata-db", type=Path, default=DEFAULT_METADATA_DB)
    parser.add_argument("--sidecar-db", type=Path, default=DEFAULT_SIDECAR_DB)
    parser.add_argument("--model", default=None)

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
    classify.add_argument("--adjudicate", action="store_true", help="Send unknown tags to AI for adjudication")
    classify.add_argument("--model", default=None)

    build = sub.add_parser("build-enriched", help="Build enriched_genres from classified source tags")
    add_release_filters(build)
    build.add_argument("--dry-run", action="store_true")

    show = sub.add_parser("show-enriched", help="Show enriched genre signature for a release")
    add_release_filters(show)

    sub.add_parser("report", help="Summarize sidecar recommendations and run counters")

    ingest_local = sub.add_parser("ingest-local", help="Ingest genres from local metadata.db genre tables as a confirmed source")
    add_release_filters(ingest_local)
    ingest_local.add_argument("--dry-run", action="store_true")
    ingest_local.add_argument("--adjudicate", action="store_true", help="Send unknown tags to AI for adjudication")
    ingest_local.add_argument(
        "--no-rebuild-signatures",
        action="store_true",
        help="Ingest and classify source tags without rebuilding final enriched signatures.",
    )
    ingest_local.add_argument("--model", default=None)

    extract_lastfm = sub.add_parser("extract-lastfm", help="Fetch Last.fm top tags via API and ingest as a source page")
    add_release_filters(extract_lastfm)
    extract_lastfm.add_argument("--dry-run", action="store_true")
    extract_lastfm.add_argument("--adjudicate", action="store_true", help="Send unknown tags to AI for adjudication")
    extract_lastfm.add_argument(
        "--no-rebuild-signatures",
        action="store_true",
        help="Ingest and classify source tags without rebuilding final enriched signatures.",
    )
    extract_lastfm.add_argument("--model", default=None)
    extract_lastfm.add_argument("--lastfm-api-key", help="Last.fm API key (overrides config.yaml/env)")
    extract_lastfm.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip releases that already have a Last.fm source page (resumable, no double effort).",
    )

    extract_bandcamp = sub.add_parser("extract-bandcamp", help="Find Bandcamp URL via AI and ingest release tags")
    add_release_filters(extract_bandcamp)
    extract_bandcamp.add_argument("--dry-run", action="store_true")
    extract_bandcamp.add_argument("--adjudicate", action="store_true", help="Send unknown tags to AI for adjudication")
    extract_bandcamp.add_argument(
        "--no-rebuild-signatures",
        action="store_true",
        help="Ingest and classify source tags without rebuilding final enriched signatures.",
    )
    extract_bandcamp.add_argument("--model", default=DEFAULT_MODEL)
    extract_bandcamp.add_argument("--openai-api-key", help="OpenAI API key (overrides env/config.yaml)")
    extract_bandcamp.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip releases already attempted (hit OR miss) so reruns never re-pay the LLM locator.",
    )
    extract_bandcamp.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between releases (politeness / rate control). Default 0.5.",
    )
    extract_bandcamp.add_argument(
        "--web-mode",
        choices=["off", "auto", "required"],
        default="required",
        help="Web search for the URL locator. 'required' (default) forces a live "
             "search; 'off' makes the model guess from memory (URLs will 404).",
    )

    review_parser = sub.add_parser("review", help="Interactive CLI review of unclassified tags")
    review_parser.add_argument("--limit", type=int, default=20)
    review_parser.add_argument("--release-key")
    review_parser.add_argument("--source-type")

    review_esc = sub.add_parser(
        "review-escalated",
        help="Interactive CLI review of AI-escalated release suggestions",
    )
    review_esc.add_argument("--limit", type=int)
    review_esc.add_argument("--release-key")
    review_esc.add_argument("--artist")
    review_esc.add_argument("--album")

    graduate = sub.add_parser("graduate-reviewed", help="Graduate human-reviewed tags into vocabulary YAML")
    graduate.add_argument(
        "--vocab-yaml",
        type=Path,
        default=ROOT / "data" / "genre_vocabulary.yaml",
    )

    graduate_ai = sub.add_parser("graduate-ai", help="Graduate AI-adjudicated tags into vocabulary YAML")
    graduate_ai.add_argument(
        "--vocab-yaml",
        type=Path,
        default=ROOT / "data" / "genre_vocabulary.yaml",
    )
    graduate_ai.add_argument(
        "--min-times-seen",
        type=int,
        default=3,
        help="Minimum times a tag must have been seen to graduate (default: 3)",
    )

    rebuild = sub.add_parser(
        "rebuild-artifacts",
        help="Rebuild data_matrices_step1.npz (legacy genres by default; enriched modes require opt-in)",
    )
    rebuild.add_argument(
        "--artifacts-dir",
        default="data/artifacts/beat3tower_32k",
        help="Directory containing the artifact NPZ (default: data/artifacts/beat3tower_32k)",
    )
    rebuild.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    rebuild.add_argument(
        "--genre-sim-path",
        default=None,
        help="Optional path to genre similarity NPZ for smoothing",
    )
    rebuild.add_argument(
        "--genre-source",
        choices=["legacy", "enriched", "hybrid_shadow", "layered_shadow"],
        default="legacy",
    )
    rebuild.add_argument(
        "--overwrite-shadow",
        action="store_true",
        help="Allow replacement of an existing fingerprinted hybrid shadow artifact",
    )

    model_prior_one = sub.add_parser("model-prior-one", help="Generate or preview one no-web album model prior")
    add_release_filters(model_prior_one)
    model_prior_one.add_argument("--dry-run", action="store_true")
    model_prior_one.add_argument("--force", action="store_true")
    model_prior_one.add_argument("--model", default=None)

    model_prior = sub.add_parser("model-prior", help="Generate no-web album model priors in a bounded batch")
    add_release_filters(model_prior)
    model_prior.add_argument("--dry-run", action="store_true")
    model_prior.add_argument("--missing-only", action="store_true")
    model_prior.add_argument("--force", action="store_true")
    model_prior.add_argument("--model", default=None)

    sub.add_parser("model-prior-report", help="Report album model-prior coverage and mapping status")
    sub.add_parser("graph-init", help="Initialize or refresh layered genre graph taxonomy tables")
    sub.add_parser("graph-report", help="Report layered genre graph taxonomy counts")
    graph_build = sub.add_parser("graph-build-assignments", help="Build layered graph assignments from existing evidence")
    add_release_filters(graph_build)
    graph_build.add_argument("--dry-run", action="store_true")

    graph_show = sub.add_parser("graph-show-release", help="Show layered graph assignments for one release")
    add_release_filters(graph_show)
    graph_show.add_argument("--release-key", help="Inspect one exact release_key, bypassing fuzzy artist/album matching")

    graph_fixtures = sub.add_parser("graph-fixture-report", help="Run layered graph smoke fixtures and report failures")
    graph_fixtures.add_argument("--fixtures", type=Path, default=DEFAULT_LAYERED_FIXTURES)
    graph_fixtures.add_argument(
        "--build-assignments",
        action="store_true",
        help="Materialize layered assignments in the selected sidecar before evaluating each fixture.",
    )

    hybrid_one = sub.add_parser("hybrid-enrich-one", help="Fuse source evidence and model prior into one album genre report")
    add_release_filters(hybrid_one)
    hybrid_one.add_argument("--dry-run", action="store_true")
    hybrid_one.add_argument("--apply", action="store_true", help="Persist accepted hybrid genres into the sidecar enriched genre tables.")
    hybrid_one.add_argument("--include-provisional", action="store_true")
    hybrid_one.add_argument(
        "--with-model-prior",
        action="store_true",
        help="Generate or reuse the no-web LLM prior before fusing evidence.",
    )
    hybrid_one.add_argument(
        "--force-model-prior",
        action="store_true",
        help="Refresh the no-web LLM prior before fusing evidence.",
    )
    hybrid_one.add_argument("--model", default=None)

    propose_growth = sub.add_parser(
        "graph-propose-growth",
        help="Gather unmapped-genre candidates and AI-propose taxonomy placements")
    propose_growth.add_argument("--out", required=True,
                                help="Path to write the editable proposal YAML")
    propose_growth.add_argument("--min-album-freq", type=int, default=3)
    propose_growth.add_argument("--limit", type=int, default=None,
                                help="Cap number of candidates proposed (cost control)")
    propose_growth.add_argument("--web-mode", choices=["off", "auto", "required"],
                                default="off")
    propose_growth.add_argument("--model", default=None)
    propose_growth.add_argument("--openai-api-key")

    ingest_growth = sub.add_parser(
        "graph-ingest-growth",
        help="Validate + append decision:keep proposals into the taxonomy YAML")
    ingest_growth.add_argument("--proposals", required=True)
    ingest_growth.add_argument("--taxonomy-path", default=None,
                               help="Taxonomy YAML to grow (default: the packaged one)")
    ingest_growth.add_argument("--new-version", required=True,
                               help="taxonomy_version to stamp after growth")
    ingest_growth.add_argument("--dry-run", action="store_true")

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
    adjudicate = getattr(args, "adjudicate", False)
    model = getattr(args, "model", None) or resolve_enrichment_model(None)
    limit: int | None = args.limit
    # Discover without limit; when adjudicating, cached releases don't count toward limit.
    releases = discover_releases(
        args.metadata_db,
        limit=None if adjudicate else limit,
        artist=args.artist,
        album=args.album,
        generic_only=args.generic_only,
        min_existing_specific_genres=getattr(args, "min_existing_specific_genres", None),
    )
    if not releases:
        print("No matching release found.")
        return 1
    uncached_count = 0
    for release in releases:
        if adjudicate and limit is not None and uncached_count >= limit:
            break
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
        ai_called = False
        for page_id in page_ids:
            if store.classify_source_tags(page_id, adjudicate=adjudicate, model=model):
                ai_called = True
        if ai_called:
            uncached_count += 1
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
                    print(f"Warning: could not read album genres for {release.release_key}: {exc}", file=sys.stderr)
                    # Don't reset raw_genres — keep artist genres already collected
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
        store.classify_source_tags(
            page_id,
            adjudicate=getattr(args, "adjudicate", False),
            model=getattr(args, "model", None) or resolve_enrichment_model(None),
        )
        if not getattr(args, "no_rebuild_signatures", False):
            store.rebuild_enriched_genres_for_release(release.release_key)
        ingested += 1
        print(f"ingested {release.release_key} tags={len(deduped)}")

    print(f"Ingested {ingested} release(s).")
    return 0


def cmd_extract_lastfm(args: argparse.Namespace) -> int:
    if getattr(args, "dry_run", False):
        releases = _discover(args)
        if not releases:
            print("No matching release found.")
            return 1
        for release in releases:
            print(json.dumps({
                "release_key": release.release_key,
                "source_type": "lastfm_tags",
                "route": ["lastfm_api", "classify_tags"],
                "network_calls": 0,
                "sidecar_writes": 0,
                "dry_run": True,
            }, ensure_ascii=False, sort_keys=True))
        return 0

    import os
    import time
    from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary, reset_vocabulary

    api_key = getattr(args, "lastfm_api_key", None) or os.environ.get("LASTFM_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            config = Config()
            api_key = config.lastfm_api_key
        except FileNotFoundError:
            pass
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).debug("config.yaml parse error while resolving Last.fm API key", exc_info=True)
    if not api_key:
        print(
            "Error: Last.fm API key required. "
            "Set LASTFM_API_KEY env var, use --lastfm-api-key, or configure in config.yaml."
        )
        return 1

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    skipped_existing = 0
    if getattr(args, "skip_existing", False):
        already = store.release_keys_with_source_type("lastfm_tags")
        before = len(releases)
        releases = [r for r in releases if r.release_key not in already]
        skipped_existing = before - len(releases)
        print(
            f"skip-existing: {skipped_existing} release(s) already have Last.fm tags; "
            f"{len(releases)} remaining."
        )
        if not releases:
            print("Nothing to do — all matching releases already scraped.")
            return 0

    vocab = GenreVocabulary(library_db_path=args.metadata_db)
    set_vocabulary(vocab)

    total = len(releases)
    extracted = 0
    empty = 0
    failed = 0
    try:
        for idx, release in enumerate(releases, start=1):
            try:
                album_name = release.normalized_album or None
                tags = fetch_lastfm_tags(
                    artist=release.normalized_artist,
                    album=album_name,
                    api_key=api_key,
                    limit=20,
                )
                if not tags:
                    empty += 1
                    print(f"[{idx}/{total}] empty {release.release_key}")
                    time.sleep(0.25)
                    continue

                album_segment = f"/album/{release.normalized_album}" if release.normalized_album else ""
                page_id = store.upsert_source_page(
                    release_key=release.release_key,
                    normalized_artist=release.normalized_artist,
                    normalized_album=release.normalized_album,
                    album_id=release.album_id,
                    source_url=f"lastfm://artist/{release.normalized_artist}{album_segment}",
                    source_type="lastfm_tags",
                    identity_status="confirmed",
                    identity_confidence=0.9,
                    evidence_summary="Last.fm top tags via API.",
                )
                store.replace_source_tags(page_id, tags)
                store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
                if not getattr(args, "no_rebuild_signatures", False):
                    store.rebuild_enriched_genres_for_release(release.release_key)
                extracted += 1
                print(f"[{idx}/{total}] ok {release.release_key} tags={len(tags)}")
            except Exception as exc:  # network blip / API error — log and keep going
                failed += 1
                print(f"[{idx}/{total}] FAILED {release.release_key}: {type(exc).__name__}: {exc}")
            time.sleep(0.25)  # Last.fm rate limit: ~5 req/s, two calls per release
    finally:
        reset_vocabulary()

    print(
        f"Last.fm collection done: extracted={extracted} empty={empty} "
        f"failed={failed} skipped_existing={skipped_existing} considered={total}."
    )
    return 0


def cmd_extract_bandcamp(args: argparse.Namespace) -> int:
    if getattr(args, "dry_run", False):
        releases = _discover(args)
        if not releases:
            print("No matching release found.")
            return 1
        for release in releases:
            print(json.dumps({
                "release_key": release.release_key,
                "source_type": "bandcamp_release",
                "route": ["openai_source_locator", "bandcamp_release_html", "classify_tags"],
                "network_calls": 0,
                "sidecar_writes": 0,
                "dry_run": True,
            }, ensure_ascii=False, sort_keys=True))
        return 0

    import os
    from src.ai_genre_enrichment.bandcamp_enrichment import fetch_bandcamp_tags
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary, reset_vocabulary

    api_key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            config = Config()
            api_key = config.openai_api_key
        except (FileNotFoundError, AttributeError):
            pass
    if not api_key:
        print(
            "Error: OpenAI API key required. "
            "Set OPENAI_API_KEY env var, use --openai-api-key, or configure in config.yaml."
        )
        return 1

    import time

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    skipped_existing = 0
    if getattr(args, "skip_existing", False):
        attempted = store.release_keys_attempted("bandcamp")
        before = len(releases)
        releases = [r for r in releases if r.release_key not in attempted]
        skipped_existing = before - len(releases)
        print(
            f"skip-existing: {skipped_existing} release(s) already attempted (hit or miss); "
            f"{len(releases)} remaining."
        )
        if not releases:
            print("Nothing to do — all matching releases already attempted.")
            return 0

    vocab = GenreVocabulary(library_db_path=args.metadata_db)
    set_vocabulary(vocab)

    delay = getattr(args, "delay", 0.5)
    total = len(releases)
    hits = 0
    misses = 0
    failed = 0
    try:
        for idx, release in enumerate(releases, start=1):
            try:
                album_name = release.normalized_album or None
                source_url, tags, locator_confidence = fetch_bandcamp_tags(
                    artist=release.normalized_artist,
                    album=album_name,
                    api_key=api_key,
                    model=args.model,
                    web_mode=getattr(args, "web_mode", "required"),
                )
                if not tags:
                    # Record the miss so reruns never re-pay the LLM locator.
                    store.record_source_attempt(release.release_key, "bandcamp", "miss")
                    misses += 1
                    print(f"[{idx}/{total}] miss {release.release_key}")
                    time.sleep(delay)
                    continue

                page_id = store.upsert_source_page(
                    release_key=release.release_key,
                    normalized_artist=release.normalized_artist,
                    normalized_album=release.normalized_album,
                    album_id=release.album_id,
                    source_url=source_url,
                    source_type="bandcamp_release",
                    identity_status="confirmed",
                    identity_confidence=locator_confidence,
                    evidence_summary="Bandcamp release tags via AI source locator.",
                )
                store.replace_source_tags(page_id, tags)
                store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
                if not getattr(args, "no_rebuild_signatures", False):
                    store.rebuild_enriched_genres_for_release(release.release_key)
                store.record_source_attempt(release.release_key, "bandcamp", "hit", source_url)
                hits += 1
                print(f"[{idx}/{total}] hit {release.release_key} tags={len(tags)} url={source_url}")
            except Exception as exc:  # locator/network/HTML error — log, do NOT record (allow retry)
                failed += 1
                print(f"[{idx}/{total}] FAILED {release.release_key}: {type(exc).__name__}: {exc}")
            time.sleep(delay)
    finally:
        reset_vocabulary()

    print(
        f"Bandcamp collection done: hits={hits} misses={misses} "
        f"failed={failed} skipped_existing={skipped_existing} considered={total}."
    )
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


def _print_release_result(
    release: "ReleasePayload",
    response_json: dict,
    route: "RouteDecision",
    idx: int,
    total: int,
    *,
    dry_run: bool = False,
    cost_usd: float | None = None,
    token_usage: dict | None = None,
) -> None:
    """Print a detailed per-album progress block to stdout."""
    w = len(str(total))
    label = f"[{idx:{w}}/{total}] {release.artist} / {release.album}"
    if dry_run:
        label += "  (dry-run)"
    sep = "─" * max(0, 72 - len(label) - 4)
    print(f"\n─── {label} {sep}")

    # Header: lane, evidence quality, confidence, cost
    conf = response_json.get("release_level_confidence")
    eq = response_json.get("evidence_quality") or "?"
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    cost_str = f"  |  ${cost_usd:.4f}" if cost_usd is not None else ""
    tok_str = ""
    if token_usage:
        inp = token_usage.get("input_tokens") or token_usage.get("estimated_prompt_tokens", 0)
        out = token_usage.get("output_tokens") or token_usage.get("estimated_output_tokens", 0)
        if inp or out:
            tok_str = f"  |  {inp}in/{out}out tok"
    print(f"  lane: {route.lane.value}  |  evidence: {eq}  |  conf: {conf_str}{cost_str}{tok_str}")

    if dry_run:
        print(f"  reasons: {'; '.join(route.reasons)}")
        return

    # Identity
    identity = response_json.get("release_identity") or {}
    id_status = identity.get("status", "?")
    id_artist = identity.get("canonical_artist", "")
    id_album = identity.get("canonical_album", "")
    id_note = identity.get("notes", "")
    id_line = f"  identity: {id_status}"
    if id_artist or id_album:
        id_line += f"  —  {id_artist} / {id_album}"
    if id_note:
        id_line += f"  ({id_note[:60]})"
    print(id_line)

    # Sources used (release-specific only)
    sources = response_json.get("source_evidence") or []
    src_types = [
        s.get("source_url") or s.get("source_type", "?")
        for s in sources
        if isinstance(s, dict) and s.get("release_specific")
    ]
    # Shorten to just domain or source_type for readability
    def _short_src(url_or_type: str) -> str:
        if "://" in url_or_type:
            from urllib.parse import urlparse as _up
            p = _up(url_or_type)
            return p.netloc.removeprefix("www.") or url_or_type[:30]
        return url_or_type
    if src_types:
        print(f"  sources: {', '.join(_short_src(s) for s in src_types)}")

    # Existing genres kept
    keep = response_json.get("existing_genres_to_keep") or []
    if keep:
        keep_names = "  •  ".join(item.get("genre", "?") for item in keep)
        print(f"  keep ({len(keep)}):  {keep_names}")

    # Existing genres pruned
    prune = response_json.get("existing_genres_to_prune") or []
    for item in prune:
        g = item.get("genre", "?")
        pt = item.get("prune_type", "?")
        reason = (item.get("reason") or "")[:70]
        print(f"  prune:  {g}  [{pt}]  \"{reason}\"")

    # New genres — auto-apply
    add_all = response_json.get("new_genres_to_add") or []
    auto_add = [item for item in add_all if item.get("auto_apply_eligible")]
    review_add = [item for item in add_all if not item.get("auto_apply_eligible")]
    if auto_add:
        parts = [f"{item.get('genre', '?')} ({item.get('confidence', 0):.2f})" for item in auto_add]
        print(f"  add / auto ({len(auto_add)}):  {',  '.join(parts)}")
    if review_add:
        parts = [
            f"{item.get('genre', '?')} ({item.get('confidence', 0):.2f}) [{item.get('recommendation_basis', '?')}]"
            for item in review_add
        ]
        print(f"  add / needs-review ({len(review_add)}):  {',  '.join(parts)}")

    # Descriptor tags
    desc = response_json.get("descriptor_tags") or []
    if desc:
        desc_names = ",  ".join(item.get("tag", "?") for item in desc)
        print(f"  descriptors ({len(desc)}):  {desc_names}")

    # Review-only suggestions (uncertain / require human judgement)
    review_only = response_json.get("review_only_suggestions") or []
    if review_only:
        parts = [f"{item.get('tag', '?')} ({item.get('confidence', 0):.2f})" for item in review_only]
        print(f"  review-only ({len(review_only)}):  {',  '.join(parts)}")

    # Warnings and escalation
    for w_msg in (response_json.get("warnings") or [])[:3]:
        print(f"  warn:  {w_msg}")
    if response_json.get("should_escalate"):
        notes = (response_json.get("uncertainty_notes") or [])[:2]
        note_str = "  —  " + "; ".join(notes) if notes else ""
        print(f"  *** ESCALATE: needs human review{note_str}")


def _run_releases(args: argparse.Namespace, releases: list[ReleasePayload]) -> int:
    import os as _os
    _api_key = getattr(args, "openai_api_key", None) or _os.environ.get("OPENAI_API_KEY")
    if not _api_key:
        try:
            from src.config_loader import Config as _Config
            _api_key = _Config().openai_api_key
        except (FileNotFoundError, AttributeError, KeyError):
            pass

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
    total = len(releases)
    w = len(str(total))

    for i, release in enumerate(releases, 1):
        prefix = f"[{i:{w}}/{total}]"
        route = route_release(release, getattr(args, "web_mode", WebMode.OFF.value))
        effective_web_mode = _effective_web_mode(route.lane, route.web_mode)
        if route.lane == EnrichmentLane.SKIP_WELL_TAGGED:
            skipped += 1
            skipped_well_tagged += 1
            print(f"{prefix} skip (well-tagged): {release.release_key}  —  {'; '.join(route.reasons)}")
            continue
        if (
            route.lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT
            and getattr(args, "max_web_enrichment", None) is not None
            and web_enrichment_used >= args.max_web_enrichment
        ):
            skipped += 1
            needs_review_count += 1
            print(f"{prefix} skip (max-web-enrichment): {release.release_key}")
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
            print(f"{prefix} cached: {release.release_key}")
            continue

        prompt = build_prompt(payload)
        if not args.dry_run:
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

        client = create_enrichment_client(
            model=args.model,
            dry_run=args.dry_run,
            web_mode=effective_web_mode,
            allowed_web_domains=getattr(args, "allowed_web_domains", None),
            api_key=_api_key,
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
            print(f"{prefix} FAIL: {release.release_key}  —  {result.error_message}")
            continue

        for key in tokens:
            tokens[key] += result.token_usage.get(key, 0)

        if args.dry_run:
            _print_release_result(
                release,
                result.response_json,
                route,
                i,
                total,
                dry_run=True,
                cost_usd=result.estimated_cost_usd,
                token_usage=result.token_usage,
            )
        else:
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
            _print_release_result(
                release,
                response_json,
                route,
                i,
                total,
                dry_run=False,
                cost_usd=result.estimated_cost_usd,
                token_usage=result.token_usage,
            )

        called += 1
        if route.lane == EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT:
            web_enrichment_used += 1
            authoritative_source_checks += 1
        else:
            no_web_checks += 1
        if not args.dry_run and result.response_json.get("should_escalate"):
            needs_review_count += 1

    if not args.dry_run:
        store.record_run_log(
            command=args.command,
            status="failed" if failed else "complete",
            releases_seen=total,
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

    print(
        f"\n{'─'*72}\n"
        f"Done: {total} releases  "
        f"|  checked: {called}  "
        f"|  skipped (well-tagged): {skipped_well_tagged}  "
        f"|  cached: {cache_hits}  "
        f"|  failed: {failed}  "
        f"|  escalated: {needs_review_count}"
    )
    if tokens.get("total_tokens"):
        print(
            f"Tokens: {tokens['input_tokens']}in / {tokens['output_tokens']}out / {tokens['total_tokens']}total"
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


def cmd_review(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    queue = store.get_review_queue(
        release_key=getattr(args, "release_key", None),
        source_type=getattr(args, "source_type", None),
        limit=args.limit,
    )
    if not queue:
        print("No tags in review queue.")
        return 0

    valid_keys = {"a": "genre_style", "d": "descriptor", "i": "instrument", "p": "place", "r": "rejected", "s": None}
    reviewed = 0
    for item in queue:
        context = store.get_review_context(item["release_key"])
        context_lines = [
            f"  {c['normalized_tag']} ({c['classification']}, {c['confidence']:.2f})"
            for c in context
            if c["normalized_tag"] != item["normalized_tag"]
        ]
        print(f"\nRelease: {item['normalized_artist']} — {item['normalized_album']}")
        print(f"Source:  {item['source_url']}")
        print(f"Current: {item['classification']} ({item['confidence']:.2f})")
        print(f'Tag: "{item["normalized_tag"]}"')
        if context_lines:
            print("Context:")
            for line in context_lines[:8]:
                print(line)
        print("[A]ccept genre  [D]escriptor  [I]nstrument  [P]lace  [R]eject  [S]kip  [Q]uit")

        while True:
            try:
                choice = input("> ").strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if choice == "q":
                return 0
            if choice in valid_keys:
                break
            print("Invalid choice. Use a/d/i/p/r/s/q.")

        classification = valid_keys[choice]
        if classification is None:
            continue

        store.record_review_decision(
            source_tag_id=item["source_tag_id"],
            release_key=item["release_key"],
            raw_tag=item["raw_tag"],
            normalized_tag=item["normalized_tag"],
            original_classification=item["classification"],
            reviewed_classification=classification,
        )
        store.rebuild_enriched_genres_for_release(item["release_key"])
        reviewed += 1
        print(f"  → {classification}")

    print(f"\nReviewed {reviewed} tag(s).")
    return 0


def cmd_review_escalated(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    queue = store.get_escalated_queue(
        release_key=getattr(args, "release_key", None),
        artist=getattr(args, "artist", None),
        album=getattr(args, "album", None),
    )
    if not queue:
        print("No escalated releases to review.")
        return 0

    limit = getattr(args, "limit", None)

    # Per-release accumulator state.
    cur_key: str | None = None
    cur: dict | None = None
    flushed = 0

    def _flush(acc: dict) -> None:
        nonlocal flushed
        store.set_user_override(
            release_key=acc["release_key"],
            normalized_artist=acc["normalized_artist"],
            normalized_album=acc["normalized_album"],
            genres_add=acc["genres_add"],
            genres_remove=acc["genres_remove"],
        )
        store.rebuild_enriched_genres_for_release(acc["release_key"])
        store.mark_check_complete(acc["check_id"])
        flushed += 1

    # Count distinct releases for the [idx/total] header.
    release_order: list[str] = []
    for row in queue:
        if row["release_key"] not in release_order:
            release_order.append(row["release_key"])
    total = len(release_order) if limit is None else min(limit, len(release_order))

    for row in queue:
        key = row["release_key"]
        if key != cur_key:
            # Boundary: flush the previous (fully traversed) release.
            if cur is not None:
                _flush(cur)
            if limit is not None and flushed >= limit:
                cur = None
                break
            cur_key = key
            notes = []
            try:
                notes = (json.loads(row["response_json"]) or {}).get("uncertainty_notes") or []
            except (TypeError, ValueError, json.JSONDecodeError):
                notes = []
            with store.connect() as _conn:
                artist_genres = [
                    r[0] for r in _conn.execute(
                        "SELECT DISTINCT genre FROM enriched_genres WHERE normalized_artist = ? ORDER BY genre",
                        (row["normalized_artist"],),
                    )
                ]
            cur = {
                "release_key": key,
                "normalized_artist": row["normalized_artist"],
                "normalized_album": row["normalized_album"],
                "check_id": row["check_id"],
                "genres_add": [],
                "genres_remove": [],
                "touched": False,
                "uncertainty_notes": notes,
                "idx": release_order.index(key) + 1,
                "artist_genres": artist_genres,
            }
            _print_escalated_header(cur, total)

        assert cur is not None
        # Context: other actionable suggestions for this release.
        keep_ctx = [
            r["genre"] for r in queue
            if r["release_key"] == key and r["suggestion_type"] == "add" and r["suggestion_id"] != row["suggestion_id"]
        ]
        prune_ctx = [
            r["genre"] for r in queue
            if r["release_key"] == key and r["suggestion_type"] == "prune" and r["suggestion_id"] != row["suggestion_id"]
        ]
        _print_escalated_suggestion(row, keep_ctx, prune_ctx)

        while True:
            try:
                choice = input("> ").strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print()
                if cur["touched"]:
                    _flush(cur)
                return 0
            if choice in {"a", "r", "s", "q"}:
                break
            print("Invalid choice. Use a/r/s/q.")

        if choice == "q":
            if cur["touched"]:
                _flush(cur)
            return 0

        cur["touched"] = True
        if choice == "a":
            if row["suggestion_type"] == "add":
                cur["genres_add"].append(row["genre"])
            else:
                cur["genres_remove"].append(row["genre"])
            print(f"  → accepted {row['suggestion_type']} {row['genre']}")
        else:
            print(f"  → {'rejected' if choice == 'r' else 'skipped'}")

    # Flush the final accumulated release (loop ended naturally).
    if cur is not None:
        _flush(cur)

    print(f"\nReviewed {flushed} release(s).")
    return 0


def _print_escalated_header(acc: dict, total: int) -> None:
    label = f"[{acc['idx']}/{total}] {acc['normalized_artist']} / {acc['normalized_album']}"
    sep = "─" * max(0, 72 - len(label) - 4)
    print(f"\n─── {label} {sep}")
    if acc.get("artist_genres"):
        print(f"  artist genres:  {'  •  '.join(acc['artist_genres'])}")
    if acc["uncertainty_notes"]:
        print(f"  uncertainty: {'; '.join(acc['uncertainty_notes'][:2])}")


def _print_escalated_suggestion(row: dict, keep_ctx: list[str], prune_ctx: list[str]) -> None:
    conf = row.get("suggestion_confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    basis = row.get("recommendation_basis") or "?"
    verb = "ADD" if row["suggestion_type"] == "add" else "PRUNE"
    print(f"\n  {verb}:  {row['genre']}  ({conf_str})  [{basis}]")
    reason = (row.get("reason") or "")[:80]
    if reason:
        print(f'  "{reason}"')
    ctx_parts = []
    if keep_ctx:
        ctx_parts.append("add → " + "  •  ".join(keep_ctx))
    if prune_ctx:
        ctx_parts.append("prune → " + "  •  ".join(prune_ctx))
    if ctx_parts:
        print("  context:  " + "  |  ".join(ctx_parts))
    print("[A]ccept  [R]eject  [S]kip  [Q]uit")


def cmd_graduate_reviewed(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    terms = store.get_graduated_terms()
    if not terms:
        print("No reviewed terms to graduate.")
        return 0

    vocab = GenreVocabulary(args.vocab_yaml)
    category_map = {
        "genre_style": "genre_style",
        "descriptor": "descriptor",
        "instrument": "instrument",
        "place": "place",
        "format": "format",
        "mood_function": "mood_function",
        "label_or_org": "label_or_org",
    }
    added = 0
    for classification, tags in terms.items():
        category = category_map.get(classification)
        if not category:
            continue
        for tag in sorted(tags):
            vocab.add_term(category, tag)
            added += 1
            print(f"  graduated {tag} → {category}")

    vocab.save()
    print(f"Graduated {added} term(s) to {args.vocab_yaml}")
    return 0


def cmd_graduate_ai(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import reset_vocabulary

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    terms = store.get_ai_graduated_terms(min_times_seen=args.min_times_seen)
    if not terms:
        print("No AI-adjudicated tags meet the graduation threshold.")
        reset_vocabulary()
        return 0

    vocab = GenreVocabulary(args.vocab_yaml)
    added = 0
    try:
        for classification, tags in sorted(terms.items()):
            for tag in sorted(tags):
                try:
                    # Check before adding so added counts net-new terms only
                    if classification == "genre_style":
                        already = vocab.classify_genre(tag) is not None
                    else:
                        already = vocab.classify_non_genre(tag) is not None
                    if not already:
                        vocab.add_term(classification, tag)
                        added += 1
                        print(f"  graduated {tag!r} → {classification}")
                except ValueError:
                    print(f"  skipped {tag!r} — unknown category {classification!r}")

        if added:
            vocab.save()
            print(f"\nGraduated {added} term(s) into {args.vocab_yaml}.")
        else:
            print("No new terms to graduate.")
    finally:
        reset_vocabulary()
    return 0


def cmd_rebuild_artifacts(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path
    from src.ai_genre_enrichment.artifact_modes import (
        GenreArtifactSource,
        make_resolver,
        publish_shadow_artifact,
        shadow_input_identities,
        shadow_output_paths,
        temporary_shadow_artifact_path,
    )
    from src.ai_genre_enrichment.policy import STABILIZED_POLICY_VERSION
    from src.analyze.artifact_builder import build_ds_artifacts

    genre_source = GenreArtifactSource.resolve(getattr(args, "genre_source", None))
    artifacts_dir = _Path(getattr(args, "artifacts_dir", "data/artifacts/beat3tower_32k"))
    active_path = artifacts_dir / "data_matrices_step1.npz"
    config_path = getattr(args, "config", "config.yaml")
    genre_sim_path = getattr(args, "genre_sim_path", None)
    is_shadow_source = genre_source in {GenreArtifactSource.HYBRID_SHADOW, GenreArtifactSource.LAYERED_SHADOW}
    emit_layered_vectors = genre_source is GenreArtifactSource.LAYERED_SHADOW
    if is_shadow_source:
        SidecarStore(args.sidecar_db).initialize()
        shadow_inputs = shadow_input_identities(
            sidecar_db=args.sidecar_db,
            active_sparse_artifact=active_path,
            metadata_db=args.metadata_db,
            config_path=config_path,
            genre_sim_path=genre_sim_path,
            policy_version=STABILIZED_POLICY_VERSION,
            prior_snapshot="none",
            dense_config={"dim": 64, "skip_prior": True},
        )
        paths = shadow_output_paths(
            artifacts_dir=artifacts_dir,
            genre_source=genre_source.value,
            **shadow_inputs,
        )
        out_path = paths.sparse_artifact
        if out_path.resolve() == active_path.resolve():
            raise ValueError("hybrid_shadow output must not overwrite the active artifact")
        if out_path.exists() and not getattr(args, "overwrite_shadow", False):
            raise ValueError(
                f"hybrid_shadow output already exists at {out_path}; "
                "pass --overwrite-shadow to replace it"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        build_path = temporary_shadow_artifact_path(out_path)
    else:
        out_path = active_path
        build_path = out_path
    resolver = make_resolver(genre_source, args.sidecar_db)

    try:
        result = build_ds_artifacts(
            db_path=str(args.metadata_db),
            config_path=config_path,
            out_path=build_path,
            genre_sim_path=genre_sim_path,
            enriched_resolver=resolver,
            read_only_metadata=True,
            emit_layered_vectors=emit_layered_vectors,
            layered_sidecar_db=args.sidecar_db if emit_layered_vectors else None,
        )
        if is_shadow_source:
            current_shadow_inputs = shadow_input_identities(
                sidecar_db=args.sidecar_db,
                active_sparse_artifact=active_path,
                metadata_db=args.metadata_db,
                config_path=config_path,
                genre_sim_path=genre_sim_path,
                policy_version=STABILIZED_POLICY_VERSION,
                prior_snapshot="none",
                dense_config={"dim": 64, "skip_prior": True},
            )
            if current_shadow_inputs != shadow_inputs:
                changed_inputs = sorted(
                    name
                    for name, identity in shadow_inputs.items()
                    if current_shadow_inputs[name] != identity
                )
                raise ValueError(
                    "hybrid_shadow inputs changed during build; refusing to publish stale "
                    f"artifact (changed: {', '.join(changed_inputs)}). Re-run the build."
                )
            publish_shadow_artifact(
                build_path,
                out_path,
                overwrite=getattr(args, "overwrite_shadow", False),
            )
    finally:
        if is_shadow_source:
            build_path.unlink(missing_ok=True)
    print(
        f"Rebuilt artifacts at {out_path} "
        f"(tracks={result.n_tracks}, genres={result.n_genres})"
    )
    return 0


def _run_model_prior_release(args: argparse.Namespace, release: ReleasePayload) -> int:
    payload = build_model_prior_payload(release)
    input_hash = stable_input_hash(payload)

    store: SidecarStore | None = None
    if not args.dry_run:
        store = SidecarStore(args.sidecar_db)
        store.initialize()
        cached = store.find_model_prior(
            release_key=release.release_key, provider=get_enrichment_provider(), model=args.model,
            prompt_version=MODEL_PRIOR_PROMPT_VERSION, taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION,
            schema_version=MODEL_PRIOR_SCHEMA_VERSION, enrichment_policy_version=STABILIZED_POLICY_VERSION,
            input_hash=input_hash,
        )
        if cached and getattr(args, "missing_only", False) and not args.force:
            print(f"existing-model-prior {release.release_key}")
            return 0
        if cached and cached["status"] == "complete" and not args.force:
            print(f"cached-model-prior {release.release_key}")
            return 0

    client = create_enrichment_client(model=args.model, dry_run=args.dry_run, web_mode="off")
    result = client.request_structured(
        payload=payload,
        prompt=build_model_prior_prompt(payload),
        response_format=model_prior_response_format(),
        validator=validate_model_prior_response,
        instructions=MODEL_PRIOR_INSTRUCTIONS,
        estimated_output_tokens=300,
    )
    if args.dry_run:
        print(json.dumps(result.response_json, ensure_ascii=False, sort_keys=True))
        return 0

    mapped_terms: list[dict] = []
    if result.status == "complete":
        vocabulary = GenreVocabulary(library_db_path=args.metadata_db)
        mapped_terms = map_model_prior_terms(result.response_json["genres"], vocabulary, payload=payload)
    assert store is not None
    store.record_model_prior(
        release_key=release.release_key, normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album, album_id=release.album_id,
        provider=get_enrichment_provider(), model=args.model, prompt_version=MODEL_PRIOR_PROMPT_VERSION,
        taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION, schema_version=MODEL_PRIOR_SCHEMA_VERSION,
        enrichment_policy_version=STABILIZED_POLICY_VERSION, input_hash=input_hash,
        status=result.status, response_json=result.response_json or None,
        warnings=result.response_json.get("warnings", []) if result.response_json else [],
        error_message=result.error_message,
        token_usage=result.token_usage, estimated_cost_usd=result.estimated_cost_usd,
        mapped_terms=mapped_terms,
    )
    print(f"{result.status}-model-prior {release.release_key}")
    return 0 if result.status == "complete" else 1


def _ensure_model_prior_for_hybrid(
    args: argparse.Namespace,
    release: ReleasePayload,
    store: SidecarStore,
) -> tuple[str, list[EvidenceTerm], str | None]:
    payload = build_model_prior_payload(release)
    input_hash = stable_input_hash(payload)
    model = getattr(args, "model", None) or resolve_enrichment_model(None)
    cached = store.find_model_prior(
        release_key=release.release_key,
        provider=get_enrichment_provider(),
        model=model,
        prompt_version=MODEL_PRIOR_PROMPT_VERSION,
        taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION,
        schema_version=MODEL_PRIOR_SCHEMA_VERSION,
        enrichment_policy_version=STABILIZED_POLICY_VERSION,
        input_hash=input_hash,
    )
    if cached and cached["status"] == "complete" and not getattr(args, "force_model_prior", False):
        return "cached", [], None

    client = create_enrichment_client(model=model, dry_run=False, web_mode="off")
    result = client.request_structured(
        payload=payload,
        prompt=build_model_prior_prompt(payload),
        response_format=model_prior_response_format(),
        validator=validate_model_prior_response,
        instructions=MODEL_PRIOR_INSTRUCTIONS,
        estimated_output_tokens=300,
    )
    if result.status != "complete":
        return result.status, [], result.error_message

    vocabulary = GenreVocabulary(library_db_path=args.metadata_db)
    mapped_terms = map_model_prior_terms(result.response_json["genres"], vocabulary, payload=payload)
    if not getattr(args, "dry_run", False):
        store.record_model_prior(
            release_key=release.release_key,
            normalized_artist=release.normalized_artist,
            normalized_album=release.normalized_album,
            album_id=release.album_id,
            provider=get_enrichment_provider(),
            model=model,
            prompt_version=MODEL_PRIOR_PROMPT_VERSION,
            taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION,
            schema_version=MODEL_PRIOR_SCHEMA_VERSION,
            enrichment_policy_version=STABILIZED_POLICY_VERSION,
            input_hash=input_hash,
            status=result.status,
            response_json=result.response_json or None,
            warnings=result.response_json.get("warnings", []) if result.response_json else [],
            error_message=result.error_message,
            token_usage=result.token_usage,
            estimated_cost_usd=result.estimated_cost_usd,
            mapped_terms=mapped_terms,
        )
        return "complete", [], None

    return "complete-transient", _mapped_terms_to_evidence(mapped_terms), None


def _mapped_terms_to_evidence(mapped_terms: list[dict]) -> list[EvidenceTerm]:
    return [
        EvidenceTerm(
            term=str(term["normalized_term"]),
            source_type="model_prior",
            confidence=float(term["confidence"]),
            canonical_slug=term.get("canonical_slug") or term["normalized_term"],
            mapping_status=str(term["mapping_status"]),
            notes=str(term.get("notes") or ""),
        )
        for term in mapped_terms
    ]


def cmd_model_prior_one(args: argparse.Namespace) -> int:
    releases = _discover(args)
    if len(releases) != 1:
        print(f"Expected exactly one release, found {len(releases)}.")
        return 2
    return _run_model_prior_release(args, releases[0])


def cmd_model_prior(args: argparse.Namespace) -> int:
    failures = 0
    for release in _discover(args):
        rc = _run_model_prior_release(args, release)
        if rc != 0:
            failures += 1
    print(f"model-prior batch complete failures={failures}")
    return 1 if failures else 0


def cmd_model_prior_report(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    print(json.dumps(store.model_prior_report(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_graph_init(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    summary = store.upsert_layered_taxonomy(taxonomy)
    summary["taxonomy_version"] = taxonomy.version
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_graph_report(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    print(json.dumps(store.layered_taxonomy_report(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_graph_build_assignments(args: argparse.Namespace) -> int:
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    if not args.dry_run:
        store.upsert_layered_taxonomy(taxonomy)

    rows: list[dict[str, object]] = []
    for release in releases:
        fused_report = _fuse_hybrid_for_release(store, release)
        if args.dry_run:
            rows.append(
                {
                    "release_key": release.release_key,
                    "dry_run": True,
                    "accepted_genres": [decision.term for decision in fused_report.accepted_genres],
                    "needs_review": [decision.term for decision in fused_report.needs_review],
                    "rejected_noise": [decision.term for decision in fused_report.rejected_noise],
                }
            )
            continue
        summary = materialize_layered_assignments(
            store,
            release_id=release.release_key,
            artist=release.normalized_artist,
            album=release.normalized_album,
            report=fused_report,
            taxonomy=taxonomy,
        )
        rows.append(
            {
                "release_key": release.release_key,
                "dry_run": False,
                "genre_assignment_count": summary.genre_assignment_count,
                "facet_assignment_count": summary.facet_assignment_count,
                "rejected_term_count": summary.rejected_term_count,
                "review_term_count": summary.review_term_count,
            }
        )

    print(json.dumps({"releases": rows}, ensure_ascii=True, sort_keys=True))
    return 0


def cmd_graph_show_release(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)
    if getattr(args, "release_key", None):
        summary = build_layered_release_diagnostics(
            store,
            release_id=args.release_key,
            taxonomy=taxonomy,
        )
        summary["lookup"] = {"mode": "release_key", "release_key": args.release_key, "ambiguous": False}
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    releases = _discover(args)
    if len(releases) != 1:
        print(f"Expected exactly one release, found {len(releases)}.")
        return 2
    summary = build_layered_release_diagnostics(
        store,
        release_id=releases[0].release_key,
        taxonomy=taxonomy,
        sparse_release=not releases[0].existing_genres_by_source,
    )
    summary["lookup"] = {
        "mode": "artist_album",
        "artist": getattr(args, "artist", None),
        "album": getattr(args, "album", None),
        "release_key": releases[0].release_key,
        "ambiguous": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_graph_fixture_report(args: argparse.Namespace) -> int:
    import yaml

    fixture_doc = yaml.safe_load(Path(args.fixtures).read_text(encoding="utf-8")) or {}
    fixtures = list(fixture_doc.get("fixtures") or [])
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)

    results: list[dict[str, object]] = []
    pass_count = 0
    fail_count = 0
    for fixture in fixtures:
        result = _run_graph_fixture(args, store, taxonomy, fixture)
        results.append(result)
        if result["failures"]:
            fail_count += 1
        else:
            pass_count += 1

    print(json.dumps(
        {
            "fixture_version": fixture_doc.get("version"),
            "summary": {"pass": pass_count, "fail": fail_count},
            "fixtures": results,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 1 if fail_count else 0


def _run_graph_fixture(
    args: argparse.Namespace,
    store: SidecarStore,
    taxonomy,
    fixture: dict[str, object],
) -> dict[str, object]:
    release_key = str(fixture.get("release_key") or "").strip()
    lookup: dict[str, object] = {"mode": "release_key" if release_key else "artist_album"}
    failures: list[str] = []
    if not release_key:
        releases = _discover(argparse.Namespace(
            metadata_db=args.metadata_db,
            limit=None,
            artist=fixture.get("artist"),
            album=fixture.get("album"),
            generic_only=False,
            min_existing_specific_genres=None,
        ))
        lookup["matched_release_keys"] = [release.release_key for release in releases]
        if len(releases) != 1:
            failures.append("ambiguous_release_match" if releases else "no_release_match")
            return {
                "id": fixture.get("id"),
                "artist": fixture.get("artist"),
                "album": fixture.get("album"),
                "release_key": None,
                "lookup": lookup,
                "failures": failures,
            }
        release = releases[0]
        release_key = release.release_key
        sparse_release = not release.existing_genres_by_source
    else:
        release = None
        sparse_release = False
    if getattr(args, "build_assignments", False):
        if release is not None:
            fused_report = _fuse_hybrid_for_release(store, release)
            materialize_layered_assignments(
                store,
                release_id=release.release_key,
                artist=release.normalized_artist,
                album=release.normalized_album,
                report=fused_report,
                taxonomy=taxonomy,
            )
        else:
            evidence = collect_hybrid_evidence(store, release_key)
            fused_report = fuse_hybrid_evidence(
                release_key=release_key,
                evidence=evidence,
                sparse_release=False,
            )
            materialize_layered_assignments(
                store,
                release_id=release_key,
                artist=str(fixture.get("artist") or "").casefold(),
                album=str(fixture.get("album") or "").casefold(),
                report=fused_report,
                taxonomy=taxonomy,
            )
    diagnostics = build_layered_release_diagnostics(
        store,
        release_id=release_key,
        taxonomy=taxonomy,
        sparse_release=sparse_release,
    )
    failures.extend(_graph_fixture_failures(fixture, diagnostics))
    return {
        "id": fixture.get("id"),
        "artist": fixture.get("artist"),
        "album": fixture.get("album"),
        "release_key": release_key,
        "lookup": lookup,
        "evidence_status": diagnostics["evidence_status"],
        "zero_assignment_status": diagnostics.get("zero_assignment_status", diagnostics["evidence_status"]),
        "model_prior_exists": diagnostics["model_prior_exists"],
        "model_prior_presence": diagnostics.get("model_prior_presence", diagnostics["model_prior_exists"]),
        "genre_assignment_count": diagnostics["genre_assignment_count"],
        "facet_assignment_count": diagnostics["facet_assignment_count"],
        "accepted_leaf_terms": [row["term"] for row in diagnostics["accepted_leaf_terms"]],
        "accepted_broad_terms": [row["term"] for row in diagnostics["accepted_broad_terms"]],
        "accepted_facets": [row["term"] for row in diagnostics["accepted_facets"]],
        "inferred_terms": [row["term"] for row in diagnostics["inferred_terms"]],
        "review_terms": [row["term"] for row in diagnostics["review_terms"]],
        "rejected_terms": [row["term"] for row in diagnostics["rejected_terms"]],
        "missing_taxonomy_terms": diagnostics["missing_taxonomy_terms"],
        "failures": failures,
    }


def _graph_fixture_failures(fixture: dict[str, object], diagnostics: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if fixture.get("expected_no_evidence") and diagnostics["evidence_status"] == "no_evidence":
        return failures
    accepted_leaf = {str(row["term"]) for row in diagnostics["accepted_leaf_terms"]}
    accepted_broad = {str(row["term"]) for row in diagnostics["accepted_broad_terms"]}
    accepted_facets = {str(row["term"]) for row in diagnostics.get("accepted_facets", [])}
    accepted = accepted_leaf | accepted_broad | accepted_facets
    inferred = {str(row["term"]) for row in diagnostics["inferred_terms"]}
    rejected = {str(row["term"]) for row in diagnostics["rejected_terms"]}
    review = {str(row["term"]) for row in diagnostics["review_terms"]}
    observed = {str(row["term"]) for row in diagnostics.get("normalized_evidence", [])}

    for term in fixture.get("expected_leaf_terms") or []:
        if str(term) not in accepted_leaf:
            failures.append(f"missing_expected_leaf:{term}")
    for term in fixture.get("expected_broad_terms") or []:
        if str(term) not in accepted and str(term) not in inferred:
            failures.append(f"missing_expected_broad:{term}")
    for term in fixture.get("expected_rejected_terms") or []:
        if str(term) in observed and str(term) not in rejected:
            failures.append(f"missing_expected_reject:{term}")
    for term in fixture.get("acceptable_review_terms") or []:
        if str(term) in observed and str(term) not in review and str(term) not in accepted and str(term) not in inferred:
            failures.append(f"missing_expected_review_or_assignment:{term}")
    for term in fixture.get("forbidden_accepted_terms") or []:
        if str(term) in accepted:
            failures.append(f"forbidden_accepted:{term}")
    for term in fixture.get("forbidden_inferred_terms") or []:
        if str(term) in inferred:
            failures.append(f"forbidden_inferred:{term}")
    if fixture.get("fail_zero_assignments_when_evidence_exists") and diagnostics["evidence_status"] == "evidence_present_no_assignments":
        failures.append("zero_assignments_with_evidence")
    return failures


def cmd_graph_propose_growth(args: argparse.Namespace) -> int:
    import os
    from src.ai_genre_enrichment import graph_growth

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    candidates = graph_growth.collapse_variants(
        graph_growth.gather_growth_candidates(
            store, taxonomy, min_album_freq=args.min_album_freq))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("No growth candidates found.")
        graph_growth.write_proposals(args.out, [])
        return 0

    api_key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            api_key = Config().openai_api_key
        except (FileNotFoundError, AttributeError):
            api_key = None
    client = create_enrichment_client(model=args.model, api_key=api_key,
                                     web_mode=args.web_mode)

    items = []
    total = len(candidates)
    for idx, cand in enumerate(candidates, start=1):
        try:
            proposal = graph_growth.propose_placement(
                cand, taxonomy, client=client, web_mode=args.web_mode)
            items.append((cand, proposal))
            print(f"[{idx}/{total}] proposed {cand.term} -> {proposal.name} "
                  f"(kind={proposal.kind}, parents={[e.get('target') for e in proposal.parent_edges]})")
        except Exception as exc:  # one bad proposal shouldn't lose the batch
            print(f"[{idx}/{total}] FAILED {cand.term}: {type(exc).__name__}: {exc}")
    graph_growth.write_proposals(args.out, items)
    print(f"Wrote {len(items)} proposal(s) to {args.out}. Review then run graph-ingest-growth.")
    return 0


def cmd_graph_ingest_growth(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment import graph_growth
    from src.ai_genre_enrichment.layered_taxonomy import (
        DEFAULT_TAXONOMY_PATH, load_layered_taxonomy)

    tax_path = args.taxonomy_path or str(DEFAULT_TAXONOMY_PATH)
    taxonomy = load_layered_taxonomy(tax_path)
    entries = graph_growth.read_proposals(args.proposals)
    kept = [e for e in entries if e.decision == "keep"]
    if not kept:
        print("No proposals marked decision: keep.")
        return 0

    approved = []
    skipped = []
    for e in kept:
        errs = graph_growth.validate_proposal(taxonomy, e.proposal)
        if errs:
            skipped.append((e.proposal.name, "; ".join(errs)))
        else:
            approved.append(e.proposal)

    for name, reason in skipped:
        print(f"SKIP {name}: {reason}")

    if not approved:
        print("All kept proposals failed validation; nothing to append.")
        return 1

    if args.dry_run:
        print(f"[dry-run] would append {len(approved)} record(s); "
              f"{len(skipped)} skipped. No write.")
        return 0

    result = graph_growth.append_approved_to_taxonomy(
        tax_path, approved, new_version=args.new_version)
    # Re-import the grown taxonomy into the sidecar graph tables.
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    store.upsert_layered_taxonomy(load_layered_taxonomy(tax_path))
    print(f"Appended {result.appended} genre(s); skipped {len(skipped)}. "
          f"Taxonomy now {args.new_version}.")
    return 0


def _fuse_hybrid_for_release(store: SidecarStore, release: ReleasePayload):
    evidence = collect_hybrid_evidence(store, release.release_key)

    # Inject artist/album-level genres from metadata.db as evidence.
    # Artist-level MusicBrainz tags are reliable genre signals but below "strong"
    # threshold, so they land as provisional via the musicbrainz-only rule.
    _SKIP_PREFIXES = ("artist:lastfm", "album:lastfm", "track:")
    for source_key, genres in release.existing_genres_by_source.items():
        if any(source_key.startswith(p) for p in _SKIP_PREFIXES):
            continue
        parts = source_key.split(":", 1)
        if len(parts) != 2:
            continue
        src = parts[1]
        if "musicbrainz" in src:
            source_type, conf = "musicbrainz", 0.75
        elif "discogs" in src:
            source_type, conf = "discogs", 0.78
        else:
            continue
        for genre in genres:
            genre_norm = genre.strip().casefold()
            if genre_norm:
                evidence.append(EvidenceTerm(
                    term=genre_norm,
                    source_type=source_type,
                    confidence=conf,
                    canonical_slug=genre_norm,
                    mapping_status="mapped",
                    classifier="metadata_db",
                ))

    return fuse_hybrid_evidence(
        release_key=release.release_key,
        evidence=evidence,
        sparse_release=not release.existing_genres_by_source,
    )


def cmd_hybrid_enrich_one(args: argparse.Namespace) -> int:
    if args.dry_run and args.apply:
        print("hybrid-enrich-one cannot combine --dry-run and --apply.")
        return 2
    releases = _discover(args)
    if len(releases) != 1:
        print(f"Expected exactly one release, found {len(releases)}.")
        return 2

    release = releases[0]
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    transient_evidence: list[EvidenceTerm] = []
    model_prior_status: str | None = None
    model_prior_error: str | None = None
    if args.with_model_prior or args.force_model_prior:
        model_prior_status, transient_evidence, model_prior_error = _ensure_model_prior_for_hybrid(args, release, store)
    evidence = collect_hybrid_evidence(store, release.release_key)
    evidence.extend(transient_evidence)
    sparse_release = not release.existing_genres_by_source
    fused_report = fuse_hybrid_evidence(
        release_key=release.release_key,
        evidence=evidence,
        sparse_release=sparse_release,
    )
    report = fused_report.to_dict()
    applied_count = 0
    layered_assignment_count = 0
    layered_facet_assignment_count = 0
    if args.apply:
        genres_to_apply = list(report["accepted_genres"])
        if args.include_provisional:
            genres_to_apply.extend(report["provisional_genres"])
        applied_count = store.replace_hybrid_enriched_genres_for_release(
            release_key=release.release_key,
            normalized_artist=release.normalized_artist,
            normalized_album=release.normalized_album,
            album_id=release.album_id,
            accepted_genres=genres_to_apply,
        )
        taxonomy = load_default_layered_taxonomy()
        store.upsert_layered_taxonomy(taxonomy)
        layered_summary = materialize_layered_assignments(
            store,
            release_id=release.release_key,
            artist=release.normalized_artist,
            album=release.normalized_album,
            report=fused_report,
            taxonomy=taxonomy,
        )
        layered_assignment_count = layered_summary.genre_assignment_count
        layered_facet_assignment_count = layered_summary.facet_assignment_count
    report["dry_run"] = bool(args.dry_run)
    report["applied"] = bool(args.apply)
    report["applied_count"] = applied_count
    report["layered_assignment_count"] = layered_assignment_count
    report["layered_facet_assignment_count"] = layered_facet_assignment_count
    report["evidence_count"] = len(evidence)
    if model_prior_status is not None:
        report["model_prior_status"] = model_prior_status
    if model_prior_error:
        report["model_prior_error"] = model_prior_error
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
