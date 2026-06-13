"""Apply the surgical delta migration to legacy graph release assignments.

Dry-run by default: computes Delta A (add missing local-tag leaves) and
Delta B (remove storefront-only observed leaves) for every covered release,
writes a full report, and touches nothing. --apply backs up the sidecar first,
then rewrites only the releases with a non-empty delta via the same
replace-per-release write path the materializer uses.

See src/ai_genre_enrichment/assignment_migration.py for the rationale
(2026-06-12: wholesale re-derivation rejected twice at the publish gate).

Usage:
  python scripts/migrate_assignments_delta.py                  # dry-run, full report
  python scripts/migrate_assignments_delta.py --artist "VV Torso"
  python scripts/migrate_assignments_delta.py --apply
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai_genre_enrichment.assignment_migration import (
    ObservedTermEvidence,
    apply_surgical_delta,
    plan_release_delta,
    select_misidentified_lastfm_removals,
    select_storefront_removals,
)
from src.ai_genre_enrichment.hybrid_evidence import collect_hybrid_evidence
from src.ai_genre_enrichment.layered_assignment import classify_layered_term
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.normalization import (
    normalize_release_artist,
    normalize_release_name,
)
from src.ai_genre_enrichment.storage import SidecarStore

NON_LEAF_KINDS = {"family", "facet", "reject", "review"}

# Last.fm tags that are locations/junk, not genres — excluded from the Delta C
# contradiction signal so they don't manufacture false "overlap".
LASTFM_NOISE = {
    "ukraine", "ukrainian", "losangeles", "seenlive", "female", "male",
    "favorites", "spotify", "all", "usa", "unitedstates", "japan", "japanese",
    "uk", "british", "german", "germany", "french", "france", "instrumental",
    "beautiful", "chill", "favorite", "vinyl", "10s", "00s", "90s", "80s", "70s",
}


def _slug(value: str) -> str:
    import re as _re

    return _re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _tags_overlap(a: set[str], b: set[str]) -> bool:
    """Fuzzy overlap: any token in a is a substring of (or contains) one in b."""
    for x in a:
        for y in b:
            if x and y and (x in y or y in x):
                return True
    return False


def _canonical_gid(taxonomy, term: str, context: list[str]) -> tuple[str | None, str]:
    c = classify_layered_term(taxonomy, term, context_terms=context)
    return (c.canonical_id or None), c.term_kind


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-db", default="data/metadata.db")
    parser.add_argument("--sidecar-db", default="data/ai_genre_enrichment.db")
    parser.add_argument("--artist", default=None, help="Limit to one artist (normalized match)")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--report-out", default="C:/tmp/delta_migration_report.txt")
    parser.add_argument(
        "--protect-file",
        default=None,
        help="File of 'release_key::genre_id' lines exempted from removal "
        "(human-reviewed storefront tags that are correct)",
    )
    args = parser.parse_args(argv)

    protected: set[tuple[str, str]] = set()
    if args.protect_file:
        for line in Path(args.protect_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rk_part, _, gid_part = line.rpartition("::")
            if rk_part and gid_part:
                protected.add((rk_part, gid_part))

    mconn = sqlite3.connect(f"file:{args.metadata_db}?mode=ro", uri=True)

    # album_id -> release_key (+ raw artist/album for the report)
    album_by_rk: dict[str, tuple[str, str, str]] = {}
    for aid, artist, album in mconn.execute(
        "SELECT DISTINCT album_id, artist, album FROM tracks WHERE album_id IS NOT NULL"
    ):
        rk = f"{normalize_release_artist(artist or '')}::{normalize_release_name(album or '')}"
        album_by_rk[rk] = (aid, artist or "", album or "")

    # local file tags per album_id (track- and album-level)
    file_tags: dict[str, set[str]] = {}
    for aid, g in mconn.execute(
        "SELECT t.album_id, LOWER(TRIM(tg.genre)) FROM track_genres tg "
        "JOIN tracks t ON t.track_id = tg.track_id "
        "WHERE tg.source = 'file' AND t.album_id IS NOT NULL"
    ):
        if g and g != "__empty__":
            file_tags.setdefault(aid, set()).add(g)
    for aid, g in mconn.execute(
        "SELECT album_id, LOWER(TRIM(genre)) FROM album_genres WHERE source = 'file'"
    ):
        if g and g != "__empty__":
            file_tags.setdefault(aid, set()).add(g)

    # release-level MB/Discogs tags per album_id (independent protection for Delta B)
    release_level: dict[str, set[str]] = {}
    for aid, g, src in mconn.execute(
        "SELECT album_id, LOWER(TRIM(genre)), source FROM album_genres "
        "WHERE source LIKE '%musicbrainz%' OR source LIKE '%discogs%'"
    ):
        if g and g != "__empty__":
            release_level.setdefault(aid, set()).add(g)

    # All non-lastfm genre evidence per normalized artist (for Delta C
    # contradiction signal): file + album + artist + MB/discogs tags.
    other_evidence_by_artist: dict[str, set[str]] = {}
    for artist, g in mconn.execute(
        "SELECT t.artist, LOWER(TRIM(tg.genre)) FROM track_genres tg "
        "JOIN tracks t ON t.track_id = tg.track_id "
        "WHERE tg.genre IS NOT NULL AND tg.genre != '__EMPTY__'"
    ):
        other_evidence_by_artist.setdefault(normalize_release_artist(artist or ""), set()).add(_slug(g))
    for artist, g in mconn.execute(
        "SELECT t.artist, LOWER(TRIM(ag.genre)) FROM album_genres ag "
        "JOIN tracks t ON t.album_id = ag.album_id "
        "WHERE ag.genre IS NOT NULL AND ag.genre != '__EMPTY__' GROUP BY t.artist, ag.genre"
    ):
        other_evidence_by_artist.setdefault(normalize_release_artist(artist or ""), set()).add(_slug(g))
    try:
        for artist, g in mconn.execute(
            "SELECT artist, LOWER(TRIM(genre)) FROM artist_genres "
            "WHERE genre IS NOT NULL AND genre != '__EMPTY__'"
        ):
            other_evidence_by_artist.setdefault(normalize_release_artist(artist or ""), set()).add(_slug(g))
    except sqlite3.OperationalError:
        pass
    mconn.close()

    store = SidecarStore(args.sidecar_db)
    taxonomy = load_default_layered_taxonomy()

    sconn = sqlite3.connect(f"file:{args.sidecar_db}?mode=ro", uri=True)
    sconn.row_factory = sqlite3.Row
    releases = sconn.execute(
        "SELECT DISTINCT release_id, artist, album FROM genre_graph_release_genre_assignments "
        "ORDER BY release_id"
    ).fetchall()

    overrides: dict[str, set[str]] = {}
    try:
        for rk, add_json in sconn.execute(
            "SELECT release_key, genres_add_json FROM ai_genre_user_overrides"
        ):
            try:
                names = json.loads(add_json or "[]")
            except json.JSONDecodeError:
                continue
            overrides[rk] = {str(n).lower() for n in names}
    except sqlite3.OperationalError:
        pass

    # Delta C: per-artist Last.fm tag set (raw normalized tags, not canonical),
    # and the contradiction flag = the artist's lastfm tags share NOTHING with
    # >=2 substantive other-evidence tags (Green-House: lastfm hip-hop vs
    # bandcamp/file ambient). Artists with thin other evidence are grandfathered
    # (we can't second-guess lastfm when nothing contradicts it).
    lastfm_by_artist: dict[str, set[str]] = {}
    for artist, tag in sconn.execute(
        "SELECT p.normalized_artist, t.normalized_tag "
        "FROM ai_genre_source_tags t "
        "JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id "
        "WHERE p.source_type = 'lastfm_tags'"
    ):
        s = _slug(tag)
        if s and s not in LASTFM_NOISE:
            lastfm_by_artist.setdefault(artist or "", set()).add(s)

    contradicted_artists: set[str] = set()
    for artist, ltags in lastfm_by_artist.items():
        other = {t for t in other_evidence_by_artist.get(artist, set()) if t and t != "empty"}
        if ltags and len(other) >= 2 and not _tags_overlap(ltags, other):
            contradicted_artists.add(artist)

    if args.artist:
        want = normalize_release_artist(args.artist)
        releases = [r for r in releases if r["artist"] == want]

    plan: list[dict] = []
    add_counter: Counter = Counter()
    remove_counter: Counter = Counter()
    skipped_unmatched = 0

    for rel in releases:
        rk = rel["release_id"]
        rows = sconn.execute(
            "SELECT genre_id, assignment_layer, confidence, source_reliability, "
            "evidence_count, rejected_by_user, provenance_json "
            "FROM genre_graph_release_genre_assignments WHERE release_id = ?",
            (rk,),
        ).fetchall()
        observed = [dict(r) for r in rows if r["assignment_layer"] == "observed_leaf"]
        if not observed:
            continue
        for o in observed:
            try:
                o["provenance"] = json.loads(o.pop("provenance_json") or "{}")
            except json.JSONDecodeError:
                o["provenance"] = {}

        matched = rk in album_by_rk
        if not matched:
            # No album match -> can't see file tags or MB/Discogs protection.
            # Delta A impossible, Delta B unsafe: skip entirely (grandfather).
            skipped_unmatched += 1
            continue
        aid, raw_artist, raw_album = album_by_rk[rk]

        # ---- evidence attribution for Delta B ----
        evidence = collect_hybrid_evidence(store, rk)
        ev_terms = [e.term for e in evidence]
        sources_by_gid: dict[str, set[str]] = {}
        for e in evidence:
            gid, kind = _canonical_gid(taxonomy, e.term, ev_terms)
            if gid and kind not in {"reject", "review"}:
                sources_by_gid.setdefault(gid, set()).add(e.source_type)
        for g in release_level.get(aid, set()):
            gid, kind = _canonical_gid(taxonomy, g, ev_terms)
            if gid and kind not in {"reject", "review"}:
                sources_by_gid.setdefault(gid, set()).add("musicbrainz")
        local_gids: set[str] = set()
        ftags = sorted(file_tags.get(aid, set()))
        for g in ftags:
            gid, kind = _canonical_gid(taxonomy, g, ftags)
            if gid and kind not in {"reject", "review"}:
                sources_by_gid.setdefault(gid, set()).add("local_metadata")
                if kind not in NON_LEAF_KINDS:
                    local_gids.add(gid)

        user_gids: set[str] = set()
        for name in overrides.get(rk, set()):
            gid, _kind = _canonical_gid(taxonomy, name, [name])
            if gid:
                user_gids.add(gid)

        observed_gids = {o["genre_id"] for o in observed}
        term_evidence = [
            ObservedTermEvidence(
                genre_id=gid,
                evidence_sources=frozenset(sources_by_gid.get(gid, set())),
                user_added=gid in user_gids,
            )
            for gid in sorted(observed_gids)
        ]

        # Surgical observed-leaf delta (Delta A add / Delta B+C remove). Applied
        # directly to the stored rows by apply_surgical_delta — NOT a full
        # re-materialization, so correct non-target tags are never collateral.
        additions, removal_list = plan_release_delta(
            observed_terms=term_evidence,
            local_leaf_genre_ids=local_gids,
            has_local_tags=bool(ftags),
            artist_lastfm_contradicted=(rel["artist"] in contradicted_artists),
        )
        removals = [g for g in removal_list if (rk, g) not in protected]
        if not additions and not removals:
            continue

        # Trigger tags for the report (which delta(s) acted).
        trig = ""
        if additions:
            trig += "A"
        if bool(ftags) and any(g in set(select_storefront_removals(term_evidence)) for g in removals):
            trig += "B"
        if rel["artist"] in contradicted_artists and any(
            g in set(select_misidentified_lastfm_removals(term_evidence, artist_lastfm_contradicted=True))
            for g in removals
        ):
            trig += "C"

        add_counter.update(additions)
        remove_counter.update(removals)
        # Full existing row set (all layers) for the surgical apply.
        all_rows = [dict(r) for r in rows]
        for r in all_rows:
            try:
                r["provenance"] = json.loads(r.pop("provenance_json") or "{}")
            except (json.JSONDecodeError, KeyError):
                r["provenance"] = {}
        plan.append({
            "release_key": rk,
            "artist": raw_artist,           # display
            "album": raw_album,             # display
            "store_artist": rel["artist"],  # normalized, preserved on write
            "store_album": rel["album"],
            "additions": additions,
            "removals": sorted(removals),
            "triggers": trig,
            "existing_rows": all_rows,
        })

    sconn.close()

    # ---- report ----
    lines = [
        f"delta migration {'APPLY' if args.apply else 'DRY-RUN'}",
        f"covered releases scanned: {len(releases)} | touched: {len(plan)} | "
        f"skipped (no album match): {skipped_unmatched}",
        f"lastfm-contradicted artists (Delta C): {len(contradicted_artists)}",
        f"total additions: {sum(add_counter.values())} | total removals: {sum(remove_counter.values())}",
        f"top additions: {add_counter.most_common(15)}",
        f"top removals: {remove_counter.most_common(15)}",
        "",
    ]
    # Sort the per-release detail so removals (the higher-risk changes) sort first.
    detail_start = len(lines)
    for item in sorted(plan, key=lambda it: (not it["removals"], it["artist"])):
        lines.insert(
            detail_start,
            f"[{item['triggers']}] {item['artist']} - {item['album']}: "
            f"+{item['additions']} -{item['removals']}",
        )
        detail_start += 1
    report = "\n".join(lines)
    Path(args.report_out).write_text(report, encoding="utf-8")
    print("\n".join(lines[:7]))
    print(f"(full report: {args.report_out})")

    if not args.apply:
        return 0

    # ---- apply: backup, then surgically rewrite each touched release ----
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{args.sidecar_db}.bak_{ts}"
    src = sqlite3.connect(f"file:{args.sidecar_db}?mode=ro", uri=True)
    dst = sqlite3.connect(bak)
    src.backup(dst)
    src.close()
    dst.close()
    print(f"sidecar backed up: {bak}")

    # Facet rows are carried through unchanged (replace_* rewrites both tables).
    fconn = sqlite3.connect(f"file:{args.sidecar_db}?mode=ro", uri=True)
    fconn.row_factory = sqlite3.Row
    for item in plan:
        rk = item["release_key"]
        new_genre_rows = apply_surgical_delta(
            item["existing_rows"],
            additions=item["additions"],
            removals=item["removals"],
            taxonomy=taxonomy,
        )
        facet_rows = []
        for r in fconn.execute(
            "SELECT facet_id, confidence, source, provenance_json "
            "FROM genre_graph_release_facet_assignments WHERE release_id = ?",
            (rk,),
        ).fetchall():
            try:
                prov = json.loads(r["provenance_json"] or "{}")
            except json.JSONDecodeError:
                prov = {}
            facet_rows.append({
                "facet_id": r["facet_id"], "confidence": r["confidence"],
                "source": r["source"], "provenance": prov,
            })
        store.replace_layered_assignments_for_release(
            release_id=rk,
            artist=item["store_artist"],
            album=item["store_album"],
            genre_assignments=new_genre_rows,
            facet_assignments=facet_rows,
        )
    fconn.close()
    print(f"applied: {len(plan)} releases surgically updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
