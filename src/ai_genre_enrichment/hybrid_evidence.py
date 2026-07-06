"""Deterministic fusion policy for hybrid album genre evidence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

DecisionKind = Literal["accepted", "provisional", "rejected_noise"]

# Trust model (rebalanced 2026-06-12 after the VV Torso/LPVV incident):
#   - Bandcamp is self-reported ONLY when the artist runs the page
#     (bandcamp_artist). Label storefronts tag whole catalogs with store
#     sections ("indie rock, pop") — they must never auto-accept.
#   - ai_enriched_accepted is an echo of a past acceptance (usually of the
#     same bandcamp page) — corroborating-only, never establishes acceptance.
#   - local_metadata is the user's hand-curated file tags; for niche releases
#     it is often the only correct source. Raised above discogs, and protected
#     by the never-drop rule in fuse_hybrid_evidence.
SOURCE_WEIGHTS: dict[str, float] = {
    "bandcamp_artist": 0.95,       # artist-run page: self-reported
    "official_release": 0.95,
    "ai_check_web": 0.88,          # run-one suggestions backed by authoritative/review web sources
    "local_metadata": 0.80,        # user-curated file tags
    "discogs": 0.78,
    "musicbrainz": 0.76,
    "bandcamp_release": 0.70,      # legacy unclassified bandcamp page
    "bandcamp_unknown": 0.70,      # single-artist domain, name mismatch (operator unknown)
    "ai_check_metadata": 0.70,     # run-one suggestions derived from local metadata only
    "model_prior": 0.68,
    "bandcamp_label": 0.60,        # multi-artist storefront: catalog-level tags
    "ai_enriched_accepted": 0.60,  # echo of past acceptance — corroborating only
    "lastfm_tags": 0.25,
}

# Zero-touch policy (2026-07-04, M1 of the zero-touch genre pipeline spec):
# lastfm-only terms publish at this hard cap instead of queue-blocking. The
# artifact weights genres by confidence, so the cap IS the blast-radius
# control — 0.40 weight nudges steering, never defines it.
LASTFM_ONLY_CONFIDENCE_CAP = 0.40

LASTFM_SOURCE_TYPES = {"lastfm_tags", "lastfm"}
STRONG_SOURCE_TYPES = {"bandcamp_artist", "official_release", "ai_check_web"}
MEDIUM_SOURCE_TYPES = {
    "local_metadata", "discogs", "musicbrainz", "ai_check_metadata",
    "bandcamp_release", "bandcamp_unknown", "bandcamp_label",
}
AI_CHECK_TYPES = {"ai_check_web", "ai_check_metadata"}
REJECTED_NOISE_TERMS = {
    "indie": "Standalone indie is a scene/context marker, not a usable genre assignment.",
    "pop/rock": "Fake retail/store-section bucket; do not accept, infer, or split without independent evidence.",
}


@dataclass(frozen=True)
class EvidenceTerm:
    term: str
    source_type: str
    confidence: float
    canonical_slug: str | None = None
    mapping_status: str = "mapped"
    notes: str = ""
    classifier: str = "deterministic"


@dataclass(frozen=True)
class FusedGenreDecision:
    term: str
    confidence: float
    basis: str
    sources: list[str]
    reason: str


@dataclass(frozen=True)
class HybridGenreReport:
    release_key: str
    accepted_genres: list[FusedGenreDecision]
    provisional_genres: list[FusedGenreDecision]
    rejected_noise: list[FusedGenreDecision]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_WEB_CHECK_BASES = {"authoritative_source", "hybrid", "review_context"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def classify_bandcamp_source(
    domain: str, release_artist: str, *, domain_artist_count: int = 1
) -> str:
    """Classify a bandcamp page as artist-run, label storefront, or unknown.

    Artist-run (subdomain matches the artist) is self-reported — top-tier
    evidence. A domain hosting multiple distinct artists in our store is a
    label storefront whose tags describe the catalog, not the release. A
    mismatched single-artist domain could be either (renamed artist page or a
    label we sampled once) — treated as unknown: usable, never auto-accepted.
    """
    sub = _slug((domain or "").split(".", 1)[0])
    artist = _slug(release_artist)
    if artist and sub and (
        sub == artist
        or (len(artist) >= 4 and artist in sub)
        or (len(sub) >= 4 and sub in artist)
    ):
        return "bandcamp_artist"
    if int(domain_artist_count) > 1:
        return "bandcamp_label"
    return "bandcamp_unknown"


def collect_hybrid_evidence(store: object, release_key: str) -> list[EvidenceTerm]:
    evidence: list[EvidenceTerm] = []

    release_artist = release_key.split("::", 1)[0]
    domain_counts: dict[str, int] = {}
    if hasattr(store, "bandcamp_domain_artist_counts"):
        domain_counts = store.bandcamp_domain_artist_counts()

    for row in store.hybrid_source_terms_for_release(release_key):
        mapping = str(row.get("mapping_status") or "")
        if mapping == "genre_style":
            mapping = "mapped"
        confidence = float(row.get("confidence") or row.get("identity_confidence") or 0.50)
        source_type = str(row["source_type"])
        if source_type == "bandcamp_release":
            domain = str(row.get("source_domain") or "")
            source_type = classify_bandcamp_source(
                domain,
                str(row.get("normalized_artist") or release_artist),
                domain_artist_count=int(domain_counts.get(domain, 1)),
            )
        evidence.append(EvidenceTerm(
            term=str(row["term"]),
            source_type=source_type,
            confidence=confidence,
            canonical_slug=row.get("canonical_slug") or row["term"],
            mapping_status=mapping,
            classifier=str(row.get("classifier") or "deterministic"),
        ))

    for row in store.accepted_enriched_genres_for_release(release_key):
        genre = str(row["genre"]).strip().casefold()
        if not genre:
            continue
        evidence.append(EvidenceTerm(
            term=genre,
            source_type="ai_enriched_accepted",
            confidence=0.88,
            canonical_slug=genre,
            mapping_status="mapped",
            classifier="enriched_genres",
        ))

    for row in store.latest_check_suggestions_for_release(release_key):
        basis = str(row.get("recommendation_basis") or "")
        source_type = "ai_check_web" if basis in _WEB_CHECK_BASES else "ai_check_metadata"
        genre = str(row["genre"]).strip().casefold()
        if not genre:
            continue
        evidence.append(EvidenceTerm(
            term=genre,
            source_type=source_type,
            confidence=float(row.get("confidence") or 0.70),
            canonical_slug=genre,
            mapping_status="mapped",
            classifier="ai_enrichment",
        ))

    for row in store.latest_model_prior_terms_for_release(release_key):
        evidence.append(EvidenceTerm(
            term=str(row["normalized_term"]),
            source_type="model_prior",
            confidence=float(row["confidence"]),
            canonical_slug=row.get("canonical_slug") or row["normalized_term"],
            mapping_status=str(row["mapping_status"]),
            notes=str(row.get("notes") or ""),
            classifier="model_prior",
        ))

    return evidence


def fuse_hybrid_evidence(
    *,
    release_key: str,
    evidence: list[EvidenceTerm],
    sparse_release: bool,
) -> HybridGenreReport:
    has_non_lastfm_release_evidence = any(
        item.source_type not in LASTFM_SOURCE_TYPES and item.source_type != "model_prior"
        for item in evidence
    )
    grouped: dict[str, list[EvidenceTerm]] = {}
    for item in evidence:
        term = _decision_term(item)
        if term:
            grouped.setdefault(term, []).append(item)

    accepted: list[FusedGenreDecision] = []
    provisional: list[FusedGenreDecision] = []
    rejected: list[FusedGenreDecision] = []

    for term in sorted(grouped):
        items = grouped[term]
        sources = sorted({item.source_type for item in items})
        score = _score(items)

        rejected_reason = _rejected_noise_reason(term)
        if rejected_reason is not None:
            rejected.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis=_basis(sources),
                sources=sources,
                reason=rejected_reason,
            ))
            continue

        if all(source in LASTFM_SOURCE_TYPES for source in sources):
            # Published at capped confidence, never review-blocked. History:
            # pre-2026-06-12 this lane published at >=0.90 FULL weight and
            # produced 'baroque' on Debussy at scale; 2026-06-12 made it
            # review-only; 2026-07-04 (zero-touch M1) publishes it capped.
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=min(score, LASTFM_ONLY_CONFIDENCE_CAP),
                basis="lastfm_only",
                sources=sources,
                reason="Last.fm-only mapped signal published at capped confidence pending corroboration.",
            ))
            continue

        if any(source in STRONG_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.90),
                basis=_basis(sources),
                sources=sources,
                reason="Strong release-specific source evidence supports this mapped genre.",
            ))
            continue

        if "model_prior" in sources and any(source in MEDIUM_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.78),
                basis=_basis(sources),
                sources=sources,
                reason="Model taxonomy agrees with existing non-Last.fm metadata.",
            ))
            continue

        if "local_metadata" in sources and any(source in LASTFM_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.80),
                basis=_basis(sources),
                sources=sources,
                reason="Local metadata and Last.fm corroborate this specific mapped genre.",
            ))
            continue

        if "local_metadata" in sources and len(sources) >= 2:
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.80),
                basis=_basis(sources),
                sources=sources,
                reason="User-curated local tags corroborated by an independent source.",
            ))
            continue

        if sources == ["local_metadata"]:
            # Never-drop invariant (2026-06-12 LPVV incident): the user's own
            # file tags are retained — provisional publishes to observed_leaf.
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis=_basis(sources),
                sources=sources,
                reason="User-curated local file tag retained (never dropped) pending corroboration.",
            ))
            continue

        if sources == ["model_prior"] and sparse_release and score >= 0.58:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="model_prior+taxonomy",
                sources=sources,
                reason="Sparse release has a high-confidence mapped model taxonomy signal.",
            ))
            continue

        if sources == ["model_prior"] and has_non_lastfm_release_evidence and score >= 0.70:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="model_prior+taxonomy",
                sources=sources,
                reason="Confirmed release evidence exists; high-confidence model taxonomy signal is usable provisionally.",
            ))
            continue

        if all(s in AI_CHECK_TYPES for s in sources) and score >= 0.75:
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis=_basis(sources),
                sources=sources,
                reason="AI-enrichment evidence (metadata or accepted) supports this mapped genre.",
            ))
            continue

        medium_sources = [s for s in sources if s in MEDIUM_SOURCE_TYPES]
        if len(medium_sources) >= 2 and score >= 0.72:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis=_basis(sources),
                sources=sources,
                reason="Multiple independent metadata sources (MusicBrainz, Discogs, etc.) corroborate this genre.",
            ))
            continue

        if sources == ["musicbrainz"] and score >= 0.70:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis=_basis(sources),
                sources=sources,
                reason="MusicBrainz artist-level genre signal; provisional pending corroboration.",
            ))
            continue

        provisional.append(FusedGenreDecision(
            term=term,
            confidence=score,
            basis=_basis(sources),
            sources=sources,
            reason="Evidence mapped but below the corroboration bar; published at evidence confidence.",
        ))

    return HybridGenreReport(
        release_key=release_key,
        accepted_genres=accepted,
        provisional_genres=provisional,
        rejected_noise=rejected,
    )


def _decision_term(item: EvidenceTerm) -> str:
    if item.mapping_status not in {"mapped", "canonical", "alias"}:
        return ""
    term = (item.canonical_slug or item.term).strip().casefold()
    if term == "__empty__":
        return ""  # pipeline sentinel, not a genre
    if term == "lofi":
        return "lo-fi"
    return term


def _score(items: list[EvidenceTerm]) -> float:
    weighted = 0.0
    total_weight = 0.0
    for item in items:
        weight = SOURCE_WEIGHTS.get(item.source_type, 0.40)
        weighted += weight * max(0.0, min(1.0, item.confidence))
        total_weight += weight
    if total_weight == 0:
        return 0.0
    agreement_bonus = min(0.10, 0.03 * max(0, len({item.source_type for item in items}) - 1))
    return min(1.0, (weighted / total_weight) + agreement_bonus)


def _basis(sources: list[str]) -> str:
    ordered = [
        source
        for source in [
            "bandcamp_artist",
            "bandcamp_release",
            "bandcamp_label",
            "bandcamp_unknown",
            "official_release",
            "discogs",
            "musicbrainz",
            "local_metadata",
            "model_prior",
            "lastfm_tags",
        ]
        if source in sources
    ]
    extra = sorted(source for source in sources if source not in ordered)
    return "+".join(ordered + extra + ["taxonomy"])


def _rejected_noise_reason(term: str) -> str | None:
    return REJECTED_NOISE_TERMS.get(term.casefold())


def fuse_release_evidence(store: object, release: object) -> HybridGenreReport:
    """Fuse sidecar evidence + metadata.db artist/album genres for one release.

    Shared by the CLI commands and the analyze ``enrich`` stage. ``release``
    exposes ``release_key``, ``normalized_artist``, ``normalized_album``,
    ``album_id``, and ``existing_genres_by_source``.
    """
    evidence = collect_hybrid_evidence(store, release.release_key)

    _skip_prefixes = ("artist:lastfm", "album:lastfm")
    for source_key, genres in release.existing_genres_by_source.items():
        if any(source_key.startswith(p) for p in _skip_prefixes):
            continue
        parts = source_key.split(":", 1)
        if len(parts) != 2:
            continue
        level, src = parts
        if level == "artist" and ("musicbrainz" in src or "discogs" in src):
            # FIX 1 (2026-06-12 GENRE_DATA_QUALITY_FINDINGS §6): artist-level
            # MusicBrainz/Discogs genres describe the artist's WHOLE catalog,
            # not this specific release, and bleed wrong tags onto albums
            # (e.g. "garage rock" onto Arctic Monkeys' lounge album). Skip
            # them the same way artist:lastfm is already skipped below;
            # release/album/track-level MB/Discogs evidence is unaffected.
            continue
        if src == "file" and level in ("track", "album"):
            # The user's own embedded file tags — first-class evidence
            # (2026-06-12: the old blanket `track:` skip kept LPVV's correct
            # hardcore/post-punk/punk tags out of fusion entirely).
            source_type, conf = "local_metadata", 0.85
        elif level == "track":
            continue
        elif "musicbrainz" in src:
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
