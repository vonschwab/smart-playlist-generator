from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "ai-genre-refinement-v3-authoritative-source"
TAXONOMY_VERSION = "local-taxonomy-v1"

SYSTEM_INSTRUCTIONS = """You are refining album-level genre metadata for a local music library.

Return strict JSON only. Do not include markdown.

Use local metadata first. The local payload already contains deterministic file, MusicBrainz, Discogs, and local genre metadata. Treat that as baseline metadata to audit, not as web evidence to rediscover.

Do not perform broad web research. Do not use web search to rediscover MusicBrainz or Discogs data already present in the payload. Do not treat Last.fm, open user-tag clouds, streaming mood tags, streaming/storefront pages such as Audiomack/Qobuz/Spotify, Wikipedia/Wikidata genre summaries, generic SEO music pages, lyrics pages, scraped mirrors, review aggregators, or unsourced recommendation pages as authoritative genre evidence.

Primary authoritative sources, in priority order:
1. Official release page from the artist, label, publisher, or release owner.
2. Bandcamp release page, especially artist- or label-run Bandcamp pages.
3. Official artist website release/discography page.
4. Official label catalog page, release notes, press release, or shop page.
5. Official distributor page only when it clearly reproduces label/artist release metadata.
6. Liner notes, artist-provided release notes, or label-provided one-sheet text when included in the payload.

Specialist reviews, record shops, blogs, and archival pages are review-only context unless they clearly quote or reproduce artist/label release language. General music knowledge is only a secondary signal.

When supplied_source_evidence includes authoritative release URLs or supplied source tags, consider those explicit payload facts. Treat supplied Bandcamp tags as release-specific artist/label evidence, but still separate usable genres/styles from descriptors, instruments, locations, moods, formats, jokes, and SEO tags.

Never claim a source says something unless that source was actually supplied or returned in this request.

Your job is not to pick one genre. Preserve multi-genre signatures and add useful specificity while keeping valid broad parent genres. Rare and specific taste signals matter when supported.

Separate genres/styles from descriptors. Instrumental, live, Japanese, remastered, demo, compilation, female vocalist, 1990s, etc. are descriptors unless part of a recognized genre/scene term.

Keep valid existing genres. Prune only clearly wrong, noisy, duplicated, malformed, descriptor-only, joke/SEO-noise, or harmful tags. Do not prune broad tags merely because more specific tags can be added.

Every keep/add/prune/review recommendation must use one recommendation_basis value: authoritative_source, hybrid, local_metadata, model_knowledge, or review_context. For authoritative_source, hybrid, or review_context recommendations, include supporting_source_indexes.

If evidence is weak, lower confidence and set should_escalate true. Concise reasons are required. Do not provide chain-of-thought.

Auto-apply guidance:
- auto_apply_eligible can be true only when release_level_confidence is at least 0.85, evidence_quality is high, and the added genre confidence is at least 0.85.
- auto_apply_eligible requires direct support from an authoritative release-specific source or strong hybrid agreement between local metadata and authoritative source evidence.
- Never mark a genre auto_apply_eligible when the reason depends mainly on album title, track titles, artist stereotypes, label names, or common associations.
- Never mark broad parent genres, descriptors, disputed microgenres, country/language/era tags, mood/function tags, joke tags, SEO tags, review_context-only suggestions, local_metadata-only suggestions, or model_knowledge-only suggestions auto_apply_eligible.
- When in doubt, set auto_apply_eligible false and put the item in review_only_suggestions or uncertainty_notes.
- Set should_escalate true when authoritative release-specific sources cannot be found, evidence_quality is low or unknown, confidence is below 0.5, the release identity is ambiguous, or recommendations are mostly review-only.
"""


def build_prompt(payload: dict[str, Any]) -> str:
    compact_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return f"Album payload JSON:\n{compact_payload}"


def build_batch_request(
    custom_id: str,
    model: str,
    prompt: str,
    response_format: dict[str, Any],
    *,
    instructions: str = SYSTEM_INSTRUCTIONS,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": prompt}],
            "text": {"format": response_format},
        },
    }
