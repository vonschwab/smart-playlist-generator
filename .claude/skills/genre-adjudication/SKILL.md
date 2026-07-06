---
name: genre-adjudication
description: Use when the user pastes one or more album/release payloads for genre adjudication or classification — batches of JSON payloads with artist/album/evidence fields expecting structured genre JSON back — or asks to adjudicate, classify, or review the genre identity of specific releases in an interactive session.
---

# Genre adjudication (interactive sessions)

Dylan pastes batches of release payloads (typically 2–16) and expects structured JSON conforming to the pipeline's `album-adjudicator-response-v1` contract. **Source of truth:** `src/ai_genre_enrichment/album_adjudicator.py` (`ADJUDICATOR_INSTRUCTIONS`, `ADJUDICATOR_RESPONSE_SCHEMA`). If this skill and that file disagree, the .py file wins — update this skill.

## Output contract

Return ONE fenced JSON array, one element per payload, **in payload order**. Each element is one response object:

```json
{
  "release_key": "<echo the payload's album_id or release_key if present, else \"<artist> — <album>\">",
  "genres": [{"term": "shoegaze", "confidence": 0.9, "layer": "core"}],
  "facets": [{"term": "female vocals", "facet_type": "vocal"}],
  "escalate": false,
  "escalate_reason": "",
  "overall_confidence": 0.85,
  "warnings": []
}
```

- `layer` ∈ {`core`, `secondary`}. Core = primary identity (~2–4); secondary = real but lesser element. Total genres ~3–6; fewer for focused releases.
- `facet_type` ∈ {`mood`, `texture`, `instrumentation`, `production`, `era`, `region`, `function`, `vocal`, `scene`, `format`, `rhythm`} (the taxonomy's facet enum).
- `release_key` is an interactive-batch convenience for unambiguous mapping — the strict pipeline schema omits it. If the paste specifies its own output format, that wins.
- No prose between payloads' results, no per-genre rationale, no chain-of-thought. A short summary AFTER the JSON block is fine.

## Core rules (mirrors the pipeline prompt)

1. **Tight and specific, no broad parents.** State what the release ACTUALLY IS — the 3–6 genres a knowledgeable listener would name. Broad parents (rock, pop, jazz, electronic, hip hop, folk, indie rock, alternative rock, experimental) are derived downstream from the genre graph; never include them. Shoegaze, not "rock"; ethio-jazz, not "world music"; trip-hop, not "downtempo".
2. **Genres ≠ facets.** Mood/texture/instrumentation/production/era/region/function/vocal/scene/format/rhythm descriptors (instrumental, lo-fi, acoustic, orchestral, 1970s, japanese, live, female vocals, drone) go in `facets`, never `genres`.
3. **User file tags are ground truth.** Every SPECIFIC `user_file_tags` genre MUST appear in `genres`, OR set `escalate: true` and name the omitted tag in `escalate_reason`. Silently dropping a specific user file tag is the single worst error. Broad-parent file tags (e.g. "rock") may be dropped without escalating.
4. **This release, not the artist.** Source tags are often artist-level and identical across albums — give THIS release its own identity. Never infer genre from artist name, nationality, language, album-title aesthetics, or demographic cues alone.
5. **Escalate over guess.** Set `escalate: true` when the release identity is ambiguous, evidence is thin and you'd be guessing, or a correct file tag would be dropped. Lower `confidence`/`overall_confidence` for sparse evidence. An escalation is a good outcome, not a failure.
6. **No web search** unless Dylan explicitly asks; never claim an external source says something it wasn't shown to say.
7. **Verify before rejecting a term.** Before declaring a proposed term bogus, check `data/layered_genre_taxonomy.yaml` (grep the term and its aliases) — legitimate niche subgenres (e.g. Kankyo Ongaku) have been wrongly challenged. Rare > common when expressing taste (design principle 12). Canonical spelling is normalized downstream — use canonical names where known.

## Related

- Pipeline path (batch, non-interactive): `adjudicate`/`apply` stages — `docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md`.
- Where accepted genres land and how to read them back: the **genre-data-authority** skill.
- Adding/editing taxonomy terms the adjudication surfaces: the **taxonomy-growth** skill.
