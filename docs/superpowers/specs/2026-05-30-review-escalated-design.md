# review-escalated: CLI review workflow for escalated AI enrichment releases

## Context

The `run`/`run-one` subcommands call the AI to enrich album-level genres. When the AI is uncertain it sets `should_escalate: true` in its response, which stores the check in `ai_genre_release_checks` with `status = 'needs_review'` and the suggestions in `ai_genre_suggestions`. These suggestions are never auto-applied.

The existing `review` command is unrelated — it handles the tag-classification pipeline (`ai_genre_source_tags`). There is currently no way to act on escalated releases from the CLI.

## Goal

Add a `review-escalated` subcommand that walks through escalated releases one suggestion at a time, lets the user accept or reject each one, and applies the accepted decisions via `set_user_override` + `rebuild_enriched_genres_for_release`.

## Queue

Source: `ai_genre_release_checks WHERE status = 'needs_review'` joined to `ai_genre_suggestions WHERE suggestion_type IN ('add', 'prune')`.

- `keep` suggestions are skipped — the genre is already present, no action needed.
- `descriptor` suggestions are skipped — not genres.
- Rows are ordered by `release_key`, then `suggestion_type` so all suggestions for a release appear consecutively.

## Display (one suggestion at a time)

```
─── [3/79] Ali Chukwumah & His Peace Makers / Nigeria 70 ───
  conf: 0.70  |  evidence: medium
  uncertainty: "unclear release boundaries; afro-funk attribution contested"

  ADD:  afro-funk  (0.90)  [local_metadata]
  "The album incorporates afro-funk without clear representation elsewhere."

  context:  keep → african  •  highlife  |  prune → soukous

[A]ccept  [R]eject  [S]kip  [Q]uit
```

Fields shown:
- Header: `[idx/total] artist / album` with a separator line
- `conf` and `evidence` from the check row
- `uncertainty` from `uncertainty_notes` in `response_json` (up to 2 notes joined with `;`)
- Suggestion block: `ADD` or `PRUNE`, genre name, confidence, recommendation_basis, reason (truncated to 80 chars)
- Context: other actionable suggestions for this release (not the current one), summarised as `keep → ...` and `prune → ...`

## Keybindings

| Key | Action |
|-----|--------|
| A   | Accept suggestion |
| R   | Reject suggestion (skip, no change) |
| S   | Skip suggestion (same as R for now) |
| Q   | Quit — flush current release if any decisions were made, then exit |

## Decision mechanics

Decisions accumulate in memory per release. A release is flushed when:
- The first suggestion of the *next* release is reached, or
- The user presses Q

Flush sequence for a release:
1. `store.set_user_override(release_key, normalized_artist, normalized_album, genres_add, genres_remove)`
2. `store.rebuild_enriched_genres_for_release(release_key)`
3. `store.mark_check_complete(check_id)` — marks the check `status = 'complete'` so it leaves the queue

A release where all suggestions were rejected or skipped is still flushed and marked complete (the user reviewed it and decided nothing applies). A release the user quit before making any decision is *not* flushed and remains `needs_review`.

`set_user_override` replaces the full override row, so all accepted decisions for a release must be accumulated before writing.

## Subcommand signature

```
review-escalated  [--limit N] [--artist ARTIST] [--album ALBUM] [--release-key KEY]
```

`--limit` counts escalated releases (not suggestions). Filters match the discovery surface of `review`.

## New storage methods

### `get_escalated_queue(limit, release_key, artist, album) -> list[dict]`

```sql
SELECT
    c.check_id,
    c.release_key,
    c.normalized_artist,
    c.normalized_album,
    c.overall_confidence,
    c.evidence_quality,
    c.response_json,
    s.suggestion_id,
    s.suggestion_type,
    s.genre,
    s.confidence AS suggestion_confidence,
    s.reason,
    s.recommendation_basis
FROM ai_genre_release_checks c
JOIN ai_genre_suggestions s ON s.check_id = c.check_id
WHERE c.status = 'needs_review'
  AND s.suggestion_type IN ('add', 'prune')
  [AND c.normalized_artist = ?]   -- if --artist
  [AND c.normalized_album  = ?]   -- if --album
  [AND c.release_key       = ?]   -- if --release-key
ORDER BY c.release_key, s.suggestion_type, s.suggestion_id
```

`--limit` is enforced at the release level in the command loop (not via SQL LIMIT), so the last release is always complete.

### `mark_check_complete(check_id: int) -> None`

```sql
UPDATE ai_genre_release_checks SET status = 'complete' WHERE check_id = ?
```

## File changes

| File | Change |
|------|--------|
| `src/ai_genre_enrichment/storage.py` | Add `get_escalated_queue`, `mark_check_complete` |
| `scripts/ai_genre_enrich.py` | Add `review-escalated` parser entry, `cmd_review_escalated` function |

No new tables. No changes to existing methods.

## Error handling

- Empty queue → print "No escalated releases to review." and exit 0.
- `set_user_override` or `rebuild_enriched_genres_for_release` failure → print error, skip marking complete, continue to next release.
