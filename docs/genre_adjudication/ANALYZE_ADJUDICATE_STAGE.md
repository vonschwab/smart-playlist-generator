# Operator note: `adjudicate` and `apply` stages in `analyze_library.py`

SP1 replaced the legacy `enrich` stage with two new stages — `adjudicate` and `apply` — which are
now part of the **default stage order**.

---

## Default stage order

```
scan → genres → discogs → lastfm → sonic → mert → adjudicate → apply → publish → genre-sim → artifacts → genre-embedding → verify
```

The `enrich` stage (OpenAI-era web scraping → signature materializer) is **opt-in legacy**. Run it
explicitly with `--stages enrich` if needed; it is no longer in the default sequence.

---

## What each stage does

### `adjudicate`

Calls Claude (Sonnet by default) once per album that has no checkpoint yet. For each album it
assembles evidence (track count, existing file tags, Last.fm tags, MusicBrainz data, Discogs,
taxonomy graph) and asks the model for a clean canonical genre list with confidences.

Results are written to the `adjudications` table in `data/ai_genre_enrichment.db` with
`status='complete'`. The stage is **incremental** — albums that already have a `complete`
checkpoint are skipped on re-runs.

The stage pauses automatically after 8 consecutive API failures (likely a usage-limit wall). Re-run
`analyze_library.py` to resume.

### `apply`

Deterministic, no LLM calls. Reads every `status='complete'` row from the adjudications table and
either:

- **Materializes** to the sidecar (`genre_graph_release_genre_assignments` — the layered assignments that `publish()` reads into `release_effective_genres`) — for
  non-escalated results; or
- **Enqueues** to the `adjudication_escalations` table — for results where `escalate=true`.

A result is escalated when the model is uncertain, evidence is thin, or a user file tag would be
silently dropped. Escalated albums are **held back** from the sidecar and do not contribute to
genre authority until a human resolves them (see below).

---

## Model selection

The default model is **`sonnet`** (claude-sonnet-4-x). Override with:

```bash
python scripts/analyze_library.py --adjudicate-model haiku
python scripts/analyze_library.py --adjudicate-model opus
```

The `--model` flag continues to control the legacy `enrich` stage (LLM reranking of scraped tags)
and is unrelated to adjudication.

---

## Escalation hold + queue

Albums that trigger `escalate=true` in the model response are held in the
`adjudication_escalations` table (`status='pending'`). They do **not** affect `release_effective_genres`
until a human decides.

File-tag floor guarantee: if the model would drop a specific user file tag (e.g. `"shoegaze"`) and
does not set `escalate=true`, the post-processor forces escalation automatically. Broad parent tags
(e.g. `"rock"`) may be dropped without escalation.

### Clearing the queue

```bash
python scripts/research/review_escalated.py
```

Interactive review: for each pending escalation the CLI shows the proposed genres, existing file
tags, and escalation reason. Commands:

| Input | Effect |
|-------|--------|
| `accept` | Write proposed genres to sidecar as-is |
| `edit genre a, genre b, ...` | Override with the supplied list, then write |
| `reject` | Leave album's existing authority untouched |
| `skip` | Defer to next session |
| `quit` | Exit immediately |

---

## One-time backfill import

The Pass-1 / Pass-2 bulk runs write results to a separate DB (`data/adjudication_pass1.db`).
Import them into the sidecar before running the pipeline so the `adjudicate` stage inherits all
completed work and does not re-call the LLM:

```bash
python scripts/research/import_backfill_adjudications.py \
    --src data/adjudication_pass1.db \
    --sidecar data/ai_genre_enrichment.db
```

Run this **once**, after the backfill completes and before the first live `analyze_library.py` run
with the new stage order. Safe to re-run — rows are upserted on `(album_id, prompt_version)`.

---

## Typical operator sequence (post-merge)

```bash
# 1. Import backfill work (one-time)
python scripts/research/import_backfill_adjudications.py \
    --src data/adjudication_pass1.db --sidecar data/ai_genre_enrichment.db

# 2. Run the pipeline (adjudicate skips already-done albums)
python scripts/analyze_library.py

# 3. Review held escalations
python scripts/research/review_escalated.py

# 4. Re-run publish to incorporate newly accepted escalations
python scripts/analyze_library.py --stages publish
```

---

## Reviewing escalations in the GUI

The **Genre Review** tab in the web UI is the recommended interface for working through the
escalation queue once `apply` has populated it.

### Starting the server

```bash
python tools/serve_web.py
```

Open `http://localhost:8770` and click the **Genre Review** tab.

### What the panel shows

The panel lists pending **album-level escalations** — one card per album — not the tag-grain review
terms from the old queue. Each card shows:

- Album / artist
- Proposed canonical genres (may be empty if SP1 canonicalization produced no matches — use **Edit**)
- Escalation reason (uncertainty, thin evidence, file-tag preservation)
- Existing file tags that would be affected

### Per-album actions

| Action | Keyboard | Effect |
|--------|----------|--------|
| **Accept** | `A` | Approve proposed genres as-is |
| **Edit** | — | Reveal a comma-separated genre input; type replacements, then confirm |
| **Reject** | `R` | Leave album's existing authority untouched |

Decisions are recorded in-session. The panel advances to the next pending escalation automatically
after Accept or Reject.

### Publishing decisions

Click **Publish decided (K)** (or press `K`) when you are ready to commit the session's decisions.
The handler:

1. Creates a timestamped backup of `data/metadata.db` before writing.
2. Calls `publish()` to write accepted/edited genres into `release_effective_genres`.

Rejected albums are left untouched. Escalations resolved this way move to `status='decided'` and
will not reappear in the queue.

### Artifact rebuild after publishing

Publishing updates the database but **does not update the generation artifact**. Run the artifact
rebuild before the next playlist generation:

```bash
python scripts/fold_2dftm_into_artifact.py
```

or the full pipeline stage:

```bash
python scripts/analyze_library.py --stages artifacts
```

### Restart trap

- **After any edit to `src/playlist_gui/worker.py`** — restart `python tools/serve_web.py`; the
  worker process is spawned at server startup and will not pick up changes otherwise.
- **After any edit to `web/src/`** — rebuild `web/dist` with `npm --prefix web run build` (or
  `npm --prefix web run dev` for live-reload during development); the server serves the pre-built
  dist directory.
