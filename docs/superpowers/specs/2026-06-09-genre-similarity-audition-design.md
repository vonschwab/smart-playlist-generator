# Genre Similarity Audition Harness — Design

**Date:** 2026-06-09
**Status:** Approved design, pending implementation plan
**Related:** `src/genre/graph_similarity.py` (Stage 2 of taxonomy integration), `docs/superpowers/plans/2026-06-02-sonic-audition-harness.md` (the structural model)

## Purpose

Validate the new **graph-derived** genre-similarity matrix against the **legacy co-occurrence** matrix by blind human judgment, the same way the sonic audition harness validated the corrected tower-weighted sonic space. Two questions, answered in a single blind rating pass:

1. **Is the graph matrix better than co-occurrence?** For a seed genre, are the graph's top neighbors rated more genuinely related than co-occurrence's top neighbors?
2. **Is the graph correct in absolute terms?** Does the graph discriminate real neighbors from random genres at all?

Both fall out of one rating pass that pools candidates from three provenances — `graph`, `cooccurrence`, `decoy` — blinds them, and slices the verdicts by provenance afterward. This mirrors the sonic harness's "5 spaces" pattern, where one neighbor could belong to multiple spaces and analysis sliced verdicts by space.

## Why a blind pass (not a side-by-side)

If the rater can see which source produced a candidate, the cleaner-looking source wins on presentation bias. Blinding to provenance — and re-attaching it only server-side at capture time — keeps the judgment about *relatedness*, not about which algorithm we're rooting for. The `decoy` provenance is the negative control: if graph neighbors aren't rated clearly above random genres, the matrix has no discriminative power regardless of how it compares to co-occurrence.

## Architecture

Three file-communicating scripts plus a static page and unit tests, mirroring the sonic harness minus audio streaming.

| File | Role |
|---|---|
| `scripts/genre_audition_build.py` | Pool graph + co-occurrence + decoy candidates per seed, blind/shuffle, write JSON manifests |
| `scripts/genre_audition_page.html` | Rating UI: seed card + blinded candidate cards, verdict radios, notes, auto-save |
| `scripts/genre_audition_serve.py` | HTTP server: blinded manifest API + progress API + YAML verdict capture (no audio) |
| `scripts/genre_audition_analyze.py` | Expand verdicts by provenance, compute contrasts, write `findings.md` |
| `tests/unit/test_genre_audition_build.py` | Tests for manifest builder |
| `tests/unit/test_genre_audition_serve.py` | Tests for server helpers |
| `tests/unit/test_genre_audition_analyze.py` | Tests for analysis aggregation |

**Output dir:** `docs/run_audits/genre_audition/` — gitignored, created at runtime. Per-seed `<slug>_manifest.json`, `index.json`, `<slug>_capture.yaml`, `findings.md`.

**Tech stack:** Python 3.11, NumPy, PyYAML, sqlite3 (stdlib), http.server (stdlib), vanilla JS/HTML. No new pip installs.

## Data sources

| Source | Path | Keys | Vocabulary |
|---|---|---|---|
| Graph (under test) | `data/genre_similarity_graph.npz` | `genre_vocab`, `S`, `stats` | 408 canonical taxonomy names |
| Co-occurrence (incumbent) | `data/artifacts/beat3tower_32k/genre_similarity_matrix.npz` | `genre_vocab`, `cooc`, `S`, `stats` | 801 raw library tokens (incl. `__EMPTY__`, compound, multilingual) |
| Vocab bridge | `src/genre/graph_adapter.py` → `canonicalize_tag()` | — | maps raw token → canonical / alias / facet / rejected / unknown |
| Example artists | `data/metadata.db` (read-only) | `track_genres`, `artist_genres`, `tracks` | per-genre representative artists |

Both matrices are symmetric with a unit diagonal. `S[i]` is the neighbor row for vocab term `i`. Co-occurrence `S` is Jaccard; graph `S` is the recipe in `graph_similarity.py`.

## Build flow (`genre_audition_build.py`)

For each seed (a canonical genre name from the graph vocab):

1. **Seed validation.** Confirm seed ∈ graph vocab; else print `SKIP` and record in the index. The `--seeds` flag overrides the default stratified list.
2. **Co-occurrence row resolution.** Find the seed's row in the co-occurrence matrix:
   - exact match on the canonical name (works for clean tokens like `rock`, `indie rock`, `jazz`);
   - else canonicalize every co-occurrence token via `canonicalize_tag()` and pick the token resolving to this canonical seed with the highest co-occurrence count (`counts`/diagonal mass);
   - no match → the seed runs with `graph` + `decoy` provenances only; record `cooc_token: null` in the manifest.
3. **Candidate collection.**
   - `graph`: top-K neighbors of the seed row in graph `S` (exclude self), default K=10.
   - `cooccurrence`: top-K neighbors of the resolved co-occurrence row (exclude self, exclude `__EMPTY__`), shown as **raw tokens**, default K=10.
   - `decoy`: N≈3 canonical genres sampled from the **low/zero graph-similarity tail** to the seed, excluding any name already proposed by graph or co-occurrence. Deterministic sample (seeded by seed name).
4. **Union into cards.** Key candidates by a normalized name (`graph_adapter.normalize_taxonomy_name` semantics). A candidate proposed by multiple sources collapses into **one card carrying all provenances**. Each card's hidden `space_data` holds, per source, the `rank` and `sim`. Co-occurrence raw tokens that do not normalize-collide with a graph candidate remain their own cards (token hygiene stays visible as a finding).
5. **Example artists.** Look up 2-3 representative artists for the seed and each candidate (see "Example-artist lookup" below). Attach as `artists: [...]`.
6. **Blind + write.** Deterministic shuffle (RNG seeded by seed name). Write `<slug>_manifest.json`:
   - `seed`: `{genre, artists}`
   - `neighbors`: `[{name, artists}]` — **no provenance, no sim, no rank**
   - `space_data`: `{name: {graph?: {rank, sim}, cooccurrence?: {rank, sim}, decoy?: {}}}` — kept separate, never sent to the blinded client
   - `cooc_token`: the resolved raw token or `null`
   Append `{slug, genre}` to `index.json`.

### Example-artist lookup

The two matrices use different vocabularies, so artist lookup must match the term's native vocabulary:

- **Graph / decoy candidates and the seed** are canonical names. `track_genres.genre` and `artist_genres.genre` store raw source tags, not canonical names, so a canonical name may not match directly. Strategy: query `artist_genres` for an exact genre match first; if empty, fall back to the set of raw tokens that `canonicalize_tag()` maps to this canonical name and query those. Pick the 2-3 artists with the most tracks in that genre.
- **Co-occurrence candidates** are raw tokens already in `track_genres`/`artist_genres` — query the token directly.
- No artists found → `artists: []`; the page renders "(no example artists)".

All lookups are read-only (`mode=ro` URI), batched under the SQLite 900-variable limit. **No writes to `metadata.db` ever.**

## Serve flow (`genre_audition_serve.py`)

Adapted from `sonic_audition_serve.py`, with the audio endpoints removed:

- `GET /` → 302 to the first seed.
- `GET /seed/<slug>` → the page HTML with the seed index injected.
- `GET /api/manifest/<slug>` → the manifest **with `space_data` and `cooc_token` stripped** (blinded).
- `GET /api/progress/<slug>` → `[{name, verdict, notes}]` from the capture file.
- `POST /api/save` → append/update one entry in `<slug>_capture.yaml`, **re-attaching the hidden `space_data` provenance server-side** so captures are self-describing for analysis.

Capture entry shape (keyed by candidate `name`, append-or-update like the sonic harness):
```yaml
entries:
  - name: shoegaze
    verdict: same        # same | related | loose | unrelated | ""
    notes: ""
    saved_at: 2026-06-09T...
    spaces:              # provenance re-attached server-side
      graph: {rank: 1, sim: 0.82}
      cooccurrence: {rank: 3, sim: 0.41}
```

## Page (`genre_audition_page.html`)

The sonic page stripped of `<audio>`:

- Top bar: seed `<select>` + prev/next nav.
- Progress bar: `<rated> / <total>`.
- Seed section: seed genre name + its example artists.
- Candidate cards: genre name, example artists, a 4-way verdict radio (`same` / `related` / `loose` / `unrelated`), a notes textarea, auto-save on change/blur, colored left-border by verdict.
- No provenance, sim, or rank is ever rendered — the client only ever sees the blinded manifest.

Verdict → score for analysis: `same`=3, `related`=2, `loose`=1, `unrelated`=0.

## Analyze flow (`genre_audition_analyze.py`)

Load every `*_capture.yaml`. For each rated entry, **expand by provenance**: an entry whose `spaces` has both `graph` and `cooccurrence` contributes one observation to each. Then write `findings.md`:

1. **Verdict distribution by provenance** — count of each verdict for `graph`, `cooccurrence`, `decoy`.
2. **Mean verdict score by provenance** — the headline numbers.
3. **Graph vs co-occurrence (Q1)** — mean-score delta overall, plus a paired view restricted to candidates both sources proposed (same card, both provenances) where available.
4. **Graph vs decoy (Q2)** — graph mean should dominate decoy mean; report the gap. A small gap is a red flag for the matrix.
5. **Similarity ↔ verdict correlation per source** — Pearson r between `sim` and verdict score for `graph` and `cooccurrence` (does higher similarity track higher rated relatedness?). Needs ≥3 points per source.
6. **Callout lists** —
   - *Graph neighbors rated `unrelated`*: candidate bad edges → feed the SP3a growth/QA loop.
   - *Co-occurrence-only neighbors rated `same`/`related`*: candidate gaps the graph missed.
   - *Notable notes*: rated entries with non-empty notes, sorted by score.

## Seed set (stratified ~12)

Validated against the graph vocab at build time; misses are `SKIP`ped and reported. Overridable via `--seeds`.

- **Broad / hub-prone** (stress the hub guard): `rock`, `electronic`, `pop`, `jazz`
- **Mid-level**: `indie rock`, `house`, `ambient`, `post-punk`
- **Niche / micro** (rare-genre neighbor quality): `slowcore`, `shoegaze`, `witch house`, `drone`

These are confirmed present or near-present in both vocabularies from inspection; the builder reports any that miss the graph vocab so the list can be adjusted.

## Edge cases

- Seed not in graph vocab → `SKIP` + index note.
- Seed has no co-occurrence row → graph + decoy provenances only; `cooc_token: null`.
- Genre with no library artists → `artists: []`, page shows "(no example artists)".
- Decoy tail smaller than N → take what exists.
- Co-occurrence `__EMPTY__` token → always excluded from neighbor lists.

## Testing

Unit tests mirror the sonic harness's three test files:

- **build**: `compute_neighbor_rows` returns top-K per source; decoys are disjoint from graph/cooc and drawn from the low tail; `build_seed_manifest` is blinded (`space_data` separate, neighbors carry no provenance); union dedupes by normalized name and merges provenances; `_slug` behavior; returns `None`/`SKIP` for unknown seed.
- **serve**: `_append_capture_entry` creates/updates/adds (keyed by `name`); blinded manifest API omits `space_data` and `cooc_token`.
- **analyze**: provenance expansion (a graph+cooc entry counts for both); mean-score-by-provenance; sim↔verdict correlation row extraction; graceful handling of entries with no `spaces`.

Plus a 3-seed end-to-end smoke run: `build --seeds "shoegaze" "rock" "jazz"` → start server → confirm blinded manifest API → rate a couple → `analyze` writes `findings.md`.

## Out of scope

- Audio playback / audio-grounded transition judgment (option #3 from brainstorming) — deferred; this harness is genre-relationship judgment only.
- The dense PMI-SVD embedding (`src/genre/pmi_svd.py`) as a third comparison source — the architecture leaves room (provenance is a set), but the initial build compares only `graph` vs `cooccurrence` vs `decoy`.
- Any change to the production similarity source flag (`genre_similarity.source`) — this harness only reads matrices; it does not wire results into generation.
