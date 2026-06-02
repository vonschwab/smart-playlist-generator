# Sonic Neighborhood Validation & Diagnosis — design

**Date:** 2026-06-01
**Branch:** sonic-neighborhood-validation (off `genre-edge-safeguards`)
**Status:** approved — this is **PHASE 2** of the initiative.
**Origin:** `docs/SONIC_INITIATIVE_NOTES.md` — sonic is now the weakest playlist axis;
mirror the genre-embedding investigation that cracked the genre space (anisotropy/IDF).

> **PHASE 1 PRECEDES THIS.** Investigation on 2026-06-01 found that the production
> sonic path silently does **not** apply the intended 0.20/0.50/0.30 tower weighting
> (config/artifact/code drift; see [[project-sonic-tower-weights-inert]]), and applies
> PCA whitening twice (compressing cosines). Phase 1 fixes that drift by rebuilding the
> artifact in a `tower_weighted` variant and cleaning the runtime path. Phase 1 design +
> plan: `2026-06-01-sonic-tower-weighted-fix-design.md` /
> `docs/superpowers/plans/2026-06-01-sonic-tower-weighted-fix.md`. This audition (Phase 2)
> runs **after** Phase 1 and validates the corrected space by ear.

## Scope (this session)

**Validate + diagnose. Stop before implementing any fix.**

Determine whether the sonic embedding's top-cosine neighbors actually sound alike,
and land on a *confirmed* root cause for the two symptoms in the notes:

1. **Cosine compression** — candidate-pool sonic similarity tops out ~0.46–0.55,
   median ~0.21 (open question 1: real problem, or harmless scaling artifact?).
2. **Negative-S edges** in otherwise-good playlists (open question 2: scoring bug,
   or legitimately fine transitions that the `T` floor correctly lets through?).

Out of scope: changing weights, floors, the metric, whitening, or PCA. Those wait
for a separate spec once the root cause is confirmed.

## Ground-truth method

**Audition-heavy, qualitative.** The user listens and describes neighbors in free
text — descriptions, not numeric scores. Rationale: "3/5" hides *which dimension*
is wrong; "same hazy tone but way more energetic, different room" localizes the
fault to rhythm vs. timbre vs. harmony — exactly the decomposition under test.

An optional 4-tag verdict shorthand rides alongside the prose for aggregation:
`match` / `close` / `off` / `wrong`. Prose is the real payload; the tag is never
required.

Cheap automated cross-checks ride along for triangulation (same-album/same-artist
neighbor rank, cosine distributions) but are secondary to the ears.

## Part A — Neighbor computation (Python, read-only, production-faithful)

For each seed, compute top-**15** nearest neighbors in **multiple sonic spaces
simultaneously**. The multi-space split is the diagnostic lever: if timbre-only
neighbors sound right but the weighted blend doesn't → weighting/centering is the
fault; if *no* space sounds right → the features/PCA underneath are broken.

Spaces:
1. **Production transition metric** — centered + tower-weighted cosine
   (rhythm 0.20 / timbre 0.50 / harmony 0.30), what the beam actually scores.
2. **Per-tower** — rhythm-only, timbre-only, harmony-only.
3. **Raw full cosine** (uncentered, tower-weighted) — to expose what centering does.

Data source: the **artifact bundle** (`data/artifacts/beat3tower_32k/data_matrices_step1.npz`)
via `load_artifact_bundle` — same path the GUI/beam use. `X_sonic` and the
per-tower slices come from the bundle; `track_artists`/`track_titles` from the
bundle; `file_path` joined **read-only** from `data/metadata.db`. No DB clustering,
no Plex, no Last.fm.

### Seeds (17, locked)

Curated for **sonic diversity** — genre is the proxy; the set spans
acoustic↔electronic, sparse↔dense, smooth↔abrasive, and deliberately includes the
rhythm-forward and drone corners that stress the timbre-dominant 0.20/0.50/0.30
weighting.

| Sonic family | Anchor artists |
|---|---|
| Electronic / synthetic | Green-House, Boards of Canada, Autechre, Charli XCX |
| Acoustic / organic | Bill Evans, Jean-Yves Thibaudet, William Tyler, Elliott Smith |
| Guitar-band texture | Duster, Real Estate, Slowdive, Sonic Youth, Minor Threat |
| Groove / rhythm-forward | James Brown, J Dilla, Beyoncé |
| Drone / textural extreme | Grouper |

Seed *track* per artist: pick a representative track (highest-presence / most
on-genre). Exact track selection is an implementation detail; the harness records
which track was used.

## Part B — Audition server + blinded page

### Blinded merge (key design choice)

Per seed: take the **union** of top-15 across all spaces, dedup to unique tracks
(~35–50 per seed after overlap), present as **one shuffled list with space/rank
hidden**. The user describes each unique track once, unbiased by knowing which
space proposed it. On save, the harness re-attaches which space(s) ranked the track
and at what rank/cosine. This is what makes the notes *localizing* rather than just
"are neighbors good."

### Server (tiny, local, read-only)

- Streams library audio with **HTTP range support** (seeking required for audio).
- Serves one page per seed; a seed selector navigates between seeds.
- **Codec guard:** on startup, detect non-browser-playable formats (ALAC / Apple
  Lossless `.m4a`); flag affected tracks instead of serving a silent player.
  FLAC/MP3/AAC are fine. (Verify-early risk: survey the library's formats as the
  first implementation step.)
- `POST /save` persists notes after every entry — crash-safe and resumable.
- Read-only: never writes to `metadata.db` or any audio file.

### Page

- Seed track pinned at top for reference (player).
- Each neighbor: audio player + optional verdict tag (`match`/`close`/`off`/`wrong`)
  + free-text notes box.

### Capture file

`docs/run_audits/sonic_audition/<seed>.yaml`, append-only / resumable. Per entry:

- `track_id`, `artist`, `title`
- `verdict` (optional), `notes` (free text)
- `spaces`: which spaces ranked it, with rank + cosine (written on save, hidden in UI)

## Part C — Analysis / diagnosis

After each batch, an aggregation script answers:

1. **Which space is most faithful?** Verdict/notes rate per space. Does timbre-only
   beat the weighted blend? Does rhythm-only sound *wrong*? → localizes fault to
   weighting/metric vs. underlying features.
2. **Is cosine compression real or just scaling?** Correlate cosine magnitude vs.
   verdict. If ~0.45-cosine neighbors consistently sound dead-on, the ~0.5 ceiling
   is a harmless scale artifact (answers open Q1). If they sound unrelated, the
   space is genuinely flat.
3. **Are negative-S edges bad?** A dedicated audition set of the *actual* negative-S
   pairs from production runs (Real Estate: Torrey→Pixies `S=-0.15`, Built to
   Spill→Beach House `S=-0.11`, Melody's Echo Chamber→Peel Dream Magazine `S=-0.09`).
   User verdict answers open Q2 directly: bug or legitimate.
4. **Theme-mining** the free text for recurring failure modes (e.g. "right tone,
   wrong energy" recurring → rhythm under-weighted at 0.20).

**Output:** a findings doc under `docs/run_audits/sonic_audition/` stating the
confirmed root cause(s). End of scope. No fixes implemented.

## Phasing

The harness supports all 17 seeds, but the audition runs in **batches** (~700 total
audition items is a lot of listening). First batch = the sonic-corner seeds
(e.g. Grouper drone, James Brown rhythm-forward, Autechre harsh-IDM, Jean-Yves
Thibaudet pure-harmonic, Sonic Youth noise) to shake out the harness, codec guard,
and capture format, and to get early signal before committing to the full set.

## Components & boundaries

- `scripts/sonic_audition_build.py` — Part A: compute per-seed multi-space
  neighbors + negative-S pairs, emit a blinded manifest the server reads.
- `scripts/sonic_audition_serve.py` — Part B: local server (audio range-streaming,
  page render, `/save`).
- `scripts/sonic_audition_analyze.py` — Part C: aggregate notes → findings.
- Audition page assets (static HTML/JS/CSS) served by the server.
- Capture + findings under `docs/run_audits/sonic_audition/`.

Each is independently runnable and testable; they communicate through files (the
blinded manifest and the per-seed capture YAML), so the build/serve/analyze stages
are decoupled.

## Invariants honored

- `transition_weights` == `tower_weights` = rhythm 0.20 / timbre 0.50 / harmony 0.30
  (read from config, not hardcoded, so the "production transition" space matches the
  beam).
- `center_transitions=True` → the production space is centered cosine; raw-full is
  the explicit uncentered comparison.
- Read-only throughout: no writes to `metadata.db`, no writes/moves of audio files.
