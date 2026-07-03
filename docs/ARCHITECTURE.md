# Architecture

Orientation map of the system as it runs today. This is the **layer-1 map** — what the pieces
are and how they fit. Two companions go deeper:

- [`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`](TECHNICAL_PLAYLIST_GENERATION_FLOW.md) — the
  code-level walkthrough of a single generation (`file:line` through every phase).
- [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) — **why** the system is shaped this way: the
  experiments, the results, and what was tried and rejected. This doc states *what is*; that doc
  records *why*. Where a choice here looks arbitrary, the rationale doc has the evidence.

> **Reading the defaults in this repo.** Three layers set behavior, and they deliberately
> differ. (1) **Dataclass defaults** (`src/playlist/pier_bridge/config.py`) are the *rollback*
> baseline — every experimental lever is `False`/`off` here so that "no config" is always safe.
> (2) **`config.example.yaml`** is the *shipped* template — the validated stack turned **on**
> (this is what a new install copies). (3) **`config.yaml`** (gitignored) is the *live* config on
> one machine. This doc describes the **shipped** behavior and names the key so you can see the
> rollback. A few validated levers are on in the live config but not yet in the shipped template
> — those divergences are tracked in [`WIRING_STATUS.md`](WIRING_STATUS.md) and `CLEANUP_LIST.md`
> and called out where they matter.

---

## System overview

```
┌──────────────────────── Offline — the analyze pipeline (scripts/analyze_library.py) ─────────────────────────┐
│                                                                                                               │
│  scan → genres → discogs → lastfm → sonic → muq → adjudicate → apply → publish →                             │
│                                       │      │        │           │        │                                  │
│                                       ▼      ▼        ▼           ▼        ▼                                   │
│                                  DB feats  MuQ      Claude      apply     release_effective_genres            │
│                                  (energy) embeds   (Sonnet)    (no-LLM)   (metadata.db — the genre AUTHORITY) │
│                                                                                                               │
│         → genre-sim → artifacts → energy → popularity → genre-embedding → verify                              │
│                          │                                                    │                               │
│                          ▼  (auto-folds the MuQ sidecar into the artifact)    ▼                               │
│              data_matrices_step1.npz  ──►  X_sonic_muq (512-d) + genre + energy matrices                      │
└───────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                 │  warm path — never re-runs offline work, never hits the network
┌──────────────────────────── Runtime — generation ──────────┴──────────────────────────────────────────────┐
│                                                                                                             │
│  CLI (main_app.py)      Browser GUI (React) ──NDJSON──► worker (src/playlist_gui/worker.py)                 │
│        └───────────────────────────┬──────────────────────────────────┘                                    │
│                                     ▼                                                                        │
│      load artifact bundle ─► build candidate pool ─► pier-bridge beam search ─► anti-sag scoring            │
│      (four mode axes +          (sonic ⊗ genre ⊗ pace,       (per segment)        (anti-center, mini-piers) │
│       optional tag-steering)     + tag-steering pool lever)                                                  │
│                                     └────────────► weak-edge recovery cascade ─► M3U / Plex export          │
│                                                    (tail-DP → repair → delete)                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two halves, by deliberate design (a *local-first* architectural commitment — external APIs
enrich offline and never gate a generation):

- **Offline** does all the heavy, networked, irreversible work **once**: filesystem scan, genre
  fetch + Claude adjudication, audio embedding, artifact build. It writes the irreplaceable
  stores (`metadata.db`, the MuQ shards/sidecar).
- **Runtime** is warm and fast: load the prebuilt artifact, build a candidate pool, beam-search
  bridges, recover weak edges, export. No network, no writes to the irreplaceable stores.

---

## Offline: the analyze pipeline

`scripts/analyze_library.py` runs a fixed, ordered stage list. The **single source of truth** is
`ANALYZE_LIBRARY_STAGE_ORDER` (`src/playlist/request_models.py`) — the CLI, the worker, and the
GUI Tools panel all drive this same list, so there is one pipeline, not three. The canonical,
current stage list is in [`GOLDEN_COMMANDS.md`](GOLDEN_COMMANDS.md); the load-bearing stages:

| Stage | Does |
|-------|------|
| `scan` | Filesystem scan (incremental; `--force` = full) + orphan cleanup. |
| `genres` / `discogs` / `lastfm` | Fetch artist/release genres + top tags from MusicBrainz / Discogs / Last.fm into the enrichment sidecar (deterministic, no LLM). |
| `sonic` | Per-track DB audio features (BPM, onset rate, etc. — used by the pace axis; the tower sonic space itself was removed, see below). |
| `muq` | Extracts the **MuQ** sonic embedding into resumable shards → `muq_sidecar.npz` (`src/analyze/muq_runner.py`). |
| `adjudicate` | **Album-grain Claude (Sonnet) genre adjudication** — the production genre path. |
| `apply` | Deterministic (no-LLM) materialize of adjudications; escalations → human review queue. |
| `publish` | Writes `release_effective_genres` — **the genre authority**. |
| `genre-sim` | Builds the genre similarity matrix (graph-based). |
| `artifacts` | Builds `data_matrices_step1.npz`, then **auto-folds** the MuQ sidecar into it (`fold_muq`). |
| `energy` | Essentia arousal/valence/danceability sidecar (WSL-only; the pace/energy axis). |
| `popularity` | Last.fm top-tracks popularity sidecar (for popular-seeds / bangers). |
| `genre-embedding` | Dense PMI-SVD genre embedding sidecar (used by genre steering + tag-steering). |
| `verify` | Post-build sanity: manifest fingerprint, row-count parity, and **`X_sonic_variant` must equal the configured variant or it errors loudly**. |

**Resumability.** Each stage is fingerprint-gated against an `analyze_state` table — an unchanged
stage is skipped — plus per-stage pending logic. `--force` bypasses the gate; `scan` never skips.

**Genre enrichment runs on Claude via the Agent SDK — no API billing** (it uses the Claude Max
subscription). The production path is `adjudicate` + `apply` (album-grain, Sonnet, single-model).
A hard rule in the adjudicator contract: *a specific user-file-tag genre missing from the output
must escalate* — "silently dropping a specific user tag is the single worst error."

> **Why publish is the authority, not the raw tags.** Enrichment once made playlists *worse* — a
> Bandcamp label-storefront page overrode the user's correct file tags, and inferred hub-families
> ("rock", "indie") saturated the genre vector until it carried almost no signal. The fix was to
> make one published table the authority and exclude inferred hubs from the artifact vectors. See
> `DESIGN_RATIONALE.md` §"Genre graph as authority."

---

## Sonic feature space

The sonic similarity space is a **single learned embedding — MuQ** (`OpenMuQ/MuQ-MuLan-large`, a
512-d contrastive audio-text model). It is the *only* sonic space in the current system: the
earlier hand-built rhythm/timbre/harmony **towers**, and the **MERT** embedding that replaced
them, were both removed (archived at `data/archive/mert_2026/`). There is no runtime variant
choice — the active variant resolves to `muq` (`sonic_variant_override` defaults to `muq` when
unset). A configured-but-missing variant key still **raises at load** — a knob that can't act is
a startup error, never a silent fallback.

| Property | MuQ |
|----------|-----|
| Model | `OpenMuQ/MuQ-MuLan-large` (contrastive audio-text) |
| Dim | 512 |
| Clip | middle 10s @ 24 kHz |
| Post-processing | **`center_l2`** — mean-center over the library, then L2-normalize (*not* whitened; whitening was tested and hurt MuQ, which is already well-conditioned) |
| Build | `muq` stage → `src/analyze/muq_runner.py` → `muq_sidecar.npz` → `fold_muq_into_artifact.py` (auto-folded by the `artifacts` stage) |

> **Why MuQ, and why a contrastive model.** The hand-built towers were perceptually coarse (the
> dominant timbre tower rated Metallica ≈ Yeah Yeah Yeahs). A learned acoustic model (MERT) beat
> them by ~45–93% on cross-catalog neighbour QA — but still missed *fine* similarity. On trusted
> soundalike triplets a **contrastive** model does better: MuQ **84%** vs MERT 73%. The fix for
> fine similarity was a contrastive objective, not a bigger acoustic model — so MuQ replaced MERT
> outright, and the towers/MERT code was deleted. The full towers → MERT → MuQ arc is in
> [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md).

**Transitions.** Edge quality is a calibrated logistic of the cosine (`pier_bridge/vec.py`),
single-sourced so the beam scorer and the post-hoc reporter never diverge; calibration is keyed
to the variant (`TRANSITION_CALIB_BY_VARIANT` — `muq` centered at 0.594). It replaced a linear
`(x+1)/2` rescale that crushed the good-vs-bad edge gap from 72% to 8%. Because the space is a
single contrastive embedding, there is no rhythm/timbre/harmony reweighting — the old
`transition_weights` / `tower_weights` knobs were removed with the towers.

---

## Genre

Genres are a **graph on a real taxonomy**, not free-text tags.

- **Authority:** `release_effective_genres` (in `metadata.db`), written **only** by the `publish`
  stage, read **only** through `src/genre/authority.py`. Every genre consumer goes through that
  facade — including the new artist-tag reader `resolved_genres_for_artist` (below), which stays
  inside the authority tables and excludes `inferred_family` hub genres.
- **Taxonomy graph:** `data/layered_genre_taxonomy.yaml` (a living, GUI-grown artifact — ~465+
  active canonical genre nodes). `graph_similarity.py` scores genre pairs over this graph, with a
  **hub guard** that caps broad family/umbrella nodes so hub genres can't glue the whole matrix
  together (the IDF lesson, applied to the graph).
- **Metric = `max`.** Runtime genre-edge similarity is the *max* tag-pair similarity over the two
  tracks' canonical tags. A soft-cosine alternative was built and **rejected** — see
  `DESIGN_RATIONALE.md`.
- **Steering:** the beam routes a per-segment genre arc through the taxonomy graph
  (`genre_steering_source: taxonomy`, rebuild-robust). Two independent, live, *soft* demotions
  (never hard gates) keep off-axis tracks down: a raw-tag compatibility penalty in the candidate
  pool, and the taxonomy-graph pair-floor penalty at the beam edge.
- **Tag-steering (artist mode):** a soft, per-request lean toward genre tags the user picks from
  the seed artist's own published genres — see the levers under Runtime. It reuses the dense
  genre embedding and the authority reader; with no tags selected it is byte-identical to legacy.

The artifact bakes graph-resolved genres (`genre_source: graph`); GUI chips are ordered
most-specific → broadest (`granularity.py`).

---

## Runtime: the pier-bridge engine

Every playlist is built by the **pier-bridge** topology (the legacy greedy constructor is dead
code, unconditionally bypassed):

1. Seed tracks become fixed **piers**. For an artist, its catalog is medoid-clustered into piers
   (`artist_style`) — *when `artist_style.enabled`*; otherwise a legacy per-seed selection runs.
2. Piers are ordered for bridgeability; each adjacent pair defines a **segment**.
3. Each segment is filled by a **constrained beam search** through the MuQ space — per-step score
   = transition + bridge (harmonic-mean similarity to both piers) + soft genre/pace penalties.
4. **Anti-sag scoring** keeps interiors from collapsing into the generic local average
   (below).
5. A **weak-edge recovery cascade** lifts any remaining weak edges (below).
6. Export to M3U / Plex.

### Anti-sag scoring (collapse prevention)

Long bridges tend to **sag** into the dense, genre-blurred "average" region rather than
representing the seeds' character. Two levers counter this during selection (shipped on;
dataclass rollbacks off):

| Lever | Key | What it does |
|-------|-----|--------------|
| **Anti-center (SP2)** | `seed_character_mode: anti_center` @ `2.0` | *Scoring* fix: demotes interior candidates closer to the local pool centroid than to their own piers. |
| **Mini-piers (SP3)** | `mini_pier_enabled: true` | *Structural* fix: splits an over-long segment by pinning a high-character waypoint as an extra pier, so the beam structurally can't sag past it. |

> Anti-center nudges candidates; mini-piers guarantees the bridge can't drift past a character
> anchor. The reasoning — including the abandoned "density-floor" lever — is in
> `DESIGN_RATIONALE.md` §"Collapse prevention."

### The weak-edge recovery cascade

After assembly, a fixed **four-pass** cascade lifts weak/broken transition edges, escalating from
least- to most-destructive. It runs **once, top to bottom** (not a retry loop):

1. **variable bridge length** (add-only) — lengthens a weak segment to land more smoothly.
2. **tail-DP** — re-optimizes the last ≤2 interior slots of each segment (never-worse).
3. **edge repair** (break-glass) — swaps one interior track to lift a broken edge.
4. **edge delete** (remove-only, last resort) — deletes one interior track if that strictly
   improves the merged edge and won't breach a bystander artist's `min_gap`.

Knob-level detail (triggers, per-pass config, known limits like the deadzone and the
repair/reporter T-mismatch) is in [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md).

**Time budget.** `generation_budget_s` bounds a whole generation; the shipped default is `0`
(disabled — quality-first). There is a 90 s hard ceiling as a design target.

---

## The four mode axes (+ tag-steering)

Cohesion-vs-discovery is exposed as four independent axes. **`cohesion_mode` drives the beam; the
other three gate the candidate pool.**

| Axis | Controls | Levels |
|------|----------|--------|
| `cohesion_mode` | beam tightness (per-mode bridge floors + edge weights) | strict / narrow / dynamic / discover |
| `genre_mode` | genre pool gating + admission floor | strict / narrow / dynamic / discover / off |
| `sonic_mode` | sonic pool gating | strict / narrow / dynamic / off |
| `pace_mode` | rhythm/tempo gating (BPM + onset bands) | strict / narrow / dynamic / off |

All default to `dynamic`. The per-mode pier-bridge knobs are keyed by `cohesion_mode`.
**Tag-steering** rides the same request → `UIStateModel` → `policy.derive_runtime_config` channel
(web-only, like the axes): the pool lever blends the tag target into the dense admission centroid
(`tag_steering_pool_blend`), and the pier lever adds an on-tag bonus to medoid scoring
(`tag_steering_pier_weight`, active only when `artist_style.enabled`).

**Pace is embedding-independent** — it gates on BPM and onset-rate log-distance bands plus a soft
rhythm penalty, reading DB features, so it survives any sonic-embedding change. "Energy" is a
separate signal (trained *arousal*, not loudness), wired for admission rescue and an arc but off
by default.

---

## Browser GUI wiring

The browser GUI is the only front-end (the PySide6 desktop app was removed).

```
Browser (React SPA, web/dist) ──/api + /ws──► FastAPI (src/playlist_web/app.py)
                                                  └─ NDJSON over stdio ─► worker (src/playlist_gui/worker.py) ─► pier-bridge
```

- FastAPI owns the worker: one long-lived worker subprocess, NDJSON over stdio (16 MiB line
  limit). A dead/stalled worker maps to 503/504, never a bare 500.
- **Two top-level tabs:** `generate` and `tools`. Genre Review and Taxonomy are **sub-tabs** of
  the right-hand `AdvancedPanel`. The four axis sliders, the popular-seeds / bangers dropdowns,
  and the **tag-steering chips** (artist mode — the seed artist's own published genres, via
  `GET /api/genres/for_artist`) live in `GenerateControls`.
- **Policy layer** (`src/playlist_gui/policy.py::derive_runtime_config`) maps UI modes +
  steering tags → runtime config. **Only the web path goes through it** — the CLI sets mode
  strings directly, so a test/harness that bypasses policy will see modes as inert (a known
  false-negative trap).
- `tools/serve_web.py` rebuilds `web/dist` on every launch (unless `--no-build`), default port 8770.

---

## Key data stores

| Path | What | Irreplaceable? |
|------|------|----------------|
| `data/metadata.db` | Track DB + `release_effective_genres` (genre authority) | **Yes** — days to re-analyze. 2× confirm + backup before any write. |
| `data/ai_genre_enrichment.db` | Enrichment sidecar: adjudications, escalation + review queues, taxonomy decisions | Regenerable but costly. |
| `data/artifacts/beat3tower_32k/data_matrices_step1.npz` | The generation artifact: `X_sonic_muq` + genre + energy matrices | Rebuildable from the sidecars. |
| `.../muq_shards/` + `muq_sidecar.npz` | MuQ embeddings | **Yes** in practice — a full re-scan is expensive. |
| `data/archive/mert_2026/` | Archived MERT shards/sidecar/transform + the 2DFTM harmony sidecar | Historical rollback material (kept, not deleted). |
| `data/layered_genre_taxonomy.yaml` | The genre taxonomy graph | Living artifact, GUI-editable. |

---

## Configuration model

`config.yaml` is gitignored — copy it from `config.example.yaml`. Behavior resolves through the
three layers described at the top. Full key reference: [`CONFIG.md`](CONFIG.md). Knob-by-knob
tuning: [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md). Two validated levers are on
in the live config but not yet in the shipped template (tracked in `CLEANUP_LIST.md`): the
`edge_repair` cascade pass, and `artist_style.enabled` (so the shipped template runs the legacy
per-seed pier path, not the medoid clustering).

## Extension points

- **Add an export format** — implement an exporter alongside `src/m3u_exporter.py` and call it
  from the generation tail in `main_app.py` / the worker.
- **Add a sonic variant** — extract a sidecar, add a `fold_<variant>_into_artifact.py`, wire its
  auto-fold into the `artifacts` stage, and add its transition calibration to
  `TRANSITION_CALIB_BY_VARIANT`. The load-time override + missing-key-raises wiring is generic.
- **Add / change a mode behavior** — edit the presets in `src/playlist/mode_presets.py` (the
  single source for mode-driven gates/weights).
