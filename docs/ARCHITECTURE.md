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
> (2) **`config.example.yaml`** is the *shipped default* — the validated stack turned **on**
> (this is what a new install copies, per the project rule "activate fixes, never default to
> legacy"). (3) **`config.yaml`** (gitignored) is the *live* config on one machine. This doc
> describes the **shipped default** and always names the key so you can see the rollback.
> The live-vs-shipped delta is tracked in [`WIRING_STATUS.md`](WIRING_STATUS.md).

---

## System overview

```
┌──────────────────────── Offline — the analyze pipeline (scripts/analyze_library.py) ─────────────────────────┐
│                                                                                                               │
│  scan → genres → discogs → lastfm → sonic → mert → adjudicate → apply → publish →                            │
│                                       │       │        │           │        │                                 │
│                                       ▼       ▼        ▼           ▼        ▼                                  │
│                                  tower feats  MERT   Claude      apply     release_effective_genres           │
│                                  (rollback)  embeds  (Sonnet)   (no-LLM)   (metadata.db — the genre AUTHORITY)│
│                                                                                                               │
│         → genre-sim → artifacts → energy → popularity → genre-embedding → verify                              │
│                          │                                                    │                               │
│                          ▼  (auto-folds MERT / 2DFTM / MuQ sidecars back in)  ▼                               │
│              data_matrices_step1.npz  ── + sidecars ──►  variant matrices + genre + energy                    │
│              (X_sonic_muq | X_sonic_mert | X_sonic_tower_weighted, X_genre_*, ...)                            │
└───────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                 │  warm path — never re-runs offline work, never hits the network
┌──────────────────────────── Runtime — generation ──────────┴──────────────────────────────────────────────┐
│                                                                                                             │
│  CLI (main_app.py)      Browser GUI (React) ──NDJSON──► worker (src/playlist_gui/worker.py)                 │
│        └───────────────────────────┬──────────────────────────────────┘                                    │
│                                     ▼                                                                        │
│      load artifact bundle ─► build candidate pool ─► pier-bridge beam search ─► collapse-prevention         │
│      (four mode axes gate)     (sonic⊗genre⊗pace)      (per segment)             (var-bridge, anti-center,   │
│                                                                                   mini-piers, roam, repair)  │
│                                     └─────────────────► edge repair ─► M3U / Plex export                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two halves, by deliberate design (a *local-first* architectural commitment — external APIs
enrich offline and never gate a generation):

- **Offline** does all the heavy, networked, irreversible work **once**: filesystem scan, genre
  fetch + Claude adjudication, audio embedding, artifact build. It writes the irreplaceable
  stores (`metadata.db`, the MERT/MuQ shards).
- **Runtime** is warm and fast: load the prebuilt artifact, build a candidate pool, beam-search
  bridges, repair, export. No network, no writes to the irreplaceable stores.

---

## Offline: the analyze pipeline

`scripts/analyze_library.py` runs a fixed, ordered stage list. The **single source of truth** is
`ANALYZE_LIBRARY_STAGE_ORDER` (`src/playlist/request_models.py`) — the CLI, the worker, and the
GUI Tools panel all drive this same list, so there is one pipeline, not three.

| Stage | Does |
|-------|------|
| `scan` | Filesystem scan (incremental; `--force` = full) + orphan cleanup. |
| `genres` | MusicBrainz artist/release genres for tracks still missing them. |
| `discogs` | Discogs release/master genres + styles per album (needs `DISCOGS_TOKEN`). |
| `lastfm` | Last.fm top tags → enrichment sidecar (deterministic classification only, no LLM). |
| `sonic` | beat3tower hand-built sonic features (rhythm/timbre/harmony towers — the rollback space). |
| `mert` | MERT embedding → resumable shards → `mert_sidecar.npz`. |
| `adjudicate` | **Album-grain Claude (Sonnet) genre adjudication** — the production genre path. |
| `apply` | Deterministic (no-LLM) materialize of adjudications; escalations → human review queue. |
| `publish` | Writes `release_effective_genres` — **the genre authority**. |
| `genre-sim` | Builds the genre similarity matrix (graph-based). |
| `artifacts` | Builds `data_matrices_step1.npz`, then **auto-folds** the 2DFTM, MERT, and MuQ sidecars back in. |
| `energy` | Essentia arousal/valence/danceability sidecar (WSL-only; the pace/energy axis). |
| `popularity` | Last.fm top-tracks popularity sidecar (for popular-seeds / bangers). |
| `genre-embedding` | Dense PMI-SVD genre embedding sidecar (legacy steering source). |
| `verify` | Post-build sanity: manifest fingerprint, row-count parity, and **`X_sonic_variant` must equal the configured variant or it errors loudly**. |

**Resumability.** Each stage is fingerprint-gated against an `analyze_state` table — an unchanged
stage is skipped — plus per-stage pending logic (Discogs skips albums already fetched, MERT skips
shard-manifest ids, etc.). `--force` bypasses the gate; `scan` never skips.

**Genre enrichment runs on Claude via the Agent SDK — no API billing** (it uses the Claude Max
subscription, `permission_mode=dontAsk`). The production path is `adjudicate` + `apply`
(album-grain, Sonnet, single-model). A hard rule in the adjudicator contract: *a specific
user-file-tag genre missing from the output must escalate* — "silently dropping a specific user
tag is the single worst error." A legacy tag-grain `enrich` stage still exists but is **opt-in
and excluded from the default run**.

> **Why publish is the authority, not the raw tags.** Enrichment once made playlists *worse* —
> a Bandcamp label-storefront page overrode the user's correct file tags, Last.fm tag-fetch by
> artist name cross-contaminated identities, and inferred hub-families ("rock", "indie")
> saturated the genre vector until it carried almost no signal. The fix was to make one
> published table the authority and to exclude inferred hubs from the artifact vectors. See
> `DESIGN_RATIONALE.md` §"Genre graph as authority."

---

## Sonic feature space

The sonic similarity space is a **single learned embedding**, selectable at load time. There are
three variants baked into the artifact; the active one is chosen by
`artifacts.sonic_variant_override` (which wins over the artifact-declared `X_sonic_variant`).
**A configured-but-missing variant key raises at load** — a knob that can't act is a startup
error, never a silent fallback.

| Variant | Dim | What it is | Role |
|---------|-----|-----------|------|
| **MuQ** | 512 | `MuQ-MuLan-large`, a **contrastive** audio-text embedding | The intended default (see below) |
| **MERT** | 768 | `MERT-v1-95M` self-supervised acoustic embedding, `whiten_l2` post-processed | Predecessor / reproducible default |
| tower blend | 163 | hand-built rhythm(10) + timbre(57) + 2DFTM-harmony(96), weighted 0.20/0.50/0.30 | Deep rollback |

> **Why learned embeddings, and why MuQ.** The hand-built towers were perceptually coarse — the
> dominant timbre tower rated Metallica ≈ Yeah Yeah Yeahs, capping playlist quality regardless of
> tuning. MERT (a learned acoustic model) beat the towers by ~45–93% on cross-catalog neighbour
> QA. But MERT still misses *fine* similarity; on trusted soundalike triplets a **contrastive**
> model does better (MuQ **86–89%** vs MERT 73% vs CLAP 68%) — the fix for similarity is a
> contrastive objective, not a bigger acoustic model. See `DESIGN_RATIONALE.md`.

**The MERT/MuQ split, precisely.** The transition scorer's calibration is variant-aware
(`transition_metrics.TRANSITION_CALIB_BY_VARIANT` — MERT centre 0.32, MuQ centre 0.594) because
the two spaces have different cosine bands; flipping variants without re-calibrating would
saturate the beam's sigmoid. The `artifacts` stage auto-folds whichever sidecar the override
selects. **Caveat (current gap):** the MuQ *fold* is on master (`scripts/fold_muq_into_artifact.py`),
but the MuQ *sidecar extraction* (the scan that produces `muq_sidecar.npz`) is not yet — so a
fresh clone can run MuQ only if it already has the sidecar. That is why `config.example.yaml`
ships with `sonic_variant_override` **commented** (MERT is the safe template default) even though
MuQ is the live default. Tracked in [`WIRING_STATUS.md`](WIRING_STATUS.md).

**Transitions.** Edge quality is a calibrated logistic of the cosine (`pier_bridge/vec.py`),
single-sourced so the beam scorer and the post-hoc reporter never diverge. It replaced a linear
`(x+1)/2` rescale that crushed the good-vs-bad edge gap from 72% to 8%.

---

## Genre

Genres are a **graph on a real taxonomy**, not free-text tags.

- **Authority:** `release_effective_genres` (in `metadata.db`), written **only** by the `publish`
  stage, read **only** through `src/genre/authority.py`. Every genre consumer goes through that
  facade.
- **Taxonomy graph:** `data/layered_genre_taxonomy.yaml` (~465 active canonical genre nodes over
  ~1000 records — a living, GUI-grown artifact). `graph_similarity.py` scores genre pairs over
  this graph, with a **hub guard** that caps broad family/umbrella nodes so hub genres can't glue
  the whole matrix together (the IDF lesson, applied to the graph).
- **Metric = `max`.** Runtime genre-edge similarity is the *max* tag-pair similarity over the two
  tracks' canonical tags. A soft-cosine alternative was built and **rejected** — once the sonic
  space dominates selection, `max`'s coarse "share a close tag?" catches the egregious disjoint
  edges just as well. See `DESIGN_RATIONALE.md`.
- **Steering:** the beam routes a per-segment genre arc through the taxonomy graph
  (`genre_steering_source: taxonomy`, on `X_genre_raw`, rebuild-robust). The legacy `dense`
  PMI-SVD source is opt-in and **raises if its sidecar is unusable** rather than silently steering
  on nothing.
- **Two soft demotions, both live and distinct:** a raw-tag *compatibility* penalty in the
  candidate pool, and the taxonomy-graph *pair-floor* penalty at the beam edge. Neither is a hard
  gate — a hard genre gate detonates the relaxation cascade and blows the time budget.

The artifact bakes graph-resolved genres (`genre_source: graph`); GUI chips are ordered
most-specific → broadest (`granularity.py`).

---

## Runtime: the pier-bridge engine

Every playlist is built by the **pier-bridge** topology (the legacy greedy constructor is dead
code, unconditionally bypassed):

1. Seed tracks become fixed **piers** (for an artist, its catalog is medoid-clustered into piers).
2. Piers are ordered for bridgeability; each adjacent pair defines a **segment**.
3. Each segment is filled by a **constrained beam search** through the sonic space — per-step
   score = transition + bridge (harmonic-mean similarity to both piers) + soft genre/pace/energy
   penalties, keeping the top `beam_width` (global 40, doubling toward 200 on infeasibility).
4. Segments concatenate; a final **edge-repair** pass swaps a single interior track if any edge
   is catastrophically anti-aligned.
5. Export to M3U / Plex.

### The collapse-prevention stack

Long bridges tend to **sag** into the dense, genre-blurred "average" region rather than
representing the seeds' actual character. Five levers counter this; the shipped `config.example`
turns the first three (the *core*) on, with dataclass rollbacks off:

| Lever | Key (shipped value) | What it does |
|-------|--------------------|--------------|
| **Variable bridge length** | `variable_bridge_length: true` | A segment flexes its interior count (within a soft total band) to land more smoothly on the next pier — lifts the worst edge instead of padding to a rigid count. |
| **Anti-center (SP2)** | `seed_character_mode: anti_center`, `_strength: 2.0` | Demotes interior candidates that sit closer to the local pool centroid than to their own piers — the direct anti-sag score. |
| **Mini-piers (SP3)** | `mini_pier_enabled: true`, `_max_interior: 5` | Splits an over-long segment by pinning a high-character waypoint as an extra pier, so the beam structurally *cannot* sag past it. Fixes the residual sag that anti-center alone plateaus on (dreampop 103%→63%). |
| Roam corridors | live-only (`config.yaml`) | On-manifold kNN geodesic corridors + minimax worst-edge ordering. |
| Edge repair | live-only / `edge_repair_enabled` | Last-mile single-swap safety for a broken edge. |

> Anti-center is a *scoring* fix; mini-piers is a *structural* fix. They compose: scoring nudges
> candidates, structure guarantees the bridge can't drift past a character anchor. The full
> reasoning — including the abandoned "density-floor" lever that failed — is in
> `DESIGN_RATIONALE.md` §"Collapse prevention."

**Time budget.** `generation_budget_s` bounds a whole generation; the shipped default is `0`
(**disabled** — quality-first, every lever runs to completion). A positive value re-arms a soft
deadline that threads into every segment loop. There is a 90 s hard ceiling as a design target.

---

## The four mode axes

Cohesion-vs-discovery is exposed as four independent axes. **`cohesion_mode` drives the beam;
the other three gate the candidate pool.**

| Axis | Controls | Levels |
|------|----------|--------|
| `cohesion_mode` | beam tightness (per-mode bridge floors + edge weights) | strict / narrow / dynamic / discover |
| `genre_mode` | genre pool gating + admission floor | strict / narrow / dynamic / discover / off |
| `sonic_mode` | sonic pool gating | strict / narrow / dynamic / off |
| `pace_mode` | rhythm/tempo gating (BPM + onset bands) | strict / narrow / dynamic / off |

All default to `dynamic`. The per-mode pier-bridge knobs (`bridge_floor_<mode>`,
`weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`.

**Pace is embedding-independent.** It gates on BPM and onset-rate log-distance bands plus a soft
rhythm penalty, reading DB features — so it survives any sonic-embedding change. A beatless pier
disables its own BPM band (BPM is meaningless on drone), keeping the onset band. "Energy" is a
separate signal — trained *arousal* (Essentia), not loudness — currently wired for admission
rescue and an arc but off by default.

---

## Browser GUI wiring

The browser GUI is the only front-end (the PySide6 desktop app was removed).

```
Browser (React SPA, web/dist) ──/api + /ws──► FastAPI (src/playlist_web/app.py)
                                                  └─ NDJSON over stdio ─► worker (src/playlist_gui/worker.py) ─► pier-bridge
```

- FastAPI owns the worker: it spawns one long-lived worker subprocess at startup and speaks
  **NDJSON over stdio** (16 MiB line limit). A dead/stalled worker maps to 503/504, never a bare 500.
- **Two top-level tabs:** `generate` and `tools`. Genre Review and Taxonomy are **sub-tabs** of
  the right-hand `AdvancedPanel` (`diagnostics | blacklist | review | taxonomy`). The four axis
  sliders and the popular-seeds / bangers dropdowns live in `GenerateControls`.
- **Policy layer** (`src/playlist_gui/policy.py::derive_runtime_config`) maps UI slider modes →
  runtime config, and is the source of truth for a set of policy-owned keys. **Only the web path
  goes through it** — the CLI sets mode strings directly, so a test/harness that bypasses policy
  will see modes as inert (a known false-negative trap).
- `tools/serve_web.py` rebuilds `web/dist` on every launch (unless `--no-build`), default port 8770.

---

## Key data stores

| Path | What | Irreplaceable? |
|------|------|----------------|
| `data/metadata.db` | Track DB + `release_effective_genres` (genre authority) | **Yes** — days to re-analyze. 2× confirm + backup before any write. |
| `data/ai_genre_enrichment.db` | Enrichment sidecar: adjudications, escalation + review queues, taxonomy decisions | Regenerable but costly. |
| `data/artifacts/beat3tower_32k/data_matrices_step1.npz` | The generation artifact: all sonic variants + genre + energy matrices | Rebuildable from the sidecars. |
| `.../mert_shards/` + `mert_sidecar.npz` | MERT embeddings | **Yes** — ~55h CPU. Never delete/overwrite. |
| `.../muq_sidecar.npz` | MuQ embeddings | **Yes** in practice — extraction not on master. |
| `data/layered_genre_taxonomy.yaml` | The genre taxonomy graph | Living artifact, GUI-editable. |

---

## Configuration model

`config.yaml` is gitignored — copy it from `config.example.yaml`. Behavior resolves through the
three layers described at the top: **dataclass (rollback, off) → `config.example.yaml` (shipped,
validated-on) → `config.yaml` (live)**. Full key reference: [`CONFIG.md`](CONFIG.md). Knob-by-knob
tuning: [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md).

## Extension points

- **Add an export format** — implement an exporter alongside `src/m3u_exporter.py` and call it
  from the generation tail in `main_app.py` / the worker.
- **Add a sonic variant** — extract a sidecar, add a `fold_<variant>_into_artifact.py`, wire its
  auto-fold into the `artifacts` stage, and add its transition calibration to
  `TRANSITION_CALIB_BY_VARIANT`. The load-time override + missing-key-raises wiring is generic.
- **Add / change a mode behavior** — edit the presets in `src/playlist/mode_presets.py` (the
  single source for mode-driven gates/weights); the policy layer and CLI both read from there.
