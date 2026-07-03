# Technical Playlist Generation Flow

**Last updated:** 2026-07-03
**Purpose:** the authoritative, code-level, `file:line`-cited walkthrough of a single playlist
generation, end to end.

This is the layer-2 companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the orientation map —
read that first if you're new) and a sibling to [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md)
(the **why** — experiments, results, rejected alternatives). This doc states **what runs, in
what order, in which file** for one generation. Where a mechanism is itself a deep subsystem
with its own document — the beam search, the weak-edge cascade knobs, genre data authority — this
doc describes it at the depth needed to follow the call chain and points to the deep-dive rather
than duplicating it.

> **Framing.** Multi-pier seed generation (2+ seeds, beam-searched bridges between them) is
> production and is the frame of this walkthrough. **Artist mode is one entry point that produces
> piers**, not a separate engine: an artist-mode request clusters one artist's catalog into piers
> and then hands off to the exact same pier-bridge builder a multi-seed request uses. Single-seed
> requests are the degenerate case (one pier, used as both start and end — an "arc"). There is
> only one playlist-construction engine in this codebase; the legacy greedy `constructor.py` path
> is dead code, unconditionally bypassed (`src/playlist/pipeline/core.py:668`, `if True: #
> Always use pier-bridge`).

| Doc | Covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System map: offline pipeline, sonic space, genre, the four mode axes, GUI wiring. |
| **This doc** | Code-level walkthrough of one generation, phase by phase, `file:line`. |
| [`DJ_BRIDGE_ARCHITECTURE.md`](DJ_BRIDGE_ARCHITECTURE.md) | Deep-dive on the beam's internal scoring math. **Caveat:** it documents the optional, off-by-default `dj_bridging` vector-mode/IDF/coverage genre-waypoint system (`dj_bridging.enabled: false` in `config.example.yaml:457`) — a *different*, older genre-routing lever than the taxonomy arc-steering described in Phase 5 below, which is what actually runs by default today. Read it for beam mechanics (pooling, scoring shape); don't take its genre-routing description as the current default. |
| [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) | Why each current method was chosen over the alternative that was tried. |
| [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md) | Knob-by-knob tuning recipes for everything in Phases 5–6. |
| [`CONFIG.md`](CONFIG.md) | Full config key reference. |
| [`WIRING_STATUS.md`](WIRING_STATUS.md) / [`CLEANUP_LIST.md`](CLEANUP_LIST.md) | Shipped-vs-live divergences and open gaps. |

---

## Phase 1 — Entry, config, and the CLI/policy split

There are two entry points, and they **do not go through the same config-derivation code**:

### 1.1 CLI (`main_app.py`)

`main()` (`main_app.py:535`) parses argparse flags — `--artist` / `--genre` (mutually exclusive,
`main_app.py:543–553`), `--track`, `--anchor-seed-ids`, `--tracks` (default 30), `--cohesion-mode`,
`--genre-mode` / `--sonic-mode` / `--pace-mode`, `--mode` (quick preset), and the `--pb-*`
experiment flags (`main_app.py:613–657`). Two flags that used to exist do **not**: there is no
`--seeds` (seed-track mode is GUI/worker-only — `GenerateMode = Literal["artist", "genre",
"seeds", "history"]`, `src/playlist/request_models.py:9`) and no `--sonic-variant` (the CLI
variant selector was retired with the towers/MERT removal; the sonic space is chosen in
`config.yaml` only, see Phase 2). `--pace-mode` also has one fewer choice than the other three
axes — `strict|narrow|dynamic|off`, no `discover` (`main_app.py:602–606`).

`PlaylistApp.__init__` (`main_app.py:34`) loads `Config(config_path)` (`src/config_loader.py:12`),
which on construction: loads the YAML, calls `_apply_mode_presets()` (resolves `genre_mode` /
`sonic_mode` / `pace_mode` strings into weights/floors via `mode_presets.py`), `_validate_config()`,
and `_publish_artifact_settings()` (config_loader.py:15–17) — the last one is Phase 2's entry
point.

CLI mode overrides are applied **directly onto the config dict**, not through a policy layer:

```python
# main_app.py:708-716
playlists_cfg = app.generator.config.config.setdefault('playlists', {})
if genre_mode:
    playlists_cfg['genre_mode'] = genre_mode
if sonic_mode:
    playlists_cfg['sonic_mode'] = sonic_mode
if pace_mode:
    playlists_cfg['pace_mode'] = pace_mode
if genre_mode or sonic_mode:
    apply_mode_presets(playlists_cfg)
```

This means the CLI **skips** every translation the policy layer does for the web path: recency
enable/lookback → `playlists.recently_played_filter.*`, artist-spacing preset →
`ds_pipeline.constraints.min_gap`, and tag-steering tags → `pier_bridge.tag_steering_tags` are all
policy-only (§1.2). A CLI run either sets these directly in `config.yaml`, or doesn't get them.

`GeneratePlaylistRequest.from_cli_args` (`main_app.py:758`, `src/playlist/request_models.py:162`)
normalizes the parsed args into one request object; `mode` dispatches to
`run_single_artist` / `run_single_genre` (`main_app.py:770+`).

### 1.2 Worker / web entry — the policy layer

The browser GUI never talks to `main_app.py`. It POSTs to FastAPI (`src/playlist_web/app.py`),
which calls the policy layer **before** building overrides for the worker:

```python
# src/playlist_web/app.py:198
policy = derive_runtime_config(ui, seed_artist_keys=seed_artist_keys)
...
overrides = policy.overrides
```

`derive_runtime_config` (`src/playlist_gui/policy.py:230`) is a pure function: `UIStateModel` in,
`PolicyDecisions` (overrides + human-readable notes) out. It resolves the four axes
(`policy.py:257–266`), tag-steering chips (**capped at 3**, `policy.py:267–276`), the Oops-All-
Bangers sonic/pace baseline override (`policy.py:284–287`), recency enable/lookback/min-playcount
(`policy.py:314–318`), artist-spacing → `min_gap` via `SPACING_MAP` (`policy.py:70–75, 330–342`),
and the `dj_bridging` gating rules (`policy.py:363–472` — note: this is the *legacy* dj_bridging
lever from §DJ_BRIDGE_ARCHITECTURE, gated separately from taxonomy steering).

`POLICY_OWNED_KEYS` (`policy.py:26–51`) is the set of config paths policy **always wins** for,
even over anything the Advanced Panel set — `merge_overrides` (`policy.py:191–223`) enforces this
by deep-merging policy overrides last and then re-asserting each owned key.

> **Trap.** A test/harness that builds `ds_pipeline` overrides by hand and skips
> `derive_runtime_config` will see mode strings as **inert** — this is the multi-pier
> slider-calibration false-negative documented in the `playlist-testing` skill. Always route
> through policy (or the CLI's direct-set path) when exercising modes.

### 1.3 Tag-steering wiring (artist mode)

Tag-steering chips are the seed artist's own published genres, fetched via
`GET /api/genres/for_artist` (`app.py:524` → `authority.resolved_genres_for_artist`,
`src/genre/authority.py:171` — excludes `inferred_family` hub genres, top 12). Selected tags
(≤3) flow `UIStateModel.steering_tags` → `policy.py:267–276` →
`overrides["playlists"]["ds_pipeline"]["pier_bridge"]["tag_steering_tags"]`. This is a **web-only**
override; the CLI has no equivalent flag. See Phase 4.3 for what it does to the candidate pool and
Phase 3.1 for the pier-side lever.

---

## Phase 2 — Loading the artifact bundle (MuQ, the sole sonic space)

### 2.1 Publishing the variant override

`Config._publish_artifact_settings` (`config_loader.py:63–72`) reads
`artifacts.sonic_variant_override` and calls `set_sonic_variant_override` (`src/features/
artifacts.py:24–37`), which stashes it in a process-wide global and clears the bundle cache if it
changed. The GUI worker does the same at startup (`src/playlist_gui/worker.py:447–452`) so both
entry points resolve the same sonic space without threading the override explicitly through every
call site.

### 2.2 `load_artifact_bundle`

Called from `generate_playlist_ds` (`src/playlist/pipeline/core.py:327`). The public function
(`artifacts.py:96–120`) resolves the override (explicit param > process-wide global) and delegates
to an `lru_cache(maxsize=2)`-wrapped loader (`artifacts.py:126–226`) — a single generation calls
this 3–7 times with the same path, and the cache collapses that into one NPZ decode.

Today the system has **one sonic embedding: MuQ** (`OpenMuQ/MuQ-MuLan-large`, 512-d, contrastive
audio-text, post-processed with `center_l2`). The hand-built rhythm/timbre/harmony towers and the
MERT embedding that replaced them were both deleted (archived under `data/archive/mert_2026/`); see
`ARCHITECTURE.md` § Sonic feature space and `DESIGN_RATIONALE.md` for the towers → MERT → MuQ arc.
`sonic_variant_override` defaults to `"muq"` when unset (`analyze_library.py:144`, `... or
"muq"`), and the artifact itself declares `X_sonic_variant="muq"` — override, artifact declaration,
and `config.yaml` all agree today.

### 2.3 Missing-key raise (a configured knob that can't act is a startup error)

```python
# artifacts.py:189-197
if sonic_variant_override:
    variant_key = f"X_sonic_{sonic_variant_override}"
    if variant_key not in data:
        raise ValueError(
            f"artifacts.sonic_variant_override='{sonic_variant_override}' is configured "
            f"but artifact {artifact_path} has no '{variant_key}' key. A configured knob "
            "that cannot act is a startup error — remove the override or fold the "
            "variant into the artifact first."
        )
```

If no override is set, the artifact-declared variant is used the same way, with a **warning**
(not a raise) falling back to plain `X_sonic` only if the declared variant's key is genuinely
absent (`artifacts.py:207–224`) — a legacy-artifact accommodation, not the normal path. Segment
matrices (`X_sonic_start/mid/end`) resolve the same variant-aware way, with a fallback to legacy
segment keys logged at INFO (`artifacts.py:228–254`).

### 2.4 What's in the bundle

`ArtifactBundle` (`artifacts.py:45–80`) carries, beyond the sonic matrices: `X_genre_raw` /
`X_genre_smoothed` (sparse tag-space matrices), `X_genre_dense` + `genre_emb` (the PMI-SVD dense
sidecar used for admission gating and genre steering), and the layered-graph shadow matrices
(`X_genre_leaf_idf`, `X_genre_family`, `X_genre_bridge`, `X_facet` + their vocabularies) used by
diagnostics and taxonomy steering. `sonic_pre_scaled=True` signals "this matrix is already the
final per-track vector, don't re-transform it" — read by `setup_embedding`
(`src/playlist/pipeline/embedding_setup.py:70–82`) to skip a spurious re-scale.

---

## Phase 3 — Seed → pier resolution

Two independent mechanisms decide which tracks become piers, and a third orders them.

### 3.1 Medoid clustering (`artist_style`) — artist mode only

When `playlists.ds_pipeline.artist_style.enabled` is true, `create_playlist_for_artist`
(`src/playlist_generator.py:1340`) clusters the seed artist's own catalog into piers instead of
using a single seed track:

1. Build `ArtistStyleConfig` from `ds_cfg.get("artist_style", {})` (`playlist_generator.py:1710–
   1747`) — includes `medoid_energy_weight`, `medoid_popularity_weight` (Last.fm hit bias), and
   `medoid_tag_weight` (`playlist_generator.py:1744–1746`, read from
   `pier_bridge.tag_steering_pier_weight`, default 0.3 — **the pier-side tag-steering lever**,
   applied inside cluster scoring at `src/playlist/artist_style.py:680–700` only when a
   `steering_target` is supplied and `medoid_tag_weight > 0`).
2. `cluster_artist_tracks` (`artist_style.py:540`) clusters and picks one medoid per cluster
   (`medoid_top_k` per cluster, `playlist_generator.py:1721,1802–1814`).
3. If `popular_seeds_mode == "fire"`, the cluster medoids are overridden outright by
   `select_popular_piers` — pure top-N Last.fm hits, no clustering (`playlist_generator.py:1882–
   1898`).
4. `order_clusters` (`artist_style.py:743`) orders the medoids; the result is capped to
   `target_pier_count` (`playlist_generator.py:1902–1909`) and becomes `anchor_seed_ids` fed
   into `generate_playlist_ds`.

> **Shipped-vs-live gap.** `config.example.yaml:173` ships `artist_style.enabled: false` — a
> fresh clone runs the **legacy per-seed pier path** below, not medoid clustering, for artist
> mode. The live config in this repo has it `true`. Tracked in `CLEANUP_LIST.md`.

### 3.2 Legacy / non-artist-style pier resolution

When `artist_style.enabled` is false (the shipped default), or for genre mode / seed mode / a
single explicit seed, piers come directly from the request: the primary `seed_track_id` plus any
`--anchor-seed-ids` (CLI) or explicit seed list (GUI seed mode). `resolve_pier_seeds`
(`src/playlist/pipeline/pier_resolver.py:24–95`, called at `pipeline/core.py:672`) maps those
track_ids to bundle indices, dedupes by `(artist, title)` track key, and guarantees the primary
seed is present (inserting it at position 0 if a caller's anchor list omitted it).

### 3.3 Pier ordering

Inside `build_pier_bridge_playlist` (`src/playlist/pier_bridge_builder.py:438`),
`_order_seeds_by_bridgeability` (`src/playlist/pier_bridge/seeds.py:77–140`) reorders the resolved
piers to maximize bridgeability: **exhaustive permutation search for ≤6 seeds**, greedy
nearest-neighbor for more. The per-pair score blends a bridgeability heuristic, raw sonic cosine,
and genre cosine (`dj_seed_ordering_weight_{bridge,sonic,genre}`); when `roam_corridors_enabled`
is set the objective switches from *sum* of pair scores to *min* (the weakest link is what
matters, not the average) — `seeds.py:94–140`.

A single seed becomes both start and end pier — an **arc** — by duplicating it
(`pier_bridge_builder.py:849–855`, `is_single_seed_arc`), producing one segment instead of zero.

Structural mini-pier insertion (SP3) happens *after* this ordering step, splitting long segments
by pinning extra waypoint piers — that's part of anti-sag scoring, covered in Phase 6.1.

---

## Phase 4 — Candidate pool construction (`build_candidate_pool`)

`build_candidate_pool` (`src/playlist/candidate_pool.py:518–565`) is called once per generation
(twice if the Oops-All-Bangers relax-to-fill cascade needs a rebuild) from a closure in
`pipeline/core.py:576–619`. It admits tracks against **three independent axes** — sonic, genre,
pace — plus artist diversity and the tag-steering pool lever, and returns a ranked, capped pool.

### 4.1 The ranking embedding vs. the gating spaces

Two different vector spaces are in play, and it's easy to conflate them:

- **`embedding`** (the function's primary argument) is a legacy 32+32-dim PCA-reduced hybrid of
  sonic and genre (`setup_embedding` → `build_hybrid_embedding`,
  `src/similarity/hybrid.py:27`), built once per run in `pipeline/embedding_setup.py:47`. Its
  cosine to the seed(s) (`seed_sim_all`, `candidate_pool.py:577–586`) is the base ranking score
  used for per-artist ordering and pool truncation.
- **Admission floors** are computed separately against the *actual* current sonic/genre spaces:
  raw `X_sonic` (MuQ) cosine for the sonic hard floor (`candidate_pool.py:642–676`), and the dense
  PMI-SVD `X_genre_dense` sidecar (preferred) or sparse raw/smoothed cosine for the genre floor
  (`candidate_pool.py:775–870`).

### 4.2 Sonic admission

`sonic_seed_sim` = max-over-seeds cosine on raw `X_sonic` (`candidate_pool.py:647–652`). The gate
is either a fixed floor (`cfg.min_sonic_similarity`) or, when `sonic_admission_percentile` is set,
a **per-seed adaptive percentile floor** — distribution-relative, so it survives embedding
rebuilds instead of hardcoding a cosine value calibrated for a since-replaced space
(`candidate_pool.py:654–676`). A duration penalty/cutoff (geometric curve vs. median seed
duration, hard-excludes beyond `duration_cutoff_multiplier`) is applied to `seed_sim_all` before
any floor (`candidate_pool.py:588–640`).

### 4.3 Genre admission (+ the tag-steering pool lever)

When `X_genre_dense` is available, the dense path is preferred (`candidate_pool.py:775–870`):
either a **centroid** aggregate (mean of all seed dense vectors, the default) or a **per_seed**
union-of-neighborhoods aggregate (each seed contributes its own floor; a track passes if it clears
*any* seed's floor). Tag-steering forces `centroid` even if `per_seed` was requested
(`candidate_pool.py:784–786`, `core.py:540–542`) because the steering blend only makes sense
against one target vector.

The **tag-steering pool lever** blends the resolved tag target into the admission centroid:

```python
# candidate_pool.py:824-835
if steering_target is not None:
    _blend = float(np.clip(steering_blend, 0.0, 1.0))
    _steered = (1.0 - _blend) * seed_dense + _blend * np.asarray(steering_target, ...)
    ...
    seed_dense = _steered / _steered_norm
```

`steering_blend` defaults to 0.5 (`tag_steering_pool_blend`, `config.example.yaml:294`);
`steering_target` is resolved once per run in `pipeline/core.py:544–559` via
`src.playlist.tag_steering.resolve_tag_steering_target` — no tags selected means `steering_target
is None` and this whole block is a no-op (byte-identical to legacy). The dense genre admission
floor is itself either fixed (`min_genre_similarity`) or an adaptive percentile
(`genre_admission_percentile`) computed against the (possibly steered) centroid's similarity
distribution (`candidate_pool.py:842–863`).

### 4.4 Pace admission

Two independent hard gates, both log-distance based (`src/playlist/bpm_axis.py:35–40`):

- **BPM** (`candidate_pool.py:680–710`): rejects candidates whose log-BPM distance to the nearest
  seed exceeds `bpm_admission_max_log_distance`, *unless* BPM is missing (NaN) or
  `tempo_stability` is below `bpm_stability_min` — a beatless/ambient track's BPM reading is
  meaningless, so low-stability tracks bypass the gate rather than being wrongly rejected or
  wrongly admitted on a garbage BPM.
- **Onset rate** (`candidate_pool.py:716–730`): same log-distance mechanism, **NaN-only** bypass
  (no stability escape hatch — onset rate is the more trustworthy beat-presence signal on
  beatless material; see `DESIGN_RATIONALE.md`).

Tracks rejected *only* by these bands (not by sonic/genre) can be re-admitted by
`select_energy_rescue` (`src/playlist/energy_rescue.py`, called at `candidate_pool.py:1075`),
evenly spaced across the sorted arousal distribution, up to `pace_rescue_k_energy` (0 in `dynamic`
mode — off by default; active in `strict`/`narrow`).

### 4.5 Diversity + recency (pre-order, never post-order)

A per-artist cap (`candidates_per_artist` (+ `seed_artist_bonus` for the seed's own artist),
`candidate_pool.py:1167–1250`) is applied at pool-build time — this is a *pool composition* cap,
separate from the beam's own per-segment one-artist-per-segment + cross-segment `min_gap`
enforcement (Phase 5.5).

**Recency exclusion never happens inside `build_candidate_pool`.** It happens earlier, in
`restrict_bundle` (`src/playlist/pipeline/bundle_restrict.py:38–162`), which masks
`excluded_track_ids` (recently-played Last.fm keys, resolved in `playlist_generator.py:978–990`)
out of the bundle *before* the pool is ever built — a hard architectural rule (see CLAUDE.md's
"don't re-introduce post-order recency filtering" gotcha): seed tracks at pier positions may be
recently played but are explicitly requested, so recency must never re-filter after ordering.

### 4.6 Oops-All-Bangers popularity gate (brief)

Separate from `popular_seeds_mode` (Phase 3's *pier* selection lever), `popularity_mode` gates
*pool admission*: `on`/`oops` hard-exclude below a Last.fm popularity-rank cutoff plus a beam soft
penalty. If the gated pool is too small, a relax-to-fill cascade loosens sonic → pace → genre →
popularity (in that order, popularity last — the only purity-breaking rung) and rebuilds
(`pipeline/core.py:632–652`). Off by default; see `ARCHITECTURE.md` and `CONFIG.md` for the full
knob set.

### 4.7 Output

`pool.eligible_indices` is deduped by `(artist, normalized title)` track key, keeping the
best-scored version per group (`dedupe_pool_by_track_key`, `pier_resolver.py:98–138`, called at
`pipeline/core.py:679–680`) before being handed to the pier-bridge builder as
`candidate_pool_indices`.

---

## Phase 5 — Per-segment beam search + transition scoring

For each adjacent pier pair (a "segment"), `build_pier_bridge_playlist`
(`pier_bridge_builder.py:438`) computes an interior length (even split with remainder to earlier
segments, `pier_bridge_builder.py:886–892` — see Phase 6.2 for how variable-length flexing
overrides this) and calls `_beam_search_segment` (`src/playlist/pier_bridge/beam.py:230`,
imported/re-exported at `pier_bridge_builder.py:135–137`, invoked at
`pier_bridge_builder.py:1479`) to fill it.

**This doc describes the beam at the level needed to follow scoring; for the full beam-search
mechanics — the per-step scoring formula, transition calibration, anti-sag levers, and the
weak-edge recovery cascade — see [`DJ_BRIDGE_ARCHITECTURE.md`](DJ_BRIDGE_ARCHITECTURE.md), which
documents the current taxonomy-steering arc as well as the separate, off-by-default legacy
`dj_bridging` lever (the S1/S2/S3 IDF/coverage system).**

### 5.1 Per-candidate combined score

At each step, for each beam-surviving path, each candidate's score (`beam.py:1256–1309`) is:

```
combined_score = weight_bridge * bridge_score + weight_transition * trans_score
                - anti_center_penalty(...)          # SP2, Phase 6.1
                - pace_penalty                       # BPM/onset out-of-band soft demotion
                - progress_penalty                   # monotonic pull toward pier B
                + weight_genre * arc_sim              # taxonomy genre-arc vote, §5.4
                (+ genre_tiebreak / genre_penalty when steering is off)
```

`bridge_score` is the harmonic mean of cosine-to-pier-A and cosine-to-pier-B
(`beam.py:1256–1259`) — a candidate must be plausible toward *both* endpoints, not just closer to
one. `dest_pull` (`eta_destination_pull`) adds a small linear bias toward pier B
(`beam.py:1262`).

### 5.2 Transition scoring: the calibrated MuQ cosine

`trans_score` comes from `src/playlist/transition_metrics.py` — the **single source of truth**
shared by the beam, the builder's own edge-stat bookkeeping, the reporter (Phase 7), and the
opt-in edge-repair pass, so none of them can silently diverge on the same edge. The raw signal is
end-of-track → start-of-track cosine on the segment-level sonic matrices; it's rescaled through a
calibrated logistic (`_calibrate_transition_cos`, `src/playlist/pier_bridge/vec.py:35–61`):

```python
z = gain * (value - center) / scale
sigmoid(z)   # numerically stable both branches
```

This replaced a linear `(x + 1) / 2` rescale that wasted its output range on negative cosines real
edges never produce, compressing the realistic band into a narrow slice near 0.6–0.75 (gap
between good and bad edges: 72% → 8%). Calibration is **keyed to the active sonic variant**
(`TRANSITION_CALIB_BY_VARIANT`, `transition_metrics.py:21–26`) — today only `"muq": (0.594,
0.092)` is registered (the MERT band was deleted along with MERT itself). `resolve_transition_calib`
(`transition_metrics.py:29–56`) **raises** for an unregistered variant rather than silently
reusing the wrong band — a mismatched calibration saturates every edge toward 1.0, which is a
much harder failure to notice than a startup error.

### 5.3 Genre during the beam: taxonomy arc steering (the default) vs. `dj_bridging` (opt-in, off)

Two *separate* genre mechanisms can influence beam scoring; only the first is on by default:

1. **Taxonomy arc steering** (`genre_steering_enabled: true`, `genre_steering_source: taxonomy` —
   `config.example.yaml:323,334`). `_require_usable_genre_steering`
   (`pier_bridge_builder.py:415–435`, called at `:645`) raises loudly if a configured source
   can't act (e.g. `source=dense` with no dense sidecar) rather than silently producing zero
   targets on every segment. Under `taxonomy` (which can always act, reading in-artifact
   `X_genre_raw`), per-segment target vectors are built via
   `TaxonomySteering.build_taxonomy_genre_targets`
   (`src/playlist/pier_bridge/taxonomy_steering.py:263`) — it walks the shortest taxonomy path
   between the two piers' genres and produces smoothed per-step targets. In the beam, a
   candidate's cosine to that step's target (`arc_sim`) is added to `combined_score` weighted by
   `weight_genre`, gated by a per-segment on-arc percentile floor and an absolute
   `genre_arc_floor` safeguard (`beam.py:1292–1306`).
2. A **pairwise genre-edge soft floor** runs alongside arc steering when `genre_pair_floor > 0`:
   built from a **tag-level max-similarity provider**
   (`taxonomy_steering.build_taxonomy_pair_provider`, wired at `pier_bridge_builder.py:661–685`
   because the *smoothed-vector* cosine was shown unable to separate bad edges from good — see
   `DESIGN_RATIONALE.md` "genre metric = max"). Below-floor adjacent edges are **demoted, not
   rejected** (`genre_pair_penalty`, subtracted in `beam.py:842–859`) — a hard gate here was shown
   to detonate the relaxation cascade on broad-genre segments.

The legacy `dj_bridging` system (vector-mode/onehot targets, IDF weighting, coverage bonus —
fully documented in `DJ_BRIDGE_ARCHITECTURE.md`) is a **separate, mutually-independent** code
path gated by `dj_bridging_enabled` (dataclass default `False`, `pier_bridge/config.py:226`;
shipped `config.example.yaml:457` also `false`). It is not wired to run alongside taxonomy
steering in normal operation — leave it off unless you're specifically experimenting with the
legacy ladder-routing behavior that doc describes.

### 5.4 Diversity enforcement during the beam

Per-segment one-track-per-artist and cross-segment `min_gap` are enforced **during** expansion,
not as a post-hoc filter: candidate artist identity is resolved via `resolve_artist_identity_keys`
(handles collaborations/ensemble suffixes) at each candidate-check site
(`beam.py:975–990, 1220, 1446, 1589`), and `disallow_seed_artist_in_interiors` blocks the seed
artist's own tracks from non-pier positions (`beam.py:981–990`) when configured. Boundary context
carried from the previous segment (`beam.py:951`) makes `min_gap` a true cross-segment constraint,
not a per-segment-only one.

---

## Phase 6 — Anti-sag scoring + the weak-edge recovery cascade

Long bridges tend to **sag**: interiors drift toward the dense, genre-blurred local average
instead of representing the piers' actual character. Two levers counter this at selection time;
a four-pass cascade cleans up whatever slips through.

### 6.1 Anti-center (SP2)

`seed_character_mode: anti_center` (dataclass default `"off"`,
`pier_bridge/config.py:106`; shipped `config.example.yaml:288–289` turns it on at strength 2.0).
Precomputed once per segment (`beam.py:823–836`): the local pool centroid is the mean of the
segment's L2-normalized candidates (piers excluded). Applied per candidate
(`beam.py:1268–1272`):

```python
# src/playlist/pier_bridge/seed_character.py:18-22
def anti_center_penalty(cand_center_sim, bridge_score, strength):
    return strength * max(0.0, cand_center_sim - bridge_score)
```

i.e. subtract a penalty proportional to *how much closer* a candidate sits to the pool's generic
center than to its own piers; zero penalty for a candidate that's more pier-like than central.
The "hubness" alternative (kNN in-degree deflation) was tested and retired (weak, didn't scale) —
`anti_center` is the only surviving mode besides `off`.

### 6.2 Mini-piers (SP3) — structural anti-sag

`mini_pier_enabled` (dataclass default `False`, `pier_bridge/config.py:110`; shipped
`config.example.yaml:272` = `true`). Where anti-center *nudges* scoring, mini-piers *structurally*
prevents sag: after pier ordering (Phase 3.3) and before segment-length allocation,
`plan_pier_sequence` (`src/playlist/pier_bridge/mini_pier_select.py:60`, called at
`pier_bridge_builder.py:857–882`) greedily splits the longest segment by pinning a
`select_waypoint` pick (`mini_pier_select.py:17–52`) as an extra pier — chosen as the most
"between" candidate (within `margin` of the best min-similarity-to-both-piers) that is *also* the
least central relative to that between-region (an anti-center pick among the smooth set), so the
inserted waypoint is on-character rather than the wallpaper average. Waypoint artists are excluded
from being the seed's or an existing pier's artist (`pier_bridge_builder.py:861–870`); a hard cap
(`total_tracks // 4`) bounds how many waypoints can be inserted.

Variable bridge length (see the cascade below) then still applies per-segment on top of this
structural split.

### 6.3 The weak-edge recovery cascade

After the beam assembles all segments, a **fixed four-pass cascade** lifts weak/broken transition
edges, escalating from least- to most-destructive. **It runs once, top to bottom — not a retry
loop**; each pass hands its (possibly mutated) playlist to the next, so a late deletion is never
re-optimized by an earlier pass.

| # | Pass | File:line | Scope | Trigger |
|---|------|-----------|-------|---------|
| 1 | **Variable bridge length** (add-only) | gate check `pier_bridge_builder.py:1724`; mechanics in `src/playlist/pier_bridge/var_bridge.py:25–49` | per-segment, pre-assembly | worst edge `< variable_bridge_min_edge` (0.30, `pier_bridge_builder.py:1731`) |
| 2 | **Tail-DP** | `pier_bridge_builder.py:2434–2514`, engine `src/playlist/pier_bridge/tail_dp.py` | per-segment, pre-assembly (re-opens last ≤2 interior slots) | window min-edge below `tail_dp.floor` |
| 3 | **Edge repair** (break-glass) | `pier_bridge_builder.py:2877–2908`, engine `src/playlist/repair/edge_repair.py:233` (`repair_playlist_edges`) | global, post-assembly (swaps ONE interior track, never changes length) | `T < t_floor` (0.30) **or** catastrophic `T_centered_cos < centered_cos_floor` (−0.5) |
| 4 | **Edge delete** (remove-only, last resort) | `pier_bridge_builder.py:2919–2939`, engine `src/playlist/repair/edge_delete.py:46` | global, post-repair (deletes ONE interior endpoint) | worst `T` below `edge_delete.floor`, up to `max_deletions` |

Variable bridge length works by building the nominal even-split segment, and if its bottleneck
edge (including the pier-return edge) is already `>= good_enough`, keeping it; otherwise it
greedily builds every interior length within `[nominal - flex, nominal + flex]` (deterministic
cap, `variable_bridge_max_flex_segments`), takes the best bottleneck, and prefers the length
closest to nominal within `epsilon` — a "prefer-N, don't pad for the sake of N" tie-break.
Edge-delete respects a bystander artist's `min_gap` (`_violates_min_gap_after_delete`,
`edge_delete.py:27`) so a deletion can't silently create an artist-spacing violation elsewhere.

Full knob table, per-pass config keys, the "fixer deadzone" (0.30–~0.75, where an
ugly-but-legal edge gets no fixer attention), and the edge-repair-vs-reporter `T`-mismatch caveat
are in [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md) Knob 4. Dataclass rollback
defaults are **off** for variable-bridge and edge-repair, **on** for tail-DP and edge-delete;
`config.example.yaml` turns all but edge-repair on (edge-repair is live-only — a shipped-template
gap tracked in `CLEANUP_LIST.md`).

**Time budget.** `generation_budget_s` (dataclass default 60.0, `pier_bridge/config.py:347`; live
`0` = disabled) bounds the whole generation, computed once as a shared deadline
(`pipeline/core.py:852–859`) so every segment loop and relaxation retry inside
`build_pier_bridge_playlist` shares one budget instead of resetting its own clock. `<= 0` disables
both the soft deadline and the per-build relaxation cap entirely — "quality-first while tuning."
90s is a design-target ceiling, not a separately enforced hard cutoff; see
[`feedback_generation_time_budget`](../CLAUDE.md) discipline and Knob 10 in
`PLAYLIST_ORDERING_TUNING.md`.

---

## Phase 7 — Reporter / quality metrics

`src/playlist/reporter.py` computes and logs post-hoc quality metrics from the **final assembled
order** — independent of whatever the beam scored internally, so a beam-vs-reporter mismatch is
itself a diagnostic signal (see the `T-mismatch` warning below).

### 7.1 Edge scores

`compute_edge_scores_from_artifact` (`reporter.py:237–447`) re-loads the artifact, resolves the
transition calibration for the active variant exactly as the beam did
(`resolve_transition_calib`, `reporter.py:305`, mirroring `pier_bridge_builder.py:490`), builds a
`TransitionMetricContext` (`reporter.py:306–318`), and scores every adjacent pair with
`score_transition_edge` — the same function the beam and edge-repair use
(`transition_metrics.score_transition_edge`). Each edge dict carries `T`, `T_raw`,
`T_centered_cos`, `S` (sonic), `G` (genre), and `H` (hybrid).

### 7.2 Percentile summary + weakest edges

`print_playlist_report` (`reporter.py:450–907`) logs, per generation:

- **T / S / G distributions**: mean, p10, p50, p90, p99, min (`_summ`, `reporter.py:706–721`) —
  the project convention is to report the **distribution and the floor**, never just the mean
  (Design Principle 21 — a strong mean can hide one broken edge).
- **`below_floor` count** against `transition_floor`.
- **Weakest transitions (bottom 3 by T)** with both endpoints' artist/title
  (`reporter.py:866–886`) — the first thing to read when a playlist "feels" wrong.
- **`diagnose_t_mismatch`** (`reporter.py:101–126`): warns when the beam's own recorded
  `trans_score_in_beam` disagrees with the final reporter `T` on a below-floor edge — a real
  regression signal (the beam and reporter should never diverge on the same edge; see CLAUDE.md's
  "don't change `transition_weights` without `tower_weights`" gotcha for the historical version of
  this failure mode, now moot post-tower-removal but the shared-metric discipline that replaced it
  still applies).
- **BPM summary**, distinct-artist count, and (with `verbose_edges`) baseline library percentiles
  for comparison.

`emit_selected_edge_audit` (`reporter.py:129–197`) and `emit_edge_repair_log`
(`reporter.py:200–234`) are opt-in per-edge diagnostic dumps (full scoring breakdown per edge;
repair swap accept/reject log) — see the `playlist-testing` skill's "diagnosing a generation
outcome" section for when to reach for these instead of trusting summary metrics alone.

### 7.3 What ships in the report

`build_pier_bridge_playlist`'s own `stats` dict (`pier_bridge_builder.py:3048–3140+`) carries
`min_transition`/`mean_transition`, per-segment bridge-floor/backoff usage, edge-repair and
edge-delete logs, soft-genre-penalty hit counts, and the full effective config snapshot — this is
what `docs/run_audits/` audit reports and the GUI diagnostics panel read from.

---

## Phase 8 — Export

### 8.1 M3U

`M3UExporter.export_playlist` (`src/m3u_exporter.py:34–100`) writes an EXTM3U file with one
`#EXTINF` + file-path line per track, plus a non-standard `#EXTVARIANT:<sonic_variant>` line
per track for provenance. The filename gets a `_sonic-<variant>` suffix when the variant isn't
`"raw"` (`m3u_exporter.py:49`). Called from `main_app.py:436–445` for CLI runs (worker/web calls
the equivalent export path after generation).

### 8.2 Plex (optional)

`PlexExporter.export_playlist` (`src/plex_exporter.py:279`) is called only if a Plex exporter was
configured (`main_app.py:447–454`) and only when `dry_run` is false. Failures are caught and
logged, never fatal to the generation itself — Plex export is a local-first *enrichment* of the
result, never a gate (Design Principle 14).

---

## See also

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system map, the four mode axes, GUI wiring, offline
  pipeline.
- [`DJ_BRIDGE_ARCHITECTURE.md`](DJ_BRIDGE_ARCHITECTURE.md) — beam pooling/scoring deep-dive
  (with the `dj_bridging`-vs-taxonomy caveat above).
- [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) — why each current method beat its predecessor.
- [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md) — knob-by-knob tuning recipes for
  Phases 5–6.
- [`CONFIG.md`](CONFIG.md) — full config key reference.
- [`GOLDEN_COMMANDS.md`](GOLDEN_COMMANDS.md) — canonical CLI invocations, including the analyze
  pipeline stage list.

**End of document.**
