# Artist-mode energy-aware spread + popular ("banger") seeds — design

**Date:** 2026-06-23
**Status:** Design approved, pending spec review → implementation plan
**Branch:** `worktree-artist-energy-popular-seeds`

## Problem

Artist-mode picks anchor "piers" by clustering the artist's tracks in sonic space
(MERT) and taking one medoid per cluster. The medoid score today is purely:

```
scores = sonic_centroid_sim * 0.7 + duration_typicality * 0.3   # artist_style.py:234
```

Two gaps:

1. **No energy awareness in the spread.** Clusters are formed in sonic (timbre)
   space only, so the chosen piers may all sit in one arousal band — a flat arc,
   which Layer-1 principle #2 ("a playlist has an arc") calls a failure mode.
2. **No notion of a recognizable representative.** Within a cluster, the medoid
   is whichever track is most sonically central + duration-typical. Duration-
   typicality screens out interludes/outros, but a normal-length **B-side, deep
   cut, or live version sails right through**. Asked for "a Nirvana playlist,"
   you want *Smells Like Teen Spirit / Come As You Are / Heart-Shaped Box*, not
   *Beeswax* or a live *Molly's Lips*. The missing signal is **popularity**.

## Goals

- **Energy-aware spread:** the set of piers should tile the artist's arousal
  range, not just cover timbre.
- **Popular ("banger") seeds:** within each pier, prefer the recognizable hit.
  "Banger" = **high popularity** (Last.fm `artist.gettoptracks`), *not*
  danceability/arousal. Energy and popularity are two different signals doing two
  different jobs.
- **Diversity preserved:** popularity must not collapse the cross-cluster
  diversity the clustering provides.
- **Local-first (hard constraint):** external APIs (Last.fm) must never gate
  runtime generation. Top-tracks are fetched offline (analyze/enrich) and cached;
  generation reads only local artifacts.
- **Opt-in, backward-compatible:** new behavior ships behind config weights /
  a checkbox; defaults reproduce today's output exactly.

## Conflict resolution (decided)

**Energy-spread wins over popularity.** The style-clusters + energy slots define
the skeleton; popularity picks the most recognizable track *within* each slot.
Where a slot has no popular option, accept a less-popular track rather than
flatten the arc. Mechanically this is just `w_energy > w_pop` in the score
vector below. Exact weights are deferred to calibration ("sensible default now,
calibrate later").

## Two knobs, one score

Energy-spread and banger-bias are **two separate knobs combined in one scoring
vector**, operating at different levels:

- **Spread** is a *set-level* property: across clusters, the piers tile the
  arousal range. Cannot collapse diversity (it operates between clusters).
- **Banger-bias** is a *within-slot tiebreak*: given a fixed style+energy slot,
  prefer the more popular track. Cannot collapse cross-cluster diversity **by
  construction** — it only chooses within a slot.

---

## Architecture

### Verified code anchors (2026-06-23)

- Clustering entry point: `cluster_artist_tracks` / `_medoids_for_cluster` in
  `src/playlist/artist_style.py`. Medoid score combined at `artist_style.py:234`;
  top-k random pick + 2nd-medoid diversity path both ride that `scores` vector.
- Callers: `src/playlist_generator.py:1712` (primary, `create_playlist_for_artist`)
  and `:2626` (legacy DS path). Returns `(clusters, medoids, medoids_by_cluster,
  X_norm)` → `order_clusters(medoids, X_norm)` → cap to `target_pier_count` →
  `pier_ids` → DS pipeline via `anchor_seed_ids`.
- **Sonic space:** L2-normalized MERT (768-dim, `bundle.X_sonic`).
  `tower_weighted` is **inert** on MERT — the 137-dim shape check in
  `sonic_variant.py:284-322` fails, so it falls back to plain L2 cosine.
- **Energy sidecar:** `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz`,
  fields `arousal_p10/p50/p90`, `valence`, `danceability`, keyed by `track_id`.
  Loaded via `src/playlist/energy_loader.py::load_energy_matrix` — library-wide
  z-score, NaN for gaps, **no caching**. Today used *only* for candidate-pool
  energy-rescue (`pace_rescue_k_energy`) + beam-step pace penalties
  (`energy_step_*`/`energy_arc_*`). Not in clustering.
- **Energy load site (new):** `cluster_artist_tracks` runs in
  `create_playlist_for_artist` with its *own* bundle load (`playlist_generator.py:1675`),
  **before** the DS pipeline; the existing `energy_matrix` is loaded later in
  `pipeline/core.py:339`. So energy must be loaded *separately* in the artist
  path — a ~3-line reuse of `load_energy_matrix`, not threaded down from the beam.
- **Re-roll already exists internally:** `cluster_seed = base_seed + seed_epoch_val`
  (`playlist_generator.py:1681`). `seed_epoch` just isn't exposed to the web request.
- **Last.fm:** `src/lastfm_client.py::LastFMClient` has `get_top_tracks` but that's
  the *user's* top tracks (`user.gettoptracks`). **`artist.gettoptracks` does not
  exist yet** — a new method. Existing `lastfm` analyze stage (`ANALYZE_LIBRARY_STAGE_ORDER`)
  has rate-limiting (5/s) + exponential backoff, writes to the gitignored
  `data/ai_genre_enrichment.db` via `SidecarStore`.

### Component 1 — Energy-aware spread (Approach A)

Localized to `artist_style.py` plus one load site in the orchestrator.

1. **Load arousal in the artist path.** Reuse
   `load_energy_matrix(bundle.track_ids, sidecar_path=…, features=(energy_feature,))`
   near the `cluster_artist_tracks` call. Default `energy_feature = "arousal_p50"`
   (central, library z-scored arousal — same signal the pace gate trusts).
2. **Assign cluster energy slots (set-level spread).** After sonic k-means:
   - Compute each cluster's energy = median z(arousal) of its members.
   - Sort the *k* clusters by that median; assign each an **evenly-spaced energy
     target** across the artist's own arousal span (lowest cluster → low target,
     … highest → high target). **Span = a robust range (p10–p90 of the artist's
     z-arousal), not min–max**, so a single outlier track can't blow the slots
     apart; the exact percentiles are calibratable.
   - **Edge case:** if the artist's catalog is energy-flat (span below a small
     epsilon), the targets collapse and the energy term goes **inert** — we don't
     manufacture an arc that isn't there (guard the division).
3. **Extend the medoid score** (`artist_style.py:234`):

   ```
   scores = w_sonic    * sonic_centroid_sim     # today 0.7
          + w_duration * duration_typicality    # today 0.3
          + w_energy   * energy_slot_proximity  # NEW, default 0.0
          + w_pop      * popularity             # NEW, default 0.0
   ```

   where `energy_slot_proximity = 1 − clamp(|z_arousal − slot_target| / span, 0, 1)`.
   All four terms live on a comparable ~[0,1] scale so the weights mean what they
   say. With `w_energy > w_pop`, the medoid is pulled to its cluster's energy slot
   first, then the most popular track *near that slot* wins.
   - **All-zero defaults (`w_energy = w_pop = 0`) reproduce today's output byte-for-byte.**
   - The existing top-k random pick (`medoid_top_k`) and 2nd-medoid diversity path
     ride the same `scores` vector → "New Seeds" reroll works unchanged.

**Energy-spread has no GUI control** — it's a structural arc-quality improvement,
not a user choice. It lives as a config weight (off until calibrated, then a
sensible default). An energy toggle can be added later if wanted.

### Component 2 — Popularity data path (local-first)

Mirrors the energy-sidecar precedent end-to-end; generation never touches the network.

1. **New client method** `LastFMClient.get_artist_top_tracks(artist)` → calls
   `artist.gettoptracks` (ranked names + playcount/listeners). Reuses the existing
   rate-limiter + exponential backoff.
2. **New offline analyze stage** (e.g. `lastfm-toptracks`, registered in
   `ANALYZE_LIBRARY_STAGE_ORDER` near the existing `lastfm` stage). For each
   library artist with ≥ `toptracks_min_artist_tracks` local tracks, fetch and
   cache the raw ranked list in `data/ai_genre_enrichment.db` (gitignored, safe to
   write — **not** `metadata.db`). Resumable/incremental (skip artists already
   fetched, mirroring how the `lastfm` stage skips via
   `release_keys_with_source_type`).
   - New table (illustrative): `lastfm_artist_top_tracks(artist, rank, track_name,
     playcount, listeners, fetched_at)`.
3. **Build step → `popularity_sidecar.npz`** (`track_ids → popularity` in [0,1]).
   The hard part is **one canonical Last.fm name → many local versions** (studio,
   remaster, live, demo can all be present for the same song). We must resolve the
   canonical name back to the *right* local `track_id`, not just "a match." Two
   stages, reusing existing infra:
   - **Gather version-candidates.** Loose-normalize **both** the Last.fm name and
     the local titles via `title_dedupe.normalize_title_for_dedupe(mode="loose")`
     (strips `remaster / live / edition / year` suffixes), and group, *within the
     artist*, every local track that collapses to the same canonical title.
     Opportunistically match by **mbid first** — Last.fm top-tracks carry an mbid
     and `track_matcher.TrackMatcher` already indexes `_tracks_by_mbid` /
     `_tracks_by_norm`; fall back to loose-title + fuzzy `SequenceMatcher`.
   - **Resolve to the canonical bearer.** Among the grouped candidates, attach the
     hit's popularity to the single track with the highest
     `title_dedupe.calculate_version_preference_score` — the project's existing,
     deliberate stance: `live −30 / demo −25 / remix −20 / acoustic −15` are
     demoted, but **`remaster` is only −5 — never penalized** ("often better
     quality," per the code comment). Other versions in the group get 0.
     Consequences (the cases discussed):
       - A remaster that is the *only* local copy is the top candidate → it
         carries the full popularity. **No remaster penalty.**
       - A live cut next to the studio version loses to it (studio 100 > live 70).
       - The "penalty" is a *version-preference rank among real matches*, not a
         title-match failure — remasters rank with the studio cut, not the live takes.
   - Score basis = per-artist **rank** (not raw playcount), because we only ever
     rank within one artist's clusters and rank is robust across artists of wildly
     different global popularity (e.g. `score = 1 − (rank−1)/N` or a rank decay).
   - Sidecar lives alongside the artifact (e.g.
     `data/artifacts/beat3tower_32k/popularity/popularity_sidecar.npz`).
4. **Runtime `src/playlist/popularity_loader.py`** mirrors `energy_loader.py`:
   `load_popularity_vector(track_ids, sidecar_path)` → per-track [0,1], **NaN → 0
   (neutral, never a penalty)**. Threaded into the artist path alongside energy.
   If the sidecar is absent, `w_pop` has nothing to act on and the feature is
   dormant — backward-compatible.

**Configured-knob-must-act rule:** if `medoid_popularity_weight > 0` (or the
"Popular Seeds" checkbox is on) but `popularity_sidecar.npz` is missing, warn
loudly (don't silently no-op), per the project's standing rule.

### Component 3 — UX controls + wiring

Two controls, both artist-mode.

**"Popular Seeds" checkbox** — a boolean that activates the popularity term. Off
⇒ `w_pop = 0` (today); on ⇒ `w_pop` takes its configured value. Rides the existing
`include_collaborations` plumbing exactly:
`GenerateRequestBody.popular_seeds` (`src/playlist_web/schemas.py`) → `to_request()`
→ `GeneratePlaylistRequest` (`src/playlist/request_models.py`) → `to_worker_args()`
→ worker `handle_generate_playlist` (`src/playlist_gui/worker.py:1060`) →
`create_playlist_for_artist`. Mirror the checkbox JSX at
`web/src/components/GenerateControls.tsx:285`.

- **Freshness is automatic:** recency exclusion runs pre-order in pool
  construction, so a recently-played hit is excluded before popularity ever ranks
  it. No new wiring. (Do not re-introduce post-order recency filtering.)

**"New Seeds" button** — adds `seed_epoch: int` to the request body (the generator
already consumes it at `playlist_generator.py:1681`). The button increments the
client-side epoch and resubmits the *same* body → a full regenerate where
clusters/slots/popularity all hold and only the top-k random tiebreak rerolls.
Mirror the Generate button at `GenerateControls.tsx:303`.

### Component 4 — Config & backward-compat

New knobs under `playlists.artist_style` (kept with the clustering they modify):

```yaml
medoid_energy_weight: 0.0       # w_energy — spread term (set >0 to enable, once calibrated)
medoid_popularity_weight: 0.0   # w_pop applied when "Popular Seeds" is ON
energy_feature: arousal_p50     # which energy column defines the slots
toptracks_min_artist_tracks: 8  # analyze stage: only fetch top-tracks for artists worth seeding
```

All-zero / checkbox-off ⇒ **byte-identical to today**.

### Component 5 — Eval-gate ("better," measured, not vibes)

Per the evaluation-methodology + playlist-testing skills: A/B through the
**`gui_fidelity` multi-pier harness** (never single-seed, never hand-built
overrides), over a panel of ~15 artists with good Last.fm + energy coverage —
not one cherry-picked artist.

| Class | Metric | Direction |
|------|--------|-----------|
| Spread (energy on vs off) | z(arousal) span across the pier set | should ↑ |
| Banger (popular-seeds on vs off) | mean within-artist popularity rank of piers; fraction of piers in Last.fm top-N | should ↑ |
| Guardrail | worst-edge T and mean T | must not regress (floor-quality, Layer-1 #5) |
| Guardrail | distinct clusters / distinct artists / min-gap | must hold |
| Guardrail | generation wall-clock | ≤ 90s (hard ceiling) |

**Acceptance:** spread ↑ *and* popularity ↑ with transitions and diversity inside
tolerance, **across the panel**. A Nirvana-style hits-vs-deep-cuts spot check is
the qualitative sanity confirmation, but the gate is the measured panel A/B.

---

## Out of scope / deferred

- **Energy-aware *arc ordering*.** Replacing the greedy nearest-neighbor
  `order_clusters` with an energy-monotonic order would make the playlist *move*
  through energy. Deferred — the spread (which tracks become piers) is the
  priority; ordering is a follow-on once spread is validated.
- **Danceability/valence as additional energy features.** Start with
  `arousal_p50` only; `energy_feature` is a config knob so a blend can be tried
  later without code change.
- **Tuning the version-preference cutoffs** in `calculate_version_preference_score`
  (e.g. should `radio edit` / `single version` rank above or below the album cut?).
  The two-stage mbid → loose-title → version-preference resolution is *in scope*;
  only the exact preference weights are a later calibration pass.
- **Final weight calibration** (`w_energy`, `w_pop`, slot spacing). Sensible
  defaults ship off; calibration is a separate eval-driven pass.
- **Legacy DS caller** (`playlist_generator.py:2626`) — confirm whether it needs
  the same energy/popularity threading or is dormant; decide during planning.

## Risks

- **MERT clusters + energy slots assume meaningful arousal variance per artist.**
  Energy-flat artists get an inert energy term (handled) — verify this is graceful,
  not a divide-by-zero.
- **Last.fm coverage / matching sparsity.** Artists missing from Last.fm or with
  unmatched titles get neutral popularity → feature dormant for them (acceptable,
  but the build step must **log match coverage** — fraction of top-tracks resolved
  to a local `track_id`, and how many hits had >1 version-candidate — so we can see
  if resolution is silently dropping hits).
- **Canonical-version resolution is the riskiest matching step.** A wrong group
  (two different songs collapsing to one canonical title, or the canonical bearer
  resolving to a live cut) attaches popularity to the wrong seed. Mitigate with
  mbid-first matching and by logging multi-candidate groups for spot-checking.
- **Stage cost.** One `artist.gettoptracks` call per qualifying artist; rate-limited
  + resumable, so a one-time-ish offline cost. The `toptracks_min_artist_tracks`
  floor bounds it.
- **`metadata.db` is irreplaceable** — the data path writes only to
  `ai_genre_enrichment.db` and a new sidecar npz; nothing touches `metadata.db`.
