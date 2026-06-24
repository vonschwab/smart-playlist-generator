# Roam Corridors — playlist control-surface redesign

**Goal:** Replace the four pool-gating mode sliders (genre/sonic/pace/cohesion) with a control surface where the **seeds define the journey** and the only controls set **how far the bridges may roam beyond the region the seeds establish**, in each of three perceptual dimensions.

**Architecture:** The pier-bridge engine keeps three things always-on — seed-defined structure, monotonic progress between consecutive seeds, and smoothing (no jagged cliffs). The controls become **three per-dimension "roam corridors"** (genre, sonic, energy) implemented as a **soft deviation penalty in the bridge objective** — not a pool gate. Presets bundle the three corridor settings as the everyday UX; a "Custom" view exposes the three dials. The candidate pool becomes broad and never-starve (it supplies, it no longer shapes).

**Tech stack:** Python pier-bridge engine (`src/playlist/pier_bridge_builder.py`, `src/playlist/pier_bridge/beam.py`), the live feature scalars (MERT `X_sonic_mert`, graph genre `X_genre_raw`/taxonomy, Essentia `arousal_p50` via `src/playlist/energy_loader.py`), the dormant `progress_arc_*` machinery, the policy layer (`src/playlist_gui/policy.py`), the web GUI, and the differentiation harness (`scripts/research/slider_differentiation_eval.py`).

## Global Constraints
- **90 s hard generation ceiling.** Corridors are soft; they must never detonate the relaxation/expansion cascade (`feedback_generation_time_budget`).
- **Never-fail on the soft axes.** The pool is broad + never-starve; corridors are relaxable preferences, never hard gates that strand a segment (`feedback_never_fail_three_axes`). Diversity (min_gap, per-artist cap) remains the only hard constraint.
- **The worst edge defines the experience.** Transition quality stays in the objective at all corridor settings (north star #5).
- **Provenance:** sonic = MERT (`X_sonic_variant=mert`), energy = trained `arousal_p50` (NOT loudness), genre = graph authority. Verified before any calibration.
- **Opt-in, reversible.** Ship behind a config flag with the legacy mode-slider path intact until calibration validates, so the two can be A/B'd and rolled back (Layer 4 #22).
- **Diagnostic logging is part of the feature** (Layer 4 #20): every generation logs the realized per-dimension roam (deviation tallies, in-bounds vs out-of-bounds bridge picks) so the corridor's effect is measurable, not asserted.

---

## 1. Problem & evidence

The 72-generation differentiation grid (`docs/run_audits/slider_differentiation/CALIBRATION_FINDINGS.md`) proved the four mode sliders are miscalibrated and often quality-negative: they all **gate the candidate pool**, and the beam then optimizes the same objective regardless, so genre gradations are inert (strict≡narrow; on/off is a cliff), sonic-strict starves the worst edge, and pace's effect on its own signal is inconsistent. **Root cause is mechanism — pool-gating under a dominant beam.** The fix is to move shaping out of the pool and into what the beam optimizes.

A second insight (from design dialogue): imposing an *arc shape* (rising/arch/flat) on a playlist is neither intuitive nor desirable — when seeds are diverse, the journey is **already** built into the seed set. The controls should not dictate a shape; they should let the seeds speak and only widen the bridges' freedom to find surprising-but-good connections.

## 2. The model

### Invariants (always on, not user-controllable)
1. **Seeds are the journey.** Their genres, sonic character, and energy band define the playlist; nothing is imposed on top.
2. **Pier ordering** is optimized for the *smoothest natural sequence* through the seeds (minimize consecutive-seed transition cost) — independent of the corridors. This is the existing seed-ordering search, repurposed to "smoothest sequence" rather than "best mode-scored sequence."
3. **Monotonic progress** between consecutive seeds — each bridge advances toward the next pier in the combined-feature progress metric (the existing pier-bridge progress invariant); no backtracking.
4. **Smoothing always** — per-dimension second-difference (roughness) penalty + slew-rate cap + monotone interpolation, so even very diverse seeds connect without cliffs. (Reuses/extends the dormant `progress_arc_*` machinery from one dimension to three.)
5. **Broad, never-starve pool** — generous, diverse candidate set; the absolute similarity floors are deleted (the deferred Task 4 finds its purpose here). The pool supplies; it does not shape.
6. **Transition quality always defended** — the worst-edge term stays in the objective at every corridor setting.

### The controls — three per-dimension roam corridors
Genre · Sonic · Energy. Each is a scalar `w_d ∈ [0, 1]` (0 = strict/in-bounds, 1 = wide-open) that sets how far a bridge track may deviate **beyond the region the seeds establish** in dimension `d`.

- **Strict (`w_d → 0`, the default end):** bridges stay within the seeds' bounds in `d` — the tightest in-bounds monotonic fill. *"I picked these for a reason; more like this, between these."*
- **Open (`w_d → 1`):** the deviation penalty relaxes, so bridges may reach adjacent territory in `d` (neighboring genres, sonic neighbors, energy headroom) — **only where the transition still holds**.
- **"Cohesion" is emergent**, not a separate axis: all three corridors narrow. Four old sliders → three orthogonal corridors.

## 3. Mechanism — why the corridor is not the inert gate

The corridor is a **soft deviation penalty in the per-edge bridge objective**, applied to the broad pool — not an admission filter. For a candidate bridge track `t` in dimension `d`:

- Define the **seed region** in `d`: for energy (a scalar), the band `[min_seed_arousal, max_seed_arousal]`; for genre/sonic (vector spaces), the neighborhood of the seed positions (distance ≤ r from the nearest seed in that space). The exact region metric and `r` are calibration-determined (§7).
- `deviation_d(t)` = how far `t`'s value in `d` falls outside the seed region (0 if inside).
- `penalty_d(t) = strength_d · softplus(deviation_d(t) − w_d · scale_d)` — i.e., deviation up to the corridor width is free; beyond it is penalized smoothly. `w_d = 0` penalizes any out-of-region pick; larger `w_d` admits progressively more roam.
- The beam edge score becomes: `transition_quality + progress_term − smoothing_penalty − Σ_d penalty_d(t)`.

Because this changes **what the beam optimizes** (the acceptable deviation), not the size of the pool it draws from, it escapes the inert-gate failure mode: at strict the beam is driven to in-region, on-path tracks; as the corridor opens it is permitted to choose surprising off-region tracks that still transition well. The broad pool + never-starve + transition term keep the worst edge defended at all settings (no starvation, unlike old sonic-strict).

Pier ordering is **corridor-independent**: the seeds are fixed points, so the smoothest sequence through them does not depend on how far the bridges roam.

## 4. Per-dimension definitions

| Dimension | Scalar / space | Seed region | Corridor opens toward |
|---|---|---|---|
| **Sonic** | MERT `X_sonic_mert` (768-d) | neighborhood of seed sonic positions | sonically adjacent but not seed-near tracks |
| **Genre** | graph genre / taxonomy (`X_genre_raw` + taxonomy graph) | seed genres + their taxonomy neighborhood | adjacent genres (e.g. chillwave → city pop) |
| **Energy** | Essentia `arousal_p50` (z-scored, via `energy_loader`) | `[min, max]` arousal across seeds | energy headroom above/below the seed band |

Genre and sonic regions are *neighborhoods* (distance-bounded); energy is a *band* (min/max). All three deviations feed the same penalty form (§3) with per-dimension `scale_d`.

## 5. Control surface

- **Presets are the everyday surface.** Each preset is a bundle of the three corridor settings `(w_genre, w_sonic, w_energy)`:
  - *Steady & Cohesive* — all tight (the default).
  - *Adventurous* — all open.
  - *Genre Voyage* — genre open, sonic + energy tight.
  - *Energy Freedom* — energy open, genre + sonic tight.
  - (final set + values are calibration outputs, §7).
- **"Custom…"** exposes the three corridor dials directly.
- Threads through the existing policy layer (`derive_runtime_config`): the preset/dials resolve to `(w_genre, w_sonic, w_energy)` overrides that reach the engine — the same path the harness must use (the artist-mode-policy-layer lesson from `project_slider_calibration`).
- Replaces the four mode sliders in the web GUI (`genre_mode`/`sonic_mode`/`pace_mode`/`cohesion_mode`).

## 6. Engine — reuse vs new

**Reuse:**
- Seed-ordering permutation search (kept; objective simplified to smoothest sequence).
- Dormant `progress_arc_*` machinery (shape/`max_step`/huber/loss) for monotonic progress + smoothing — extended from one dimension to three.
- The three live scalars (MERT, graph genre, `arousal_p50`) and their loaders.
- The broad-pool work already merged (`sonic_admission_percentile`, `min_pool_size` backstop, decoupled genre percentile).

**New / changed:**
- Per-dimension corridor deviation penalty in the beam edge score (`beam.py`) and the seed-region computation in the builder (`pier_bridge_builder.py`).
- Per-dimension smoothing (extend `progress_arc` to genre + energy alongside sonic).
- Delete the absolute similarity floors (`min_sonic_similarity` / `min_genre_similarity`) — the pool is now broad-by-design (the deferred Task 4).
- Policy-layer mapping: preset/dials → `(w_genre, w_sonic, w_energy)`.
- GUI: replace the four mode sliders with the preset selector + Custom dials.
- Remove the now-dead mode-preset pool-gating in `mode_presets.py` (behind the opt-in flag during transition).

## 7. Calibration & validation

- **Re-point the differentiation harness** (`slider_differentiation_eval.py`) at the corridors/presets, on both **artist-mode and seeds-mode**, across the locked breadth-surveyed corpus.
- **Measure, per corridor sweep (strict → open):**
  - *Differentiation* — does opening the corridor move the playlist (track overlap drops)?
  - *Direction* — does opening it measurably increase **roam** (mean distance-from-seed-region; count of genres/energies introduced beyond the seed bounds)?
  - *Quality floor* — does the worst MERT edge stay sane (smoothing + transition term hold it)?
- **Calibration sets the values** that are deliberately left open by this design: the per-dimension `scale_d` and `strength_d`, the seed-region radius `r`, the smoothing `λ`/slew cap, and the preset bundles — including **the everyday default's exact roam** (whether dead-strict or a touch of roam makes the better default is a calibration outcome, not a design decision).
- **Gate:** eval-gate on worst-edge, then blind audition, before the corridors become the default (flip the opt-in flag).

## 8. Phases (decomposition for the implementation plan)

1. **Engine** — broad pool + delete floors; per-dimension corridor penalty in beam + seed-region computation; extend monotonic + smoothing to three dimensions; diagnostic logging of realized roam. Behind an opt-in flag.
2. **Control surface** — preset → `(w_genre, w_sonic, w_energy)` via the policy layer; Custom dials; GUI swap of the four sliders.
3. **Calibration** — harness on presets, both modes, eval-gate, audition; set the open parameters; flip the default.
4. **Deferred (task #67):** valence (mood) + danceability (groove) corridors (dimension-agnostic — drop into the same machinery); draw-the-arc UX.

## 9. Out of scope / deferred
- Valence + danceability corridors and draw-the-arc UX (task #67) — after the core three are proven.
- A real per-dimension *energy contour shape* (rising/arch) — explicitly rejected; the journey comes from the seeds, not an imposed shape.
- The 2DFTM harmony re-fold (task #63) — independent; inert for MERT generation.

## 10. Risks & open questions
- **Seed-region metric for genre/sonic.** Neighborhood radius vs convex region; calibration will choose. Risk: too-tight regions re-create starvation — mitigated by the broad pool + never-starve + soft penalty.
- **Energy band with outlier seeds.** One very-high-energy seed widens the band; may need a robust band (e.g. p10–p90 of seed arousal) rather than raw min/max. Calibration to decide.
- **Smoothing vs roam tension.** Opening a corridor must not reintroduce cliffs; the slew cap + roughness penalty are always on, so roam is *smooth* roam. Verify in calibration.
- **Default strictness** — resolved as a calibration parameter (§7), not left ambiguous.
