# Roam Corridors — playlist control-surface redesign

**Goal:** Replace the four pool-gating mode sliders (genre/sonic/pace/cohesion) with a control surface where the **seeds define the journey** and the only controls set **how far the bridges may roam beyond the region the seeds establish**, in each of three perceptual dimensions.

**Architecture:** The pier-bridge engine keeps three things always-on — seed-defined structure, monotonic progress between consecutive seeds, and smoothing (no jagged cliffs). The controls become **three per-dimension "roam corridors"** (genre, sonic, energy), each a **width-controlled soft corridor around an on-manifold kNN-graph reference path** (not a pool gate), with a min-bottleneck worst-edge guard and hubness-corrected distances. Presets bundle the three corridor settings as the everyday UX; a "Custom" view exposes the three dials. The candidate pool becomes broad and never-starve (it supplies, it no longer shapes).

**Tech stack:** Python pier-bridge engine (`src/playlist/pier_bridge_builder.py`, `src/playlist/pier_bridge/beam.py`), the live feature scalars (MERT `X_sonic_mert`, graph genre `X_genre_raw`/taxonomy, Essentia `arousal_p50` via `src/playlist/energy_loader.py`), the dormant `progress_arc_*` machinery, the policy layer (`src/playlist_gui/policy.py`), the web GUI, and the differentiation harness (`scripts/research/slider_differentiation_eval.py`).

**Mechanism grounding:** validated and refined against four independent literatures in `docs/PLAYLIST_TRAJECTORY_CONTROL_LITERATURE.md` — §3 reflects its four corrections (width-as-knob, on-manifold kNN reference, hubness correction, minimax worst-edge).

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
- **Open (`w_d → 1`):** the corridor widens, so bridges may reach adjacent territory in `d` (neighboring genres, sonic neighbors, energy headroom) — **only where the transition still holds**.
- **"Cohesion" is emergent**, not a separate axis: all three corridors narrow. Four old sliders → three orthogonal corridors.

## 3. Mechanism — a width-controlled soft corridor around an on-manifold reference

Grounded in the literature review (`docs/PLAYLIST_TRAJECTORY_CONTROL_LITERATURE.md`, four independent literatures): the soft corridor is the right family, but with four corrections that each fix a known failure mode. The corridor is a **soft, never-starving preference in the bridge objective** applied to the broad pool — not a pool gate — refined as follows.

**(a) The reference path is on-manifold, not a linear chord.** The "direct path" between two piers is **not** the straight line in feature space — that chord cuts through low-density holes where the nearest real track is perceptually wrong (our timbre-ceiling failure, [[project_timbre_embedding_ceiling]]). Instead the bridge routes over a **distance-weighted kNN graph of the real candidate pool** (Alamgir & von Luxburg: kNN-graph shortest path → manifold geodesic), so every hop is a real, on-manifold track. Deviation is measured relative to that geodesic + the seed region, never to a synthetic midpoint.

**(b) Distances are hubness-corrected.** The kNN graph is built on **mutual-proximity-corrected** distances (Flexer/Schnitzer), at least for the MERT sonic dimension, so high-dimensional hub tracks don't colonize every bridge — the validated music-specific fix for our exact worst-edge risk.

**(c) The knob is corridor WIDTH, not penalty weight.** Each corridor `w_d ∈ [0,1]` maps to a **width** — the kNN `k` / neighbourhood radius / allowed perpendicular distance from the geodesic — a smooth, monotonic primitive. (A penalty *weight* is not: the λ→realized-deviation curve is non-linear and pool-density-sensitive — that was the unpredictability of the old sliders.) Never-fail = **relax the width** in the existing cascade until the beam succeeds; the soft penalty beyond the width → 0 at the boundary, so the pool never starves.

**(d) The worst edge is protected by a minimax term, not an additive average.** An additive penalty in the beam's *sum* objective optimizes the average and will trade one terrible edge for several mild ones. So the objective carries a **min-bottleneck (widest-path) / max-violation term** (or a per-step single-edge cut) on deviation, so no single bridge edge is allowed to crater (north star #5; independently corroborated by Spotify's min-Hamiltonian-path sequencing framing).

Putting it together, the beam edge score is roughly
`transition_quality + progress_term − smoothing_penalty − Σ_d soft_corridor_d(t)`,
with a **secondary minimax sort** on the worst single-edge deviation, all over the broad pool routed on the hubness-corrected kNN graph. `soft_corridor_d(t)` penalizes deviation beyond width `w_d` only; inside the corridor it is free.

This changes **what the beam optimizes** (acceptable on-manifold deviation), not the pool size — escaping the inert-gate failure the grid exposed: strict drives the beam to in-region, on-geodesic tracks; opening a corridor permits surprising adjacent tracks that still transition well.

Pier **ordering** stays corridor-independent: the seeds are fixed points, so the smoothest sequence through them (a min-bottleneck path over the seed-to-seed transitions) does not depend on how far the bridges roam.

## 4. Per-dimension definitions

| Dimension | Scalar / space | Seed region | Corridor opens toward |
|---|---|---|---|
| **Sonic** | MERT `X_sonic_mert` (768-d) | neighborhood of seed sonic positions | sonically adjacent but not seed-near tracks |
| **Genre** | graph genre / taxonomy (`X_genre_raw` + taxonomy graph) | seed genres + their taxonomy neighborhood | adjacent genres (e.g. chillwave → city pop) |
| **Energy** | Essentia `arousal_p50` (z-scored, via `energy_loader`) | `[min, max]` arousal across seeds | energy headroom above/below the seed band |

Genre and sonic regions are *kNN-neighbourhoods on a hubness-corrected (mutual-proximity) graph* of the pool; energy is a *band* (min/max, or a robust p10–p90 of the seed arousals when one seed is an energy outlier). This **unifies with the already-decided genre approach** in [[project_genre_steering_two_system]] (a geodesic router over real genres + a mutual-proximity metric) — the literature confirms the same treatment for the sonic dimension. All three deviations feed the same width-controlled soft corridor (§3).

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
- A **distance-weighted kNN graph of the candidate pool** as the on-manifold bridge reference (corridor width = `k`/radius), built on **mutual-proximity-corrected** distances (hubness fix), at least for the MERT sonic dimension.
- Per-dimension width-controlled **soft corridor** term in the beam edge score (`beam.py`) + the seed-region/geodesic computation in the builder (`pier_bridge_builder.py`).
- A **minimax / min-bottleneck worst-edge term** (or per-step single-edge cut) so no bridge edge can crater (additive penalties alone average it away).
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
- **Calibration sets the values** that are deliberately left open by this design: the per-dimension corridor width→`k`/radius mapping and the soft-corridor slope beyond the boundary; whether mutual-proximity is needed on genre/energy or the sonic dimension suffices; the minimax-vs-additive blend for the worst-edge guard; the smoothing `λ`/slew cap; the robust energy band (min/max vs p10–p90); and the preset bundles — including **the everyday default's exact roam** (dead-strict vs a touch of roam is a calibration outcome, not a design decision).
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
- **Off-manifold reference (resolved by design).** Penalizing deviation from a linear chord would re-import the timbre-ceiling worst-edge failure; the on-manifold kNN-graph reference (§3a) + hubness correction (§3b) address this directly. Residual: the kNN `k` / edge-weighting must keep the graph connected (too sparse → no path) — mitigated by width-relaxation + broad pool + never-starve.
- **Compute cost.** Building/using a kNN graph + mutual-proximity over the pool must stay inside the 90 s budget. Mitigation: the sonic kNN + mutual-proximity can be **library-wide and precomputed/cached**, not per-generation; verify in calibration.
- **Energy band with outlier seeds.** One very-high-energy seed widens the band; may need a robust band (e.g. p10–p90 of seed arousal) rather than raw min/max. Calibration to decide.
- **Smoothing vs roam tension.** Opening a corridor must not reintroduce cliffs; the slew cap + roughness penalty are always on, so roam is *smooth* roam. Verify in calibration.
- **Default strictness** — resolved as a calibration parameter (§7), not left ambiguous.
