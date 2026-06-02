# Genre Edge Safeguards & Steering — Design

**Status:** Design approved 2026-05-30. Approach A (genre floor + first-class genre term in the beam edge score).
**Author context:** Drafted after tracing why The Smiths → Ramones / Halo Benders transitions occur. Root cause confirmed in code: the pier-bridge beam orders interior tracks ~95% on sonic criteria, while genre enters only as a 0.05 additive tiebreak + a weak soft penalty — and that genre signal runs on the **sparse** `X_genre_smoothed`, not the dense PMI-SVD embedding fixed earlier (see `[[project-genre-embedding-anisotropy]]`). A historical genre hard gate (`genre_conflict_*`) was deleted in commit `7c4f201`. This design reintroduces a genre safeguard + steering, on the dense scale, in the ordering stage.

## 1. Problem

The genre-embedding fixes made the candidate **pool** genre-coherent (admission). But the beam's **ordering** within that pool is sonic-dominated:

```
# beam.py:_beam_search_segment, ~967
combined_score = weight_bridge·bridge_score + weight_transition·trans_score   # both SONIC
combined_score += genre_tiebreak_weight(0.05)·genre_sim                       # tiny
# then: if genre_sim < genre_penalty_threshold:  combined_score *= (1 − strength)  # weak
```

- `bridge_score` = harmonic mean of sonic similarity to the two surrounding piers (rewards "in-between" tracks).
- `trans_score` = sonic transition smoothness.
- `genre_sim` = cosine on `X_genre_norm` = normalized **sparse** `X_genre_smoothed` — NOT the dense embedding.

Consequence: within a genre-clean pool, a stylistically-off track that is a good *sonic* bridge (e.g. Ramones between two Smiths piers) wins the slot, because the coarse sonic space can't distinguish jangle-feel from punk-feel and genre barely gets a vote. The monotonic-progress sonic mechanism is **intentional and correct**; the gap is that genre no longer guards or steers the ordering.

## 2. Goal & non-goals

**Goal:** Reintroduce, in the beam edge-scoring stage, (a) a hard genre **floor** (safeguard) that forbids egregious cross-genre transitions, and (b) a first-class genre **steering** term, both computed on the dense embedding. Net effect: genre-appropriate tracks (Ducks Ltd, The Sundays for a Smiths seed) displace stylistically-off-but-sonically-plausible tracks (Ramones, Halo Benders) in segment interiors, without breaking feasibility for genre-sparse seeds.

**Non-goals (tracked separately):**
- General sonic-scoring re-examination (the monotonic-progress / bridge / transition formulas) — a **separate future initiative**. Untouched here.
- The pier-dedup spacing bug (5 vs 6 piers; two cluster-medoid anchors collapse via `dict.fromkeys` at `pier_bridge_builder.py:324`) — real but orthogonal; its own small fix.
- Mode admission-floor recalibration (broader genre tuning) — separate; edge-floor calibration here will inform it.

## 3. Architecture

All changes live in the pier-bridge beam edge scoring. The sonic mechanism is untouched.

### 3.1 Genre signal switch (enabling change)
- Today the beam's `_get_genre_sim` runs on `X_genre_norm` (normalized sparse `X_genre_smoothed`). Route it to the **dense embedding** (`bundle.X_genre_dense`, already L2-normalized, and now carried through `restrict_bundle` per `[[project-genre-embedding-anisotropy]]`).
- Sparse remains the fallback when no dense sidecar is present.
- Edge genre similarity for an edge is `dense(current_track, candidate)` — consecutive-track coherence — computed at each beam step (`beam.py` ~949) and at the final-pier connection (~1412), the same two points the sonic transition is scored.
- **Genreless endpoints:** if either track's dense vector is ~zero (~3% of library, mostly pool-excluded), skip both the floor and the steering term for that edge and fall back to legacy scoring (can't assess genre → don't punish; no new infeasibility from missing data).

### 3.2 Floor (safeguard)
- In the beam candidate loop, after computing `genre_dense_sim`, reject the candidate (`continue`) if it is finite and `< genre_edge_floor`. This is the reincarnation of the deleted `genre_conflict` gate, on the dense scale. Applied at interior steps and the final connection.
- `genre_edge_floor` is keyed by `cohesion_mode` (strict/narrow tighter; discover loose), config-overridable.

### 3.3 Relaxation backstop
- Reuses the established pattern from the `transition_floor` relaxation work: if a segment goes infeasible, progressively step `genre_edge_floor` down toward `min_genre_edge_floor` before declaring failure.
- Gated by `infeasible_handling.enabled` (new sub-fields `genre_floor_relaxation_enabled`, `min_genre_edge_floor`). Genre-sparse seeds (e.g. Charli XCX) degrade gracefully instead of erroring.

### 3.4 Steering term (first-class genre factor)
- Replace the 0.05 additive tiebreak with a three-way blend that sums to 1:
  ```
  score = w_bridge·bridge_score + w_transition·trans_score + w_genre·genre_dense_sim
  ```
- Per-mode weights, renormalized. Example (illustrative; final values from calibration):
  - narrow: bridge 0.70 / transition 0.30 → bridge 0.55 / transition 0.25 / **genre 0.20**
  - tighter modes lean genre harder; discover keeps genre light.
- **Retire the overlapping soft penalty in the new path.** The multiplicative `×(1−strength) if genre<threshold` triple-counts genre once a hard floor and a continuous term exist. It is disabled in the new path; the floor handles "too low," the term handles steering. It stays intact in legacy mode.

## 4. Configuration & backward compatibility

- **Master flag `genre_steering_enabled`** (project opt-in convention):
  - OFF → exact legacy behavior (sparse genre, 0.05 tiebreak, soft penalty, no floor).
  - ON → dense signal, three-way weights, hard floor + relaxation, soft penalty disabled.
- **Code default OFF; `config.yaml` default ON** — the same pattern `infeasible_handling` uses. Tests and other callers preserve legacy; the live app gets the new behavior.
- **New per-mode knobs:** `weight_genre_<mode>`, `genre_edge_floor_<mode>`; plus `min_genre_edge_floor` and `genre_floor_relaxation_enabled` under `infeasible_handling`. All documented with a tuning-recipe note (tunability principle).
- Resolved in `config.py::resolve_pier_bridge_tuning`; new fields added to `PierBridgeConfig`.

## 5. Calibration

- Set floor/weights on the dense scale empirically via `scripts/research_genre_similarity.py`: same-scene indie pairs sit ~0.85; cross-scene (punk vs jangle) much lower/negative. Choose `genre_edge_floor` to sit in that gap (separating "same scene" from "adjacent-but-off"); set `weight_genre` per mode; validate on the three reference seeds (Charli, Acetone/Smiths-like, Beach Boys).
- This is **edge-floor** calibration — narrower than, but feeding, the deferred mode-admission-floor recalibration.

## 6. Testing

- **Unit (beam):**
  - Below-floor edge is rejected (`continue`).
  - With sonic scores ~tied, the higher-dense-genre candidate wins (steering).
  - Relaxation lowers the floor when a segment is infeasible and `infeasible_handling.enabled`.
  - A genreless endpoint skips the floor/term (fallback), no rejection.
  - Flag OFF reproduces legacy edge scoring exactly.
- **Integration:**
  - The Smiths / narrow: Ducks Ltd / The Sundays displace Ramones / Halo Benders in segment interiors; mean per-edge dense genre-sim measured before/after.
  - Charli XCX / narrow: still feasible (relaxation fires).
  - Pains of Being Pure at Heart: stays coherent (no regression).
- **Regression:** full `pytest -m "not slow and not gui"` green.
- **Success criterion:** named misfits disappear from the Smiths interior; mean per-edge dense genre-sim rises, especially the weakest edges clearing the floor.

## 7. Affected files (anticipated)

- `src/playlist/pier_bridge/beam.py` — dense genre signal, floor gate, three-way steering term, retire soft penalty in new path.
- `src/playlist/pier_bridge_builder.py` — pass dense genre matrix to the beam; genre-floor relaxation in the segment backoff loop.
- `src/playlist/config.py` — resolve new per-mode knobs + flag.
- `src/playlist/run_audit.py` — `InfeasibleHandlingConfig` sub-fields (`genre_floor_relaxation_enabled`, `min_genre_edge_floor`).
- `config.yaml` — defaults ON with per-mode values.
- Tests: `tests/unit/` (beam edge scoring), `tests/integration/` (reference-seed coherence).
