# Local genre smoothness via `soft_genre_penalty` recalibration

**Date:** 2026-05-23
**Type:** Design (config-only tuning experiment)
**Status:** Draft — awaiting approval

## Problem

In artist mode (and likely other modes), the playlist occasionally takes a one-track genre detour out of the seed's home genre and immediately back in. The observed case: a Violent Femmes track sandwiched between two dream pop tracks (positions 7–9 of a Sundays-seeded narrow/narrow playlist), with `T=0.836` into it and `T=0.864` out. Sonically the transitions are strong, so the beam picks it; but the genre jump produces a "shuffled, not curated" feel that breaks the playlist's narrative.

User-reported severity: a single jarring transition breaks the playlist — "the worst edge defines the experience" (Layer-1 principle #5).

## Diagnosis

The pier-bridge beam already has a **local genre edge penalty** mechanism:

- `beam.py:986` computes `_get_genre_sim(int(current), int(cand))` — genre similarity between the previous emitted track and the candidate.
- `beam.py:1030` applies `combined_score *= (1 - strength)` when `genre_sim < threshold`.
- Per-mode override resolution exists via `_resolve_mode_number_with_source` (`config.py:268-276`).
- Diagnostics already track per-segment and total `soft_genre_penalty_hits` and `edges_scored`.

The mechanism is **not missing — it's miscalibrated**:

- Current defaults: `threshold=0.20`, `strength=0.10` (all modes).
- Observed edge genre-sim distribution on the Sundays/narrow playlist: `min=0.719, p10=0.733, median=0.866, max=0.922`.
- The 0.20 threshold never fires because every edge is well above it. The mechanism was designed as a safety net against genuine genre conflicts (raw zero overlap), not as a continuity enforcer in the high range where real edges live.

The genre tiebreaker (`genre_tiebreak_weight=0.05`) is too weak to flip selections where the genre-outlier has a meaningful T-score advantage.

## Approach

Recalibrate `soft_genre_penalty_threshold` and `soft_genre_penalty_strength` per mode so the penalty fires on edges that are noticeably below the playlist's typical neighborhood genre similarity, with strength proportional to the tightness mode promises.

Stricter modes (`strict`, `narrow`) tolerate less local genre variation; `dynamic` keeps near-current behavior; `discover` and `off` keep the safety-net-only calibration.

### Proposed per-mode calibration (initial guesses, to be validated empirically)

| Mode      | threshold | strength | Rationale |
|-----------|-----------|----------|-----------|
| `strict`  | 0.82      | 0.40     | Aggressively suppresses anything below the typical narrow median; large multiplier so a marginal T advantage cannot rescue a genre-outlier. |
| `narrow`  | 0.78      | 0.30     | Below the observed median (0.866) but above the observed minimum (0.719) — the Violent Femmes-style edge (likely in the 0.75-0.80 range) gets a real penalty. |
| `dynamic` | 0.55      | 0.15     | Light continuity nudge for the "natural drift" mode; preserves current behavior on most edges. |
| `discover`| 0.20      | 0.10     | Safety-net only — keep current behavior. |
| `off`     | 0.20      | 0.10     | Safety-net only — keep current behavior. |

These are starting points. The calibration loop is part of the work: run representative playlists per mode, inspect `soft_genre_penalty_hits` vs. `edges_scored`, adjust until single-track detours suppress without starving narrow-mode bridges.

### Where the change lives

- `config.example.yaml` — replace the single `soft_genre_penalty_threshold` and `soft_genre_penalty_strength` keys with per-mode variants (`_strict`, `_narrow`, `_dynamic`, `_discover`, `_off`), matching the existing pattern used by `bridge_floor_*` and `weight_bridge_*`.
- `config.yaml` (user's local, gitignored) — same per-mode keys, only if user wants to override the example.
- `docs/PLAYLIST_ORDERING_TUNING.md` — new section ("Local genre continuity") documenting the knob's new role, expected `soft_genre_penalty_hits` ranges per mode, and how to diagnose over- vs. under-pruning.

**No code changes required.** All wiring already exists.

### Validation

Calibration is iterative. For each mode (`strict`, `narrow`, `dynamic`):

1. Generate ≥3 playlists with representative seeds (single-artist, multi-seed, genre-only).
2. Confirm `soft_genre_penalty_hits > 0` (penalty is firing) and `edges_scored` is sane.
3. Confirm playlists still complete without bridge relaxation cascades or starved segments.
4. Spot-check the tracklist: do single-track genre detours feel suppressed?
5. Compare `G genre` quantile stats in the post-generation summary against pre-recalibration baselines.

Acceptance criteria:
- Narrow mode on the Sundays seed no longer produces the Violent Femmes-style one-track detour (or equivalents).
- No regression in `min_transition` or `mean_transition` beyond ~0.02 from baseline.
- No new bridge-relaxation events triggered in narrow mode.
- `G genre min` improves by ≥0.05 in narrow mode without dropping playlist completion rate.

## Risks

1. **Over-pruning in narrow mode.** A penalty that's too strong on a tight pool can starve segments and force bridge-floor relaxation. Mitigation: validate per-mode separately; back off `strength` if relaxation events appear.
2. **Semantic overload of one knob.** The penalty previously served as a safety net against genuine conflicts (raw genre overlap ≈ 0). Recalibrated, it now also enforces continuity. If a future case needs both behaviors at different thresholds, we'll hit the limit and need Strategy B (a separate `local_genre_edge_penalty`). Documented as a known limitation in `PLAYLIST_ORDERING_TUNING.md`.
3. **Per-mode tuning is empirical, not derived.** The proposed initial values are educated guesses from one observed distribution. Real values come from playing with multiple seeds and modes.
4. **Discover/off modes silently keep old behavior.** Intentional — those modes prioritize variety over continuity, so the safety-net calibration is correct for them. Documented.

## Out of scope

- **Strategy B** (separate `local_genre_edge_penalty` mechanism): deferred. Adopt only if A's single-knob semantics become uncomfortable.
- **Strategy C** (adaptive/relative threshold): deferred. Adopt only if fixed per-mode thresholds fail to generalize across seed genres.
- **Strategy D** (per-segment pool tightening at pre-beam): deferred. Adopt only if in-beam penalty proves too late.
- **Strategy E** (`genre_tiebreak_weight` bump): superseded by A. Tiebreaker stays at 0.05.
- **Macro genre arc** (genre piers, target curves): explicitly not in scope per the brainstorm — user opted for local smoothness only.
- **GUI exposure** of per-mode knobs: stays in `config.yaml` for now.

## Open questions

None blocking. Calibration values are guesses; the work *is* finding the right ones.
