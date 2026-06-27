# Design: fix the centered transition score (v6 sonic) with a calibrated sigmoid

**Date:** 2026-06-25
**Status:** design — awaiting review
**Supersedes/implements:** `docs/HANDOFF_2026-06-25_sonic_centered_transition.md`
**Memory:** `project_centered_transition_rescale_flaw`

---

## 1. Problem (measured)

The beam scores each playlist edge (track A → track B) with a transition score `T`. With `center_transitions=true` (the production default — `config.yaml:96`), `T` is built from cosine similarities that are squashed into `[0,1]` by:

```python
r(x) = clip((x + 1) / 2, 0, 1)        # transition_metrics.py:_rescale_centered_cos
```

This maps the *theoretical* cosine range `[-1, +1]` onto `[0, 1]`. But real edge cosines never use that range. Measured over 60 destinations, each restricted to its realistic top-2000 sonic-neighbor pool, the dominant end→start cosine band is:

```
p1 = 0.138,  p50 = 0.269,  p99 = 0.501   (min 0.042, max 0.709) — entirely positive
```

So `(x+1)/2` crushes the real operating band `[0.14, 0.50]` into `[0.57, 0.75]`. Consequences, reproduced firsthand with the production scoring math (`scripts/research/sonic_centering_probe.py`):

| | raw cosine | legacy centered `T` |
|---|---|---|
| good edge (Beach House → RD "Heaven's on Fire") | 0.260 | 0.625 |
| bad edge (Yuji Nomi orchestral cue → same) | 0.151 | 0.579 |
| good-vs-bad separation | **72%** | **8%** |
| field median | 0.021 | **0.498** |

Half of all candidate edges read as `T ≥ 0.5` ("fine"), and the safety floor that should reject weak edges (`transition_floor ≈ 0.45`) sits **below the entire compressed field**, so it never fires. This is the **sonic-side root cause** of the symptom chased on the genre side: an orchestral film cue placed next to an energetic indie single scored as an acceptable transition.

## 2. Root cause

This is textbook **embedding anisotropy**: learned audio embeddings (MERT) concentrate vectors in a narrow cone, so cosines between real tracks bunch in a narrow positive band. The standard mitigations are mean-centering / whitening / **z-scoring** the score, then calibrating it. The `(x+1)/2` rescale is an *un-calibrated* affine map to the wrong (theoretical) range — it preserves ordering but adds a large constant offset and halves the scale, so variation is swamped and floors can't bite.

Confirmed it is **offset/scale, not monotonicity**: measured across remaps, the good-vs-bad separation *as a fraction of the used output range* (`gap/poolrange`) is ~identical (0.385–0.41) for every monotonic remap. Legacy separates as well as anything *within its range*; it just parks the whole field high. Any de-offsetting, range-stretching remap fixes it.

## 3. Goal & success criteria

Replace the rescale so a genuinely bad sonic transition scores meaningfully worse than a good one, keeping `T ∈ [0,1]`. **This is designed for roam** (corridors + worst-edge minimax), the target topology: roam's minimax optimizes on this `T`, so a discriminating `T` is exactly what makes the worst-edge guard meaningful. Accordingly we **remove the `transition_floor` hard gate** rather than recalibrate it — roam shapes candidates with a soft corridor penalty and minimizes the worst edge, it does not eliminate. Ship the calibrated score **activated by default**, and remove dead/duplicate wiring while we're in it.

Success = the verification gate (§7) passes **with roam enabled**: gap restored toward 72%, median off ~0.5, suite green, and a real roam generation log + audition confirm the worst edge improves and is no longer washed out.

**Dependency (locked, structural).** This sonic fix is the blocker for the genre soft-metric calibration: the genre floor/strength tune against *playlist quality*, which is confounded until this `T` discriminates — calibrating sooner fits to noise that is about to change. Order: **sonic fix lands → genre calibration → merge.** Do NOT flip roam to default or delete the legacy beam here; that is a separate roam-promotion effort this *unblocks*, and coupling it would delay the genre dependency.

## 4. Design — calibrated sigmoid (Platt-style)

### 4.1 The remap

Replace `(x+1)/2` with a **calibrated logistic** — standardize the cosine to its operating band, then squash:

```python
r(x) = 1 / (1 + exp( -gain * (x - center) / scale ))
```

This is Platt scaling (`σ(w·x + b)` with `w = gain/scale`, `b = -gain·center/scale`). It removes the offset, stretches the real band across `[0,1]`, is smooth (no hard clips → no ties at the extremes, honoring design principle #19), and stays in `[0,1]`.

Measured against the probe: restores the good-vs-bad gap to **88%** (vs legacy 8%, raw 72%), median `T` ≈ 0.30, strong edges ≈ 0.70.

### 4.2 Parameters: fixed, not re-fit per run

`center`, `scale`, `gain` are **constants derived once** from the library's centered end→start cosine distribution and stored in config. *Not re-fit per run* — per-run fitting would reintroduce the percentile failure mode (non-stationary scores, floors stop meaning anything, a segment of all-bad candidates still emits a top edge ≈ 0.9). The cosine distribution is a stable property of the embedding; if the library is ever re-analyzed, we re-derive these three numbers — same discipline as the whiten transform.

Starting values from the measured band (to be confirmed by the calibration script on the full library, then perceptually):
- `center` ≈ 0.32 (midpoint of the [p1, p99] band)
- `gain/scale` ≈ 16 (so p1 → ~0.05, p99 → ~0.95)
- equivalently `center = 0.32, scale = 0.0625, gain = 1.0`

Calibration **target**: map the operating band [p1, p99] to ≈ [0.05, 0.95], so the median realistic edge lands ≈ 0.30 and strong edges ≈ 0.70. This is a judgment call on aggressiveness — confirmed in the verification gate (a too-steep `gain` over-saturates; the literature's Platt over-confidence caveat).

### 4.3 Where it lives & the config knob

- `src/playlist/transition_metrics.py`: `_rescale_centered_cos` becomes the calibrated logistic (renamed `_calibrate_transition_cos`). Applied **per component** (end→start, mid→mid, full→full) with **shared** params (calibrated on the dominant 0.70-weight end→start band). Per-component params are the documented fallback if verification shows mid/full are poorly served.
- New `transition_calibration` block in `config.yaml` + `config_loader.py` + the pier-bridge config: `center`, `scale`, `gain`. **No `mode: legacy` switch** — sigmoid is the only behavior (per "ship activated, no dead wiring"). A configured calibration that fails to load is a **startup error**, not a silent fallback (project rule).
- Calibrated on **centered** cosines (production mean-centers the clip matrices first; calibrate on what production produces).
- The `center_transitions` flag and its raw branch (`t_val = t_raw`) **stay** — the flag is a legitimate toggle (`True` in prod, `False` in some tests), not dead wiring. Only the rescale *function* in the `True` branch changes.

### 4.4 Calibration + tuning deliverable

A `scripts/research/` script that computes the library cosine band → emits proposed `center/scale/gain` → sweeps the `T`-consuming floors against the new `T` distribution and reports recommended values. This *is* the tuning recipe (principle #23) and is re-runnable after any re-analysis.

## 5. Remove the transition hard gate (roam-only)

Roam avoids hard gates: it shapes candidates with a soft corridor penalty (`corridor_penalty`, slope-beyond-width) and minimizes the worst edge (`_state_min_edge`), never eliminating. The legacy `transition_floor` is a hard gate (`is_broken_transition`: `T < transition_floor` → broken; `beam.py:679` direct-floor reject) that also triggers the gate-and-expand cascade. With a discriminating `T`, the beam objective *and* roam's worst-edge minimax already prefer good edges by optimization — eliminating candidates only adds cascade/90s-budget risk. And because the rescale changes `T`'s scale, leaving `transition_floor=0.2` would *accidentally* flip a dead floor (rejected ~0% under the old compressed `T`) into a ~28%-rejecting gate — the opposite of intent.

**So: remove the `transition_floor` hard gate** — drop the `T < transition_floor` check from `is_broken_transition`, the direct-floor reject (`beam.py:679`), and the now-dead `transition_floor` config knob (`config.yaml:99`, `PierBridgeConfig`, the `default_ds_config` resolver). No recalibration. Removing it is strictly *fewer* hard rejections → *fewer* cascade triggers → budget-safe.

**Keep** the `−0.5` catastrophic anti-alignment gate (`centered_cos_floor`, `beam.py:638`) — it gates the **raw** `T_centered_cos`, not the remapped `T`, is essentially inert in the anisotropic MERT space, and is a cheap "edge is actively opposite" safety assertion.

**Out of scope (roam-promotion, deferred):** the other legacy gates (`bridge_floor` on raw pier-sim `beam.py:1209`, the relaxation cascade, pool-size caps) and flipping roam to the default. Removing those + promoting roam is the next effort this unblocks; doing it here couples a calibration-gated migration to the metric fix and delays the genre dependency.

## 6. Cleanup — wall-scoped (verified live-vs-dead)

Confirmed by a read-only call-graph trace. **Live scorer is `transition_metrics.py::score_transition_edge`** (beam builds `TransitionMetricContext` directly at `pier_bridge_builder.py:730`, always non-None).

### 6.1 In scope (this PR)
1. **The fix** (§4).
2. **Collapse 3 rescale copies → 1 source of truth:**
   - Repoint the opt-in audit (`pier_bridge/audit_summary.py:93-98`) to call `score_transition_edge` instead of `vec.py::_compute_transition_score_raw_and_transformed`. *Also fixes a latent bug:* post-fix the audit would otherwise compute `T` differently from the beam.
   - Delete the `pier_bridge/vec.py` rescale copies (`_compute_transition_score`, `_compute_transition_score_raw_and_transformed`) + their back-compat wrappers (`pier_bridge/config.py:373,392`) + the dead fallback branch (`beam.py:643-654`, unreachable because context is always non-None).
   - Delete the production-dead `scoring/transition_scoring.py` module (imported only by `tests/unit/test_scoring.py`); repoint those tests onto `score_transition_edge`.
3. **Delete the vestigial `T_used` field** (`transition_metrics.py:182` — written, read nowhere).
4. **Remove `transition_gamma`'s dead local storage in `transition_metrics.py`** (the unused context field + `edge["gamma"]` there). It is never applied in the live scorer.
5. **Remove the `transition_floor` hard gate** (§5) + **regenerate affected goldens** (config goldens + any playlist-`T` goldens), confirming the only diffs are the intended `T`/gate-removal/cleanup changes.

### 6.2 Preserve (verified load-bearing — do NOT touch)
- `T_centered_cos` + the `-0.5` anti-alignment gate (raw cosine, independent of the remap).
- `S` field (consumed by `_layered_transition_delta`, `beam.py:412`).
- `build_hybrid_embedding` inside `build_transition_metric_context` (feeds the reporter's `H` display).
- The `center_transitions` flag + raw branch.

### 6.3 Deferred (flagged, NOT this PR)
- Full `transition_gamma` removal end-to-end (config plumbing, worker, reporter, `ds_pipeline_runner`) and the dead single-seed `constructor.py` / `pipeline.py` construction path (`pipeline/core.py:529 "if True: # Always use pier-bridge"`). This belongs to the existing dead-code cleanup program — pulling an entire legacy path on a metric-fix PR is scope creep with regression risk.

## 7. Verification gate (all must pass before "done") — with roam enabled

Roam is the target path, so judge quality there: enable `roam_corridors_enabled` + `worst_edge_minimax_enabled` (via `overrides[pier_bridge][roam]`) for the generation checks.

1. **Config sanity:** confirm `center_transitions=True` AND the roam flags resolve in a real generation log (`pier_bridge_builder.py` logs them) — insurance against this codebase's "wired but inert" history.
2. **Probe:** good-vs-bad gap restored toward 72% (measured ~88%), median `T` off ~0.5; **rank fidelity** — blended-`T` ordering tracks the raw blended ordering (Spearman); the Yuji edge is demoted (no longer a top neighbor of the destination).
3. **Full fast suite green** (`python -m pytest -q -m "not slow"`, run directly, bounded — never piped through `tail`). Goldens regenerated deliberately; diffs confirmed to be only intended changes.
4. **Real roam generation + READ THE LOG** through the `gui_fidelity` harness with roam enabled and `pier_bridge.emit_selected_edge_audit: true` — confirm from the per-edge audit that the **worst edge** improved and the selected-edge `T` distribution is de-compressed (a summary metric can't distinguish a true null from a knob that didn't apply). Confirm generation < 90s.
5. **Perceptual audition** on a few seeds (roam) — does the worst edge actually sound better.
6. **Hand off to the genre session:** a discriminating sonic objective under roam is the result the genre calibration is structurally blocked on.

## 8. Process

- Implement in a **dedicated git worktree on its own branch** (this spec is committed there as the first commit). Subagents launch in the MAIN checkout — so any subagent work is either inline-in-worktree or cd-guarded + branch-verified (`feedback_subagents_run_in_main_checkout`).
- **Merge order:** this sonic fix merges **first**; the parallel genre soft-cosine session then calibrates its penalty against a sound sonic objective (avoids the moving-target trap).
- Ship **activated by default**.

## 9. Research grounding & honest caveats

- **Grounding:** the diagnosis (anisotropy → narrow positive cosine band) and the remedy (z-score + logistic = Platt scaling; monotonic transforms to calibrate cosine for a threshold) are standard ([anisotropy](https://dev.to/gabrielanhaia/cosine-similarity-lies-heres-what-to-use-when-your-embeddings-all-cluster-at-085-3dfe), [Platt scaling](https://changyaochen.github.io/platt-scaling/), [Relevance Filtering for Embedding Retrieval, arXiv:2408.04887](https://arxiv.org/html/2408.04887v1)).
- **Caveat 1 — unsupervised calibration.** Classic Platt fits the logistic *supervised* on labeled relevant/irrelevant pairs. We have no transition labels, so ours is a *distribution-based* calibration (spread the observed band across [0,1]). Defensible and sufficient for restoring discrimination; if we later want true probability calibration, we'd harvest labeled edges from auditions.
- **Caveat 2 — don't over-steepen.** A too-large `gain` over-saturates (Platt over-confidence). The calibration targets band → [0.05, 0.95], not harder, confirmed perceptually.
- **Caveat 3 — attribution.** The *compression* is measured and definitive. That it was THE deciding factor for the specific Song-of-Baron placement (vs. genre-off + diversity in that run) still wants a real per-segment log to confirm — part of the gate (§7.4). The flaw is real regardless.

## 10. Risks

- Removing the transition gate is strictly *fewer* hard rejections, so it can only *reduce* cascade triggers — budget-safe. The residual risk is the opposite: with no floor, a sparse segment's best-available edge is accepted as-is; roam's corridor + worst-edge minimax is the intended quality mechanism there, verified under roam (§7). Quality should therefore be judged with roam **on**, not on the legacy beam.
- Per-component vs shared calibration: if mid/full cosine bands differ materially from end→start, shared params under-serve them. Verification checks this; per-component is the fallback.
- Goldens: many fixtures encode `center_transitions`/`T`; regeneration must be deliberate and diff-audited.
