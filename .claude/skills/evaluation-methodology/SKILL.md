---
name: evaluation-methodology
description: Use when designing, running, or reporting any similarity/ranking/A-B evaluation — sonic or genre audition, whitening or transform validation, artifact A/B comparison, threshold or floor calibration — or before presenting any quantitative conclusion that one embedding, matrix, or config beats another. Also use when building or extending an evaluation/audition harness, and BEFORE reporting that a space, artifact, or metric has collapsed, regressed, or silently degraded — a health/collapse claim is an evaluation and needs the same controls.
---

# Evaluation methodology

**Methodology validity is the deliverable.** A wrong-but-confident comparison costs more than no comparison: it burned a full session when whitening was "validated" against a seeds-only pool, and another when a production NPZ had been silently overwritten so A-vs-B was actually A-vs-A. Run the pre-flight checklist before trusting any number you produce, and the reporting rules before presenting it.

## Pre-flight checklist (before trusting any number)

1. **Full pool, not samples.** Rank/score against the complete scanned library (the full `beat3tower_32k` artifact, ~32k tracks), never a seeds-only or convenience subset. A sample "validation" is how the whitening evaluation went wrong. The full-pool run is minutes, not hours — pay it.
2. **Verify data provenance of both arms.** Before comparing A vs B, confirm each side is what you think it is: file mtimes, array shapes, and a fingerprint (hash a few rows) against expectations. We once compared against a production NPZ an experiment had overwritten — the comparison was A-vs-A and the conclusion was garbage.
3. **Experiments never write to production artifact paths.** Outputs go under `docs/run_audits/<experiment>/` (the audition convention) or an explicitly experimental path. Anything that must touch `data/artifacts/` requires the timestamped-backup discipline in CLAUDE.md. If you discover a production path *was* overwritten, treat it as a live incident, not just an invalid experiment — the runtime may be serving the experimental data right now. Check what's live and restore from backup before anything else.
4. **Generation-based evals go through the fidelity harness.** Multi-pier seeds via `generate_like_gui` — see the **playlist-testing** skill. Single-seed topology produces pathological segment structures that invalidate calibration.
5. **Blind the perceptual arm.** If a human (or LLM judge) compares quality, they must not know which arm is which, and include a decoy arm to measure discrimination (the audition pattern — see Existing harnesses).
6. **Check the metric isn't circular.** Don't validate a space with a metric computed in that same space (genre QC scored inside the same enriched-genre space said everything was fine while the tags were junk). At least one arm of the evidence must be independent: human ears, held-out labels, a different modality.
7. **Reconcile contradictory probes before trusting either.** If two measurements of the same space disagree, the contradiction IS the finding — resolve it before reporting anything. The 2026-06-24 "MERT collapse" false alarm shipped a catastrophic conclusion from the scary probe (rank 21,525) while a healthy probe (rank 31) on the same bytes sat unreconciled; the scary probe had a selection bug (first-track instead of max-over-seed-tracks).
8. **Health claims need a null baseline and the runtime's own metric.** "Collapsed/degraded" is an evaluation: compare same-artist (or golden-neighbor) stats against a **random-pair baseline** (the missing control in the MERT false alarm — healthy gap was +0.214 vs random), and measure what the runtime actually consumes (rank via max over seed tracks), never raw cosine against a fixed target — whitening intentionally recenters cosines to ~0, so a cosine "drop" is expected, not damage.

## Reporting rules

- **Distributions, not means.** Report min / p10 / p50 / p90. Centered or saturated metrics hide floor failures — T-centering masked bad edges; hub-genre saturation put random-pair G at p50 ≈ 0.414. The worst edge defines the experience (north star #5).
- **State N, and don't overgeneralize.** "+0.033 on 5 seeds" is a deferral, not a win — say so. Name the seed count and pool size in every conclusion.
- **Re-run the winner against the full pool before declaring.** A candidate that won on the dev subset must be confirmed on the complete library.
- **List the assumptions that could invalidate the comparison** (stale sidecar, vocab drift, mismatched normalization, recently rebuilt artifact not reloaded) — one line each, checked or unchecked.

## Red flags — stop and re-verify

- "The sample is representative enough" → run the full pool.
- "The production file is the convenient output path" → it's how we destroyed a comparison AND possibly what production is now serving; write to `run_audits`.
- "Mean improved" → check p10 and min first.
- "QC says it's fine" → is QC measuring the same space you changed?
- "I know which arm is which but I'll be objective" → blind it or don't report it.
- "Preliminary / directionally promising" framing of an unsound result → an unsound result is not reportable at any confidence level.
- "Two probes disagree — the alarming one must be right" → reconcile first; same bytes + corrected method may be healthy.
- "Raw cosine dropped after whitening → collapse" → whitening recenters cosines; check rank/discrimination against a random-pair baseline instead.
- "One ad-hoc number justifies an incident report" → a collapse/regression claim gets the full pre-flight plus a second independent probe before any incident doc, memory, or recovery plan is written.

## Existing harnesses (adapt, don't rebuild)

- **Sonic audition:** `scripts/research/sonic_audition_build.py` / `_serve.py` / `_analyze.py`; findings pattern in `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`.
- **Genre audition:** `scripts/research/genre_audition_build.py` / `_serve.py` / `_analyze.py` (graph vs co-occurrence vs decoy); spec `docs/superpowers/specs/2026-06-09-genre-similarity-audition-design.md`, plan `docs/superpowers/plans/2026-06-10-genre-similarity-audition-harness.md`.
- **Output convention:** manifests + `index.json` under `docs/run_audits/<experiment>/`.

## Maintenance protocol

When an evaluation turns out to be invalid for a *new* reason, add the failure to the pre-flight checklist and a matching red flag. This skill is the index of how we evaluate, not a one-time doc.
