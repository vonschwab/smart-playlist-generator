# Sonic Phase 2 — Harmony Diagnosis & Richer-Feature Investigation

**Date:** 2026-06-02 (investigation); 2026-06-03 (head-to-head audition + rebuild — see §12)
**Branch:** `sonic-harmony-keyinvariant` (off `master`)
**Status:** **SHIPPED.** Full 40k 2DFTM extraction complete; blind A/B audition confirmed the
win; harmony tower rebuilt into the production artifact (86→162 dim). See §12.
**Scope:** Validate the corrected (tower-weighted) sonic space by ear, localize the
weakest tower, and decide whether to act. Read-only throughout — no writes to
`metadata.db`, no writes/moves of audio files; all extraction is cached to sidecars.

---

## TL;DR

1. **Root cause of weak harmony found and confirmed:** the harmony tower encodes
   **absolute key** (pitch-class energy), which is *noise* for harmonic-character
   similarity — not a hub, not anisotropy, not merely low dimensionality. In isolation
   it is **anti-correlated** with human judgement (Spearman **−0.144** vs. verdicts).
2. **A validated fix exists:** a key-invariant descriptor — the **2D Fourier Transform
   Magnitude (2DFTM)** of the chromagram — flips harmony from −0.144 to **+0.210**, and
   improves the full blend **+0.308 → +0.361**.
3. **Rhythm needs nothing.** The current rhythm tower is already healthy (isolated
   **+0.277**); richer rhythm features do not help the blend and *hurt* when combined
   with the harmony fix. The "rhythm is the same bug" hypothesis was **falsified**.
4. **The shipped 0.30 harmony weight is miscalibrated** for a noisy tower; dropping it
   to 0.20 lifts the blend +0.308 → +0.328 via a cheap re-weight (no re-extraction).
5. **Decision deferred.** The full 2DFTM rebuild's marginal value over the free
   re-weight is only **+0.033** on **5 seeds / 1 rater** — at the resolution limit of the
   current ground truth. Recommendation: gather ~6–8 more audition seeds (harmonic
   corners + a J Dilla recheck) before committing to a 3-hour re-extraction.

---

## 1. Methodology

### 1.1 Ground truth — blinded audition

Per seed, the harness ([`scripts/research/sonic_audition_build.py`](../scripts/research/sonic_audition_build.py)
/ `sonic_audition_serve.py` / `sonic_audition_page.html`) computes the union of
top-15 nearest neighbours across several sonic spaces, **deduplicates and shuffles
them with the originating space hidden**, and the listener rates each unique track:

| verdict | score | meaning |
|---|---|---|
| `match` | 3 | belongs next to the seed |
| `close` | 2 | plausible, same neighbourhood |
| `off` | 1 | you can hear *why* it was picked, but it doesn't fit |
| `wrong` | 0 | no audible sonic relationship |

Blinding is the key: the listener describes each track once, unbiased by which space
proposed it; the originating space + rank + cosine are re-attached on save. Spaces
audited: `full_track`, `production_transition`, `timbre`, `harmony` (rhythm-only was
dropped — see §4.1). Verdicts are stored in `docs/run_audits/sonic_audition/*_capture.yaml`.

**Seeds rated (5, 250 verdicts):** Green-House (synth ambient), J Dilla (hip-hop),
Duster (slowcore), Real Estate (jangle/dream-pop), Jean-Yves Thibaudet (solo classical
piano — the decisive harmonically-rich seed).

### 1.2 The probe — quantitative representation test

For any candidate feature representation, the metric is:

> per seed, **Spearman correlation between (seed→track cosine) and (verdict score)**,
> averaged across seeds.

Higher = the representation ranks perceptually-similar tracks closer. Features are
z-scored across the pool then L2-normalised before cosine, so each representation is
compared on equal footing. This single number lets us rank rival feature designs
against the listener's ears, cheaply, without rebuilding the artifact.

### 1.3 The gate framework

The expensive step (re-extracting all ~40k tracks, ~3 h) is gated behind cheap tests:

- **Probe** — does a richer harmony representation beat the current tower *in isolation*?
- **Gate 1** — does it beat the current tower *in the blend* (rhythm 0.20 / timbre 0.50 /
  harmony 0.30)? (Timbre may already mask harmony.)
- **Gate 2** — does richer rhythm help, so a single re-extraction can fix both towers?
- **Weight sweep** — is the harmony blend weight optimal?

All gates run on the 286 already-extracted rated tracks; none require touching the library.

---

## 2. Finding A — Audition (qualitative)

| seed | character | read |
|---|---|---|
| Green-House | synth ambient | Healthiest; **timbre-dominant**, self-identifies sonically. |
| Duster | slowcore | Healthy; `full_track`/`timbre` carry every match. Off/wrong theme = **slow-ambient bleed** (slow+quiet ≠ genre). |
| Real Estate | jangle/dream-pop | Healthy; **harmony threw high-cosine wrongs** (solo piano 0.785, Motown 0.760). Timbre's #1 was a soul track (warm-vintage blind spot). |
| J Dilla | hip-hop | Noisiest (~40 % wrong); matches real but low-ranked. Hip-hop identity is production/cultural, not sonic. |
| **Jean-Yves Thibaudet** | **solo classical piano** | **Decisive.** `timbre`/`full_track` flawless (15/15, 14/14 matches). **Harmony still broken** even here: r1 Hüsker Dü 0.854, r2 Beach Boys 0.844 — both *wrong*. |

The Thibaudet result killed the benign explanation ("harmony is fine, just
non-discriminative for simple-harmony genres"): the seed that should be harmony's best
showcase still returns punk and chamber-pop at 0.85.

**Two distinct tower failure modes identified:**
- **Harmony fails seed-*independently*** — throws ~0.8 cosines at unrelated tracks
  regardless of seed. Structural.
- **Timbre fails seed-*dependently*** — only fooled by genuinely similar texture (warm
  analog soul vs. warm analog indie). Not brokenness, a narrow blind spot.

---

## 3. Finding B — Structural diagnostic (40k tracks)

[`scripts/research/sonic_tower_diagnostic.py`](../scripts/research/sonic_tower_diagnostic.py) →
`docs/run_audits/sonic_phase2/tower_diagnostic.json`. Measured per tower on
L2-normalised rows over all 39 887 tracks; 400 000 random pairs for cosine stats.

| tower | dims | hub strength | top-1 var | participation (eff. dim) | cos p99 (raw) | cos p99 (centered) | isotropic p99 = 2.33/√d |
|---|---|---|---|---|---|---|---|
| rhythm | 9 | 0.047 | 0.134 | 8.9 | 0.766 | 0.765 | 0.775 |
| timbre | 57 | 0.020 | 0.023 | 56.7 | 0.317 | 0.318 | 0.308 |
| harmony | 20 | 0.056 | 0.068 | **19.6** | 0.575 | 0.572 | 0.520 |
| full_86 | 86 | 0.040 | 0.039 | 66.1 | 0.302 | 0.301 | 0.251 |

**Interpretation — two hypotheses falsified:**
- **No hub / mean offset.** Hub strength ≈ 0; centering moves harmony's p99 by 0.003.
  → "center the tower" would be a no-op.
- **No dominant direction.** Participation ratio 19.6/20 ≈ full rank; top-1 holds 6.8 %.
  → there is no anisotropic spike to null. The genre-embedding anisotropy fix has no
  analog here.
- **The towers are essentially isotropic random spaces of their dimensionality.** Observed
  p99 ≈ the isotropic prediction. Harmony's high cosines are the *arithmetic of 20-dim
  cosine*, not pathology.

This ruled out every transform-based ("EQ the tower") fix and pointed to the *features
themselves*.

---

## 4. Finding C — Richer-harmony probe (286 tracks)

[`scripts/research/sonic_harmony_richer_probe.py`](../scripts/research/sonic_harmony_richer_probe.py) →
`richer_harmony_probe.json`. Richer features extracted on the **harmonic-separated**
signal (`librosa.effects.harmonic`); cached to `richer_harmony_cache.npz`.

| representation | mean ρ | per-seed (duster / green / **jean** / jdilla / real) |
|---|---|---|
| **current** (shipped 20-dim tower) | **−0.144** | −0.09 / −0.09 / **−0.60** / +0.12 / −0.07 |
| cens (CENS mean+std) | +0.015 | −0.08 / +0.08 / −0.23 / +0.20 / +0.10 |
| chroma_cov (12×12 corr.) | +0.089 | −0.20 / +0.21 / +0.32 / +0.12 / +0.00 |
| **twodftm (2D-FFT mag — key-invariant)** | **+0.210** | −0.06 / +0.20 / **+0.53** / +0.16 / +0.21 |
| tonnetz (mean+std) | +0.049 | −0.21 / +0.04 / +0.24 / +0.14 / +0.03 |
| richer_all (concat) | +0.144 | −0.17 / +0.23 / +0.40 / +0.19 / +0.06 |

**2DFTM is the clear winner (+0.354 over baseline).** It flips the decisive harmonically-
rich seed from **−0.60 → +0.53** and improves every seed. Crucially, **adding raw
dimensions does not help** (chroma_cov 66-dim → +0.089; tonnetz 12-dim → +0.049); only the
**key-invariant transform** does. Confirms root cause: the tower encoded absolute key,
which is noise for harmonic-character similarity (transposition → phase, discarded by the
FFT magnitude).

---

## 5. Finding D — Gate 1: does it improve the *blend*?

[`scripts/research/sonic_gate1_blend.py`](../scripts/research/sonic_gate1_blend.py) → `gate1_blend.json`.
Reconstructs the tower-weighted blend and swaps only the harmony block.

| representation | mean ρ |
|---|---|
| harmony_only_current | −0.131 |
| harmony_only_2dftm | +0.210 |
| blend_shipped (`X_sonic` directly) | +0.308 |
| blend_current (reconstructed) | +0.308 ✓ *matches shipped — reconstruction validated* |
| **blend_2dftm** | **+0.361** |

**Gate 1 PASS** (blend delta **+0.053**). But note the asymmetry: the *isolated* swing was
+0.34, the *blend* swing only +0.053 — **timbre at 0.50 was already covering most of
harmony's job.** Gain is concentrated in harmonically-clear seeds (Thibaudet +0.70 → +0.82,
Real Estate +0.30 → +0.39, Green-House +0.17 → +0.28); Duster flat (+0.14 → +0.15);
**J Dilla regressed (+0.23 → +0.17)** — sample-based hip-hop may carry identity in absolute
key, which key-invariance discards. One seed, small effect, flagged for recheck.

---

## 6. Finding E — Gate 2: rhythm (NEGATIVE result)

[`scripts/research/sonic_gate2_rhythm.py`](../scripts/research/sonic_gate2_rhythm.py) → `gate2_rhythm.json`.
Richer rhythm = `librosa.beat.plp` pulse clarity + `librosa.feature.tempogram_ratio`
(tempo-invariant pattern); cached to `richer_rhythm_cache.npz`.

**Isolated:**

| representation | mean ρ | per-seed |
|---|---|---|
| rhythm_current (9-dim tower) | **+0.277** | already healthy |
| pulse_only | +0.096 | **duster +0.32, green +0.32**, others negative |
| tempo_ratio_only | +0.266 | — |
| rhythm_rich | +0.271 | ≈ current — no gain |

**Blend:**

| blend | mean ρ | Δ vs current |
|---|---|---|
| blend_current | +0.308 | — |
| blend_2dftm_harmony | +0.361 | +0.053 |
| blend_rich_rhythm | +0.304 | −0.004 |
| blend_both | +0.332 | +0.024 |

**Conclusions:** (a) the current rhythm tower is already good — *not* the harmony situation;
(b) richer rhythm doesn't help the blend and **degrades the harmony fix when combined**
(+0.361 → +0.332); (c) pulse clarity is a *narrow* instrument — it helps only the slow
seeds (Duster/Green-House) and hurts the rest. **Do not bundle rhythm. Rhythm ships
unchanged.**

---

## 7. Finding F — Harmony-weight sweep

[`scripts/research/sonic_harmony_weight_sweep.py`](../scripts/research/sonic_harmony_weight_sweep.py) →
`harmony_weight_sweep.json`. Rhythm 0.20 / timbre 0.50 fixed; harmony weight swept.

| w_harmony | current | 2dftm |
|---|---|---|
| 0.20 | **+0.328** | +0.353 |
| 0.30 *(production)* | +0.308 | +0.361 |
| 0.40 | +0.294 | +0.361 |
| 0.50 | +0.263 | +0.355 |
| 0.60 | +0.228 | +0.357 |
| 0.75 | +0.190 | +0.339 |
| 0.90 | +0.158 | +0.343 |

- **2DFTM blend is flat** across weights (~0.36) — raising harmony weight buys nothing.
- **Current-harmony blend *falls* as weight rises** — the signature of noise. The shipped
  **0.30 weight is too high**; 0.20 lifts the blend +0.020 with **no re-extraction** (a
  Phase-1-style re-concat of existing per-tower matrices + matching `transition_weights`).
  The optimum may be below 0.20 (the sweep floor) — extend downward before fixing a value.

---

## 8. Research grounding (librosa)

The richer-feature designs are not guesses — they follow established MIR practice for our
toolchain (librosa 0.11):

- **Harmony / matching:** CENS (`chroma_cens`) for dynamics/timbre invariance; **chroma
  cleaning** before extraction (`effects.harmonic` HPSS, `decompose.nn_filter`, median
  filtering); **key invariance** via HPCP+OTI (pairwise) or **2DFTM** (fixed-length,
  per-track) — Bertin-Mahieux & Ellis, ISMIR 2012. Our shipped tower used *none* of these
  (raw `chroma_cqt` median = dynamics-sensitive **and** absolute-key).
- **Timbre:** our tower already matches best practice (MFCC + delta + spectral shape + ZCR);
  it is the best-performing space. Only optional addition: `spectral_flatness` for the
  noisy-vs-tonal (abrasive) axis. Not urgent.
- **Rhythm:** `beat.plp` (pulse clarity) and `tempogram_ratio` (tempo-invariant pattern)
  exist and were tested — but Gate 2 shows the current tower already suffices.

---

## 9. What we learned (incl. corrected hypotheses)

The investigation's value is as much methodological as factual. **Four plausible
hypotheses were falsified by measurement**, each of which we could have "shipped" had we
moved faster:

1. ✗ *Harmony has an anisotropy/hub* (like the genre embedding) → structural diagnostic:
   no hub, isotropic.
2. ✗ *Harmony is too low-dimensional; needs more features* → probe: adding dims (chroma_cov,
   tonnetz) barely helped; only the **key-invariant transform** did.
3. ✗ *Rhythm is the same absolute-tempo bug* → Gate 2: current rhythm is already healthy.
4. ✗ *A better harmony tower wants a higher blend weight* → sweep: 2DFTM is weight-flat.

**The durable lens:** the right question for any tower is not "is it noisy / low-dim /
hubby" but **"does it encode the perceptual quantity, or an absolute correlate of it?"**
Timbre works because MFCC ≈ texture. Harmony failed because chroma-median ≈ absolute key.

**The masking lesson, quantified:** harmony was **actively anti-correlated** (−0.144) yet
the blend audited fine, because timbre at 0.50 overpowered it. A strong component hides a
harmful one — which is why nobody noticed for years, *and* why the fix's blend-level payoff
(+0.053) is far smaller than its isolated payoff (+0.34).

---

## 10. Validation supplements (post-decision)

Two further checks after the gates, before committing to any rebuild:

- **Key-invariance proven empirically** ([`sonic_keyinvariance_check.py`](../scripts/research/sonic_keyinvariance_check.py)):
  pitch-shifting a track leaves 2DFTM at cosine **0.99+** across 1–7 semitones, while
  chroma_median and CENS fall to **0.6–0.8**. The +0.210 is a deterministic property, not a
  5-seed fluke. Also explains CENS underperforming: it stays key-sensitive (cover-song
  matching in the *same* key), so it's the wrong invariance for cross-key similarity.
- **Beat-sync vs frame-level 2DFTM is a wash** ([`sonic_beatsync_2dftm.py`](../scripts/research/sonic_beatsync_2dftm.py)):
  isolated +0.230 vs +0.210, blend +0.348 vs +0.361 (−0.013, within noise). The
  Bertin-Mahieux/Ellis beat-synchronous method buys nothing measurable here. **Use the
  simpler frame-level** — cheaper, and uniform across the beatless ambient/drone corner where
  beat tracking would constantly fall back.

## 11. Decision & path forward

**The proxy metric (Spearman vs verdict) has hit its resolution limit at 5 seeds** — the
2DFTM rebuild's marginal over the free re-weight is ~+0.033, ambiguous. Rather than gather
more proxy seeds, we switched to **stronger evidence: a blind legacy-vs-2DFTM ear A/B over
the real library**, which also produces the rebuild's main artifact as a byproduct.

**In progress:**
1. **Full-library frame-level 2DFTM extraction** ([`extract_harmony_2dftm_sidecar.py`](../scripts/extract_harmony_2dftm_sidecar.py))
   → `data/artifacts/beat3tower_32k/harmony_2dftm_sidecar.npz` (gitignored). Read-only,
   multiprocessing, resumable. **Cost correction: this is an overnight run (~9–18 h depending
   on cores), NOT the "~3 h" estimated earlier — the HPSS step dominates and I undercounted.**
2. **Blind head-to-head wired into the audition** (`sonic_audition_build.py --head-to-head`):
   `harmony_legacy` vs `harmony_2dftm` as blinded spaces over an identical candidate pool;
   the existing per-space analysis yields the A/B directly. Output dir `sonic_audition_h2h`.

**Then decide once, with ear evidence:**
- If the blind A/B confirms 2DFTM neighbors beat legacy → fold the sidecar into the artifact
  (cheap re-concat — the extraction is already done) + matching schema/version bump, rebuild
  `tower_weighted`, re-audition to confirm in production.
- If it's a wash by ear → ship the **cheap re-weight** (harmony 0.30 → ~0.20, matching
  `transition_weights`; extend the sweep below 0.20 first) and skip the rebuild.

Rhythm is excluded from any rebuild (Gate 2). Beat-sync is excluded (§10 wash).

**Banked regardless of the decision:** root cause confirmed (absolute-key encoding);
validated fix in hand (2DFTM); rhythm confirmed healthy; harmony weight shown
miscalibrated; a reusable probe methodology that worked across three investigations.

---

## 11. Reproducibility

**Scripts (all on branch `sonic-harmony-keyinvariant`, read-only, cached):**

| script | commit | output |
|---|---|---|
| `sonic_tower_diagnostic.py` | 73198e6 | `tower_diagnostic.json` |
| `sonic_harmony_richer_probe.py` | 73198e6 | `richer_harmony_probe.json`, `richer_harmony_cache.npz` |
| `sonic_gate1_blend.py` | 0de7be3 | `gate1_blend.json` |
| `sonic_gate2_rhythm.py` | e806cf3 | `gate2_rhythm.json`, `richer_rhythm_cache.npz` |
| `sonic_harmony_weight_sweep.py` | f9382b6 | `harmony_weight_sweep.json` |

Outputs + caches live under `docs/run_audits/sonic_phase2/` (gitignored — regenerate by
re-running). Audition verdicts under `docs/run_audits/sonic_audition/*_capture.yaml`.
Artifact: `data/artifacts/beat3tower_32k/data_matrices_step1.npz` (variant `tower_weighted`).

**Data safety:** every step is read-only. No writes to `data/metadata.db`; audio files
read for extraction only, never modified; all derived features written to sidecar npz/JSON.

---

## 12. Head-to-head audition + production rebuild (2026-06-03)

The deferred decision (§10) was resolved by completing the full-library extraction and
running a **blind legacy-vs-2DFTM A/B** rather than committing on the 5-seed probe alone.

### 12.1 Extraction

`extract_harmony_2dftm_sidecar.py` extracted frame-level 2DFTM harmony for the whole
library: **39,879 tracks, 0 failures**, ~17h on 18 workers, written to
`data/artifacts/beat3tower_32k/harmony_2dftm_sidecar.npz` (39,887 total incl. smoke-test
rows). Read-only throughout; atomic checkpointed saves.

### 12.2 Blind A/B audition

`sonic_audition_build.py --head-to-head` builds two blinded harmony spaces
(`harmony_legacy` = shipped chroma-median; `harmony_2dftm` = key-invariant), each
ranked over an identical candidate pool. Provenance hidden from the rater. Three seeds
rated (avg verdict score: match=3, close=2, off=1, wrong=0):

| Seed | 2DFTM avg | Legacy avg | Δ | note |
|---|---|---|---|---|
| Jean-Yves Thibaudet | **2.53** | 0.87 | **+1.66** | classical piano — key-invariance kills cross-key false positives (bossa nova, chillwave, noise rock) |
| Green-House | 1.67 | 1.53 | +0.13 | ambient new-age — marginal 2DFTM win |
| Minor Threat | 1.29 | **2.13** | **−0.84** | **counter:** power chords (root+fifth, no third) have ~no harmonic texture; absolute key acts as a genre proxy for legacy |
| **Overall (44/45 rated)** | **1.84** | **1.51** | **+0.33** | |

**Read:** the win is real but concentrated in harmonically-distinct material
(classical/jazz). Minor Threat is a genuine, deliberately-sought counter — for
texture-poor hardcore, legacy's key sensitivity is accidentally useful. Net +0.33
favors 2DFTM; harmony is only 30% of the sonic blend, so the punk regression is
attenuated in production. Within-space cosine↔verdict correlation is weak for both
(r≈0.07 2DFTM): 2DFTM surfaces a *better pool* but does not *rank within it* much
better — fine for retrieval, less so for fine-grained edge ordering.

**Decision: full rebuild** (Jean-Yves +1.66 outweighs Minor Threat −0.84 ~2:1).

### 12.3 Rebuild record

`fold_2dftm_into_artifact.py` replaced the 20-dim chroma-median harmony tower with the
96-dim 2DFTM tower, z-scored over the 39,887-track valid pool then folded via
`sqrt(w)·L2(tower)` (weights unchanged at 0.20/0.50/0.30). Blend grew **86 → 162 dim**.
Backup: `data_matrices_step1.npz.bak_20260603_204655`. The shipped representation is
**identical** to the audited `harmony_2dftm` space (same z-score → L2 pipeline) for every
track with features; the 70 audio-less tracks get zero harmony vectors.

**Validation:** 1425 tests pass. End-to-end CLI generation healthy — Bill Evans seed
clustered pure classic jazz (Miles Davis, Brubeck, Max Roach, Dexter Gordon, Ahmad Jamal,
Ellington/Coltrane), `min_transition=0.546`, zero below-floor edges.

### 12.4 Known caveats (deliberate / latent)

1. **Segment harmony is now global.** Start/mid/end harmony all use the full-track 2DFTM
   (no per-segment 2DFTM was extracted). The beam's transition harmony component therefore
   measures *overall harmonic compatibility* rather than *end-of-A → start-of-B* flow.
   Defensible (arguably better for character matching) but **not separately validated** —
   the audition tested full-track retrieval only. Rhythm/timbre segments remain
   position-specific.
2. **`tower_pca_dims` mis-slicing — FIXED.** The artifact stores the authoritative
   `tower_dims=[9,57,96]`, but `ArtifactBundle` did not expose it, so the GUI
   track-replacement path inferred the split from total dim
   (`_infer_tower_pca_dims(162) → (40,81,41)`, wrong) when slicing X_sonic into rhythm/
   color axes. Fixed 2026-06-03: `ArtifactBundle` now loads `tower_dims`, and
   `worker._resolve_tower_pca_dims` prefers it (validated against blend width) over config
   override or width-inference. The CLI pace gate was unaffected — it builds a self-contained
   `tower_pca` variant and degrades gracefully on mismatch. Covered by
   `test_worker_tower_pca_dims.py` and `test_artifact_tower_weighted_load.py`.

### 12.5 Scripts added

| script | role |
|---|---|
| `extract_harmony_2dftm_sidecar.py` | full-library 2DFTM extraction (resumable, atomic) |
| `fold_2dftm_into_artifact.py` | surgical harmony-tower replacement (backup + atomic write) |
| `sonic_audition_build.py --head-to-head` | blinded legacy-vs-2DFTM A/B manifests |

Audition verdicts: `docs/run_audits/sonic_audition_h2h/*_capture.yaml` + `findings.md`.
