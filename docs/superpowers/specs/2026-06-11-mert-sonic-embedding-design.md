# Learned Sonic Embedding (MERT) — Design

## Purpose

Replace the hand-built sonic similarity space (`tower_weighted`: rhythm 9 + timbre 57 + harmony 96) with a learned music embedding (MERT-v1-95M) as an opt-in `sonic_variant`. The towers are the validated root cause of mediocre playlist quality: all three are perceptually unreliable, so the beam cannot find transitions that gel no matter how the engine is tuned.

## Evidence (2026-06-11 diagnosis)

Full write-up in memory `project_timbre_embedding_ceiling`; diagnostic scripts `C:\tmp\diag_{seeds,edge,timbre,towers,mert}.py`.

| finding | data |
|---|---|
| Rhythm tower (9-dim) saturated | 0.9+ cosine to everything; Glen Campbell @0.94 / Tom Waits @0.96 nearest to Yeah Yeah Yeahs |
| Timbre tower (57-dim, weight 0.50) coarse | global max ~0.5–0.6; Metallica/Pixies nearest to YYY, Squarepusher/Autechre nearest to St Vincent |
| Harmony tower (96-dim) cross-genre | Jay-Z/Aaliyah nearest to YYY; Busta Rhymes/De La Soul nearest to Courtney Barnett |
| Reporter `T` hides clunkers | Hives→The Drums raw blend +0.06 (timbre −0.07) displayed as T=0.520 — centering/gamma compresses near-zero edges into the "fine" band |
| **MERT prototype validates the fix** | 8 seeds + 160 random tracks: MERT top-6 neighbor genre-coherence **0.183 vs towers 0.100** (+83%, 6/8 seeds); removes all Metallica-class errors; Arctic Monkeys → split system / Ramones / Minor Threat / Pavement |
| MERT anisotropy is fixable | raw cosines saturate at +0.72…+0.90; mean-centering spreads them to −0.28…+0.29 with rankings essentially intact (coherence 0.183→0.167). (Whitening-on-168-samples result is rank-starved noise — the real fit happens on 40k.) |

Consequence accepted in this design: the "sonic ⊗ genre fusion" has been running with genre carrying the entire product. Fixing sonic is a **features** problem — not tuning, not architecture.

## Model

- **MERT-v1-95M** (`m-a-p/MERT-v1-95M`), pinned to a specific HF revision at implementation time (`trust_remote_code` — pin is mandatory, the remote code files are executed).
- Runtime: torch CPU (2.12.0+cpu) + transformers (5.11.0), already installed. ~1.7 s per 24 s clip on this machine.
- Embedding: mean over all 13 hidden-state layers, mean over time → 768-d float32. (Layer-weighted variants are a calibration option, not a commitment.)
- Input: 24 kHz mono, decoded via librosa/soundfile from the original FLAC/MP3 — **audio files are read-only, no exceptions**.

Why MERT over CLAP/OpenL3: music-specific pretraining, already validated locally on this library, no text-tower baggage, acceptable CPU speed.

## Clip strategy

The pipeline scores transitions as end-of-A vs start-of-B, so the variant must supply start/mid/end representations, mirroring the existing `X_sonic_{start,mid,end}` contract:

- **start** = 24 s from offset 0
- **mid** = 24 s centered at track midpoint
- **end** = final 24 s
- **whole** = mean of the three clip embeddings, re-normalized

3 clips × ~1.7 s ≈ 5.1 s/track CPU → ~57 h for 39,957 tracks (resumable overnight runs), or a few hours on any GPU. Tracks shorter than ~30 s: single clip used for all three slots.

## Post-processing (anisotropy correction)

Raw MERT cosines are saturated. Pipeline: **mean-center → (optional robust whiten) → (optional PCA-k) → L2-normalize**, with the transform fitted on the full library and persisted alongside the embeddings (same discipline as the dense genre embedding's anisotropy fix, memory `project_genre_embedding_anisotropy`).

Open parameters resolved empirically in the calibration phase (~2k stratified subset):
- centering only vs. robust whitening (whitening fit on 40k is meaningful; on 168 it was not)
- PCA dimension: 768 (none) / 256 / 128 — PCA both denoises and cuts beam FLOPs (768-d is ~4.7× the current 162-d per edge cosine; generation must stay inside the 90 s budget)

Acceptance metric for calibration: taxonomy-coherence of top-k neighbors (the metric that produced 0.183 vs 0.100) plus a fixed manual spot-check list (e.g., "YYY's neighbors contain no metal/jazz/hip-hop").

## Integration

New artifact keys, following the `tower_weighted` variant precedent:

- `X_sonic_mert`, `X_sonic_mert_start`, `X_sonic_mert_mid`, `X_sonic_mert_end` (post-processed, L2-normalized, float32)
- Selected by existing config: `sonic.variant: mert` (resolution already logs "Using precomputed sonic variant ... from artifact key ...")
- Build path: extraction script → sidecar NPZ → fold script (`fold_2dftm_into_artifact.py` precedent: backup artifact, identity-check `track_ids` exactly, write keys). Sidecar is identity-locked like the dense genre sidecar — a vocab/track change drops it on load **loudly**, never silently.

### What happens to the towers under `mert`

- **transition_weights / tower_weights:** not applicable — single space; beam and reporter both score plain cosine in the same space, so beam/reporter alignment (the v4.1 invariant) holds by construction. Resolution must hard-error if someone configures tower-style transition_weights with `variant: mert` (a configured knob that can't act is a startup error).
- **Pace gate:** `rhythm_matrix` currently comes from `tower_dims`, which `mert` doesn't have. Pace continues to run on the perceptual-BPM machinery (`bpm_array`, pace admission/bridge gates), which is independent of the towers. If `pace_bridge_floor > 0` and no rhythm dims exist, the existing loud-warning path fires; the plan upgrades this to a documented, tested fallback (BPM-only pace gating), not a silent no-op.
- **Harmony 2DFTM:** remains in the artifact untouched; `tower_weighted` stays fully functional as the default and the rollback path.

### Architectural note (Layer-2 commitment #8)

"Sonic feel is multi-dimensional" was written assuming the towers carry independent meaning. The diagnosis shows they don't (all three perceptually broken). Under this design the decomposition becomes: **MERT = sonic texture/similarity space; perceptual BPM = rhythm/pace axis (already separate); genre taxonomy = cultural axis.** That preserves the multi-axis *intent* with axes that actually work. The commitment text should be amended when the variant becomes default, not silently contradicted.

## Validation gates (before default flip)

1. **Neighbor QA** — re-run the diagnostic suite against the folded artifact; coherence ≥ prototype levels; zero Metallica-class errors on the spot list.
2. **8-seed playlist A/B** — `sonic_variant: mert` vs `tower_weighted`, same seeds; structural metrics + the per-tower edge decomposition replaced by per-edge MERT cosine.
3. **Listen test** — the user's ear is the ground truth this whole effort is anchored to; metrics alone do not flip the default.
4. **Blind audition harness** — reuse the sonic-audition pattern (memory `project_sonic_audition_phase2`) for an unbiased pass.
5. **Performance** — generation ≤ 90 s on the 10-seed stress case (hard budget, memory `feedback_generation_time_budget`).

After the flip: revisit `weight_genre=0` / bridge-transition balance (commit 0abfb7e) — that tuning handed edge selection entirely to the sonic signal *because* it was assumed trustworthy; with a working sonic space the balance question reopens with better inputs.

## Out of scope

- Regenerate-button stochasticity (separate, user-approved future feature)
- Genre lane changes (taxonomy coverage holes — `indie`, `lo-fi`, `arena rock` — are a separate cheap-win lane)
- Re-analysis of BPM or other metadata.db features (`metadata.db` is untouched by this entire effort)
- GUI changes
- Per-edge `bpm=n/a` reporter display bug (trivial separate fix)

## Risks

| risk | mitigation |
|---|---|
| HF remote-code drift / supply chain | pin model revision; embeddings cached locally; no runtime HF dependency after extraction |
| 57 h CPU extraction stalls midway | resumable shard design; progress logging; per-track error skip + manifest |
| Anisotropy fix degrades rankings at 40k scale | calibration phase decides on a 2k subset first; centering-only fallback is already validated |
| 768-d blows the 90 s generation budget | PCA-k option in calibration; perf gate in validation |
| Genre-spanning tracks (St Vincent dance-pop) still pull odd neighbors | known limit; genre lane still gates; not a regression vs towers |
