# Pace Audition Harness — Design

**Date:** 2026-06-13
**Status:** Approved design, pending implementation plan
**Related:** `project_pace_gate_retune` memory + final commit `4dcefa0` (the shipped narrow caps under test); `scripts/sonic_audition_*.py` (the structural model — transition playback + range streaming + YAML capture); `.claude/skills/evaluation-methodology/SKILL.md` (the discipline this harness must satisfy); `tests/support/gui_fidelity.py` (`generate_like_gui` / `resolve_gui_overrides`, the only fidelity-correct generation path).

## Purpose

Validate, by blind human judgment, that the shipped `pace_mode: narrow` caps are doing perceptible work without over-tightening — the one open item the pace-gate retune left behind. The retune's own quality numbers (T / S / onset distances) are computed in the very spaces the gate acts on, so they cannot answer "is this audibly better"; only an independent perceptual arm can (evaluation-methodology skill, anti-circularity rule).

Three questions, answered in one blind rating pass over transitions:

1. **Does narrow audibly help vs the default?** Are narrow's bridge edges rated more pace-continuous than dynamic's?
2. **Does pace gating do anything at all?** Are narrow/dynamic edges rated above `off` (no pace gating), or is the whole feature inaudible?
3. **Is the win pace-specific (not incidental)?** Does narrow's *continuity* advantage exceed its *smoothness* advantage — i.e., is it the pace gate, not a coincidental genre/sonic difference?

A `decoy` arm of deliberately pace-discontinuous edges is the negative control: if the rater can't score the decoy worst on continuity, the rating pass has no discriminative power and no other verdict is trustworthy.

## Why transitions, not whole playlists

Pace coherence is fundamentally an **edge** property (north star #5 — the worst transition defines the experience), and a transition unit makes the audition cheap (~50 min of focused rating vs hours), high-N (≈30 edges/arm → real distributions vs 6 playlists/arm), and lets us reuse the sonic harness's transition-playback machinery almost directly. The cost is losing the whole-arc "is narrow monotonous" judgment; that returns as a cheap **structural** check in analysis (onset-rate variance per playlist), explicitly labeled non-perceptual.

## Why a blind pass (not side-by-side)

If the rater can see which arm produced an edge, the arm we're rooting for wins on expectation bias. Each edge gets a blind ID; the `blind_id → (arm, seed, track_a, track_b)` mapping is held server-side only and re-attached at capture time. Edges are presented in randomized, arm-interleaved order with arm labels never shown.

## Arms

| Arm | Pace config | Question it answers |
|---|---|---|
| `narrow` | onset/bpm adm/brd 0.50/0.60, soft 0.25/0.15 (shipped) | The change under test |
| `dynamic` | 0.75/0.85, soft 0.15/0.10 (default) | Does tightening 0.75→0.50 audibly help? |
| `off` | pace gating disabled (genre/sonic gating still on) | Does pace gating do anything vs nothing? |
| `decoy` | synthesized pace-distant edges (negative control) | Can the rater perceive pace at all? |

Expected ordering: `decoy < off ≤ dynamic ≤ narrow` on continuity. Where the real perceptual gaps fall is the finding.

## Architecture

Three file-communicating scripts plus a static page and unit tests, mirroring the sonic harness (it already streams local audio with HTTP range support and captures verdicts to YAML).

| File | Role |
|---|---|
| `scripts/pace_audition_build.py` | Generate the 3 real arms per seed via `generate_like_gui`, sample interior bridge edges, synthesize decoy edges, blind/shuffle, write manifest + index + provenance |
| `scripts/pace_audition_page.html` | Edge-player UI: play A-tail→hard-cut→B-head, two 1-5 sliders (continuity, smoothness), note field, auto-save, progress |
| `scripts/pace_audition_serve.py` | HTTP server: range-streamed audio windows + blinded manifest API + progress API + YAML capture (forks `sonic_audition_serve.py`) |
| `scripts/pace_audition_analyze.py` | Un-blind, join scores to arms, compute distributions + the three reads + structural monotony check, write `findings.md` |
| `tests/unit/test_pace_audition_build.py` | Edge sampling, decoy synthesis, blinding/shuffle determinism, provenance capture |
| `tests/unit/test_pace_audition_serve.py` | Range parsing, manifest blinding (no arm leak), capture upsert |
| `tests/unit/test_pace_audition_analyze.py` | Distribution aggregation, contrast computation, confound + discrimination logic |

**Output dir:** `docs/run_audits/pace_audition/` — gitignored, created at runtime. `manifest.json`, `index.json`, `capture.yaml`, `findings.md`.

**Tech stack:** Python 3.11, NumPy, PyYAML, sqlite3 (stdlib), http.server (stdlib), vanilla JS/HTML. No new pip installs.

## Data sources

| Source | Path | Role |
|---|---|---|
| Production artifact | `data/artifacts/beat3tower_32k/data_matrices_step1.npz` | Generation + sonic/onset features (all real arms generate from THIS, only pace config differs) |
| Track DB (read-only) | `data/metadata.db` | BPM + `onset_rate` (`sonic_features.full.rhythm.onset_rate`), genre, file paths for playback |
| Fidelity harness | `tests/support/gui_fidelity.py` | `generate_like_gui` / `resolve_gui_overrides` — the only production-faithful generation path |

## Seeds

Six multi-pier seed sets, both pace regimes:

- **Ambient:** Green-House, Hiroshi Yoshimura, Brian Eno
- **Rhythmic:** J Dilla, De La Soul, Beastie Boys

Each generated with ≈4-5 of the artist's own tracks as piers (artist-style multi-pier, matching the calibration and the `test_pace_narrow_feasible_for_ambient_piers` regression), length 30. Seeds present in fewer than 4 tracks in the artifact are skipped and recorded in the index.

## Build flow (`pace_audition_build.py`)

For each seed set:

1. **Resolve piers.** Look up the artist's track_ids in the artifact; require ≥4; else `SKIP` and record.
2. **Generate the 3 real arms.** Call `generate_like_gui(seeds=piers, pace_mode=<arm>, length=30, random_seed=0, …)` with `cohesion/genre/sonic` modes held fixed across arms (only `pace_mode` varies). Record the resolved overrides per arm via `resolve_gui_overrides` for provenance.
3. **Sample real edges.** From each arm's playlist, take **interior bridge edges only** (exclude pier-adjacent edges — the pace gate governs bridge selection, not pier placement). Seeded-random sample **5 edges/arm/seed**. Each edge records `(track_a, track_b)`, onset/BPM log-distance, genre similarity.
4. **Synthesize decoy edges.** From the seed's own candidate pool, pick pairs with onset/BPM log-distance **> 1.0** (well above the dynamic cap) AND genre similarity **≥ the pool median** — pace-discontinuous while holding genre roughly constant, so a low decoy *continuity* score isolates pace. ~5 decoy edges/seed.
5. **Blind + shuffle.** Assign blind IDs, write `blind_id → (arm, seed, track_a, track_b, file_a, file_b)` to the server-side manifest; write the shuffled, arm-stripped edge list for the page. Interleave arms so no run of one arm is visible.
6. **Provenance block.** Artifact path + mtime + shape fingerprint; per-arm pace config; seed→pier mapping; random seed. (Eval-methodology data-provenance rule — confirm all real arms share one artifact.)

Total ≈ 6 seeds × (3 arms × 5 + 5 decoy) = **120 edges**.

## Serve flow (`pace_audition_serve.py`)

- Forks `sonic_audition_serve.py`: HTTP range streaming, blinded-manifest API, progress API, YAML capture (upsert keyed by `blind_id`).
- Playback window: serve a byte range covering the **last ~12s of track A** and the **first ~12s of track B**; the page plays A-tail then **hard-cuts** to B-head (no crossfade — faithful to m3u8/Plex playback). Exact windowing strategy (server-side trim vs client seek over a range-streamed file) is an implementation detail for the plan; the requirement is "tail-of-A → hard cut → head-of-B, no crossfade."
- The page never receives arm/seed labels — only blind IDs and audio.

## Analyze flow (`pace_audition_analyze.py`)

Un-blind via the server-side manifest, join to captured scores, then:

1. **Distributions, not means.** Per arm, report min / p10 / p50 / p90 for both `continuity` and `smoothness`. Sliced **ambient vs rhythmic**.
2. **Discrimination check.** Is `decoy` continuity p50 clearly below all real arms? If not, flag the pass as non-discriminative (verdict untrustworthy).
3. **Core contrasts.** `narrow` vs `dynamic` (does tightening help?) and `{narrow,dynamic}` vs `off` (does gating do anything?), on continuity.
4. **Confound check.** Compare narrow's continuity gain vs its smoothness gain over dynamic. Pace-specific win ⇒ continuity gain > smoothness gain; comparable gains ⇒ the difference may be incidental, not the pace gate.
5. **Structural monotony check (non-perceptual).** Onset-rate variance/range across each full playlist per arm. If narrow collapses onset variance toward zero relative to dynamic, flag as a monotony risk — labeled structural signal, not a perceptual verdict.
6. Write `findings.md` with the verdict, N per arm/regime stated explicitly, and the assumptions-that-could-invalidate list (stale artifact, single-listener, library-specific).

## Evaluation discipline (must hold)

- All real arms generate from the **same artifact**; build records mtime + fingerprint and analyze asserts they match (the A-vs-A incident this skill exists to prevent).
- **Full uncapped pool**, as production — no convenience subset.
- **Decoy present** as negative control; **blinded**; **distributions not means**; **N stated** in every conclusion.
- Output **only** under `docs/run_audits/pace_audition/`; **never** writes to `data/artifacts/`, `data/metadata.db`, or audio files (all read-only).

## Scope & limitations (honest)

- Single listener, one library: directional evidence that narrow is *perceptibly doing its job without over-tightening* — not a cap-value optimizer and not generalizable beyond this library.
- ≈30 edges/arm (≈15 per regime) is a genuine distribution and far better than 6 playlists, but small; conclusions state N and stay proportionate.
- Transition unit cannot judge whole-arc monotony perceptually; the structural onset-variance check is a proxy, not a substitute.
- Decoy genre-constancy is approximate (best-effort pace isolation), so the confound check is the primary defense against misattribution, not the decoy alone.

## Out of scope

- Re-tuning cap values (this audition tells us *whether* narrow works, not the optimal number).
- `strict` mode (not the shipped change under question).
- Automated/LLM judging (the independent arm here is specifically human ears).
- Whole-playlist perceptual auditioning (deliberately deferred; structural proxy only).
