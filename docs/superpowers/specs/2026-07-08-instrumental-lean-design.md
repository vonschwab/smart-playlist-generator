# Instrumental lean — design spec

**Date:** 2026-07-08
**Status:** design approved, pending spec review → writing-plans
**Feature:** an "Instrumental" checkbox that strongly demotes vocal-classified tracks from
bridge candidates, as a guard against spoken-word / poetry-over-ambient tracks in ambient
playlists — without ever hard-failing generation.

## Summary

Add a per-track `voice_prob ∈ [0,1]` signal (Essentia `voice_instrumental-musicnn-msd-2`,
reusing the `msd-musicnn` embedding the energy pass already computes) and wire a **continuous,
never-failing soft penalty** into the beam + candidate pool. A GUI checkbox toggles it. When on,
each bridge candidate is demoted proportionally to `voice_prob`; user-chosen seeds/piers are
exempt; a confession line on the result reports any vocal track admitted as a last resort.

This is deliberately **not** a hard filter. It is a fourth soft preference axis in spirit,
behaving like sonic/genre/pace (fall back gracefully), never like diversity (the one hard
constraint).

## Design-principle alignment (why soft lean, decided upfront)

- **Continuous gradients beat hard cliffs (Layer 4 #19):** the classifier hands us a continuous
  `P(voice)`; a penalty proportional to it *is* the gradient. Thresholding into a binary
  drop throws away the graduated signal.
- **Never fail on playlist generation (never-fail-three-axes):** a score term never starves the
  pool. A thin instrumental library yields a best-effort playlist, never an error.
- **Graceful failure on a known-imperfect signal:** the classifier will mis-flag some instrumental
  ambient (wordless vocal pads, breath/choir textures) as vocal. Under soft lean that track is
  demoted, not deleted — recoverable when the pool is thin. A hard gate would silently lose it.
- **Activate fixes; never a silent no-op:** if the checkbox is on but `voice_prob` data is missing,
  warn loudly (see Config) rather than silently doing nothing.

The known classifier weakness — heavily-processed vocals (vocoder/talkbox, e.g. Black Moth Super
Rainbow) reading as *instrumental* — is **low-harm for this use case**: clean spoken-word poetry
(the target to exclude) is exactly what the model classifies well, and processed-vocal tracks that
slip through are usually ambient-texture-friendly anyway. The higher-harm direction is false
negatives, which soft lean handles gracefully.

## Non-goals (YAGNI)

- **Not** a hard filter / exclusion, and **not** a length-shortening guard.
- **Not** a new "ambient mode" or preset coupling — the checkbox is a standalone global toggle.
- **Not** a MuQ-based classifier (would require hand-labeling + training a custom head; a much
  bigger project — see Alternatives).
- **Not** a sidecar-consolidation refactor (see Future work / housekeeping).

## Component 1 — Extraction (isolated instrumental sidecar)

**Model:** Essentia `voice_instrumental-musicnn-msd-2.pb`, a 2-class head that runs on the *same*
`msd-musicnn` embedding already computed in `scripts/extract_energy_sidecar.py::_process`
(`emb = _emb(audio)`, line ~132). Adding it is one extra `TensorflowPredict2D` head; the expensive
audio-decode + embed is shared.

**Isolation decision:** write to a **separate** `instrumental_sidecar.npz` under
`<artifact>/instrumental/`, parallel to `<artifact>/energy/` — do **not** re-run `--force` on the
energy sidecar. Re-running the energy extractor risks perturbing the live pace axis (Essentia
version drift, a track that newly errors). A dedicated sidecar shares the extraction scaffold
(path resolution, JSONL checkpoint, merge, WSL invocation) but touches nothing in the working
energy/pace path — zero regression risk for identical compute. Prefer factoring the shared scaffold
into a helper over copy-paste (reuse-first).

**Scope & cost:** the artifact's 42,143 track_ids (the exact generation set). One full audio-decode
+ embed pass — unavoidable because the energy pass discarded its embeddings (only derived scalars
were persisted). Estimated **~2–5 h**, 14 workers, resumable (append-only checkpoint), read-only on
audio + `metadata.db`, writes only the sidecar. This is the same pass already run once for energy —
**not** the multi-day `metadata.db` re-analysis.

**Validate-first gate (evaluation-methodology):** before the full run, `--limit` smoke on a curated
label set — a few pure-instrumental ambient tracks, a few spoken-word / poetry-over-ambient, and the
Black Moth vocoder case — and confirm `voice_prob` separates them. The smoke run also yields the
real `trk/s` → exact ETA before committing the hours. If it does not separate cleanly, stop and
rethink before the full pass.

**Output:** per-track `voice_prob` (float32; NaN for missing/errored tracks). Store the raw
per-frame aggregation choice explicitly (mean of the per-frame instrumental/voice softmax).

**Prerequisite:** the `voice_instrumental-musicnn-msd-2.pb` weights + companion `.json` must be
fetched from the Essentia model zoo into `/opt/ess/models` (the energy preflight in
`src/analyze/energy_runner.py::preflight_wsl` currently checks only msd-musicnn / emomusic /
danceability — extend it to also require the voice_instrumental `.pb`, so a missing model fails the
preflight loudly instead of mid-run).

**To verify at implementation time:** the model's class-column order (`[instrumental, voice]` vs
reverse) — read from the model's companion `.json`, never guess. `voice_prob` must be the *voice*
column.

## Component 2 — Read path

Extend `src/playlist/energy_loader.py` (or a sibling loader following the same pattern) to load
`instrumental_sidecar.npz` and expose per-track `voice_prob`, aligned to artifact track_ids. No new
sidecar *format*, no metadata.db write. Missing sidecar → `voice_prob` all-NaN, feature inert +
startup warning (Config).

## Component 3 — Mechanism (reuse the beam soft-penalty seam)

Not a new subsystem. Rides the additive soft-penalty machinery the beam already uses for
`soft_genre_penalty` (`src/playlist/pier_bridge/beam.py`, `pier_bridge/config.py`) and the
onset-band pace penalty.

- **Beam term:** each candidate's edge score gets an additive penalty `= penalty_weight × voice_prob`.
- **Pool demotion:** apply a matching demotion at candidate-pool admission scoring
  (`src/playlist/candidate_pool.py`) so a demoted-but-present vocal track cannot re-enter via a
  strong transition edge (the tag-steering "beam was blind → too weak" lesson).
- **Continuous:** penalty scales with `voice_prob`. A 0.95-voice poetry track is hammered; a
  0.55-ambiguous track is nudged. `voice_prob = NaN` → zero penalty (unknown, never punished).
- **Seeds/piers exempt:** never demote a user-seeded/pier track (parallels recency, which excludes
  seeds pre-order). Only bridge candidates are subject to the penalty.
- **Never hard-fails:** it is only a score term; a thin instrumental library yields a best-effort
  playlist, never an error.

## Component 4 — Config (`config.yaml`, principle 23)

```yaml
instrumental:
  penalty_weight: <tuned decisive>   # near-exclusion in a normal pool; tuning recipe in the plan
```

- `enabled` is a per-request runtime flag (from the checkbox), not a config default.
- **Missing-data path warns loudly**, does not silently no-op: if a request sets the flag but the
  instrumental sidecar is absent/empty, log a clear warning (the "configured knob that can't act is
  a startup error, not a silent no-op" gotcha). The generation still proceeds (never-fail), but the
  operator is told the guard was inert.

## Component 5 — Policy routing (critical)

The flag MUST be threaded through the shared policy layer (`src/playlist_gui/policy.py`,
`derive_runtime_config`) — the same seam the mode axes use. Modes/knobs that bypass policy go inert
in the real GUI path (known trap: slider-calibration false negatives). Add the flag to:
- `web/src/lib/types.ts` `GenerateRequestBody` and `src/playlist_web/schemas.py`.
- `src/playlist/request_models.py::GeneratePlaylistRequest` (new field, both `from_ui_state` and the
  args constructor, and `to_args`).
- The policy derivation so it reaches the worker + beam/pool config.

## Component 6 — GUI

Checkbox in **Row 3** of `web/src/components/GenerateControls.tsx` (the pool-composition/filter row,
next to `freshness` / `skip recent seeds`), using the existing
`<label><input type="checkbox" className="accent-[#5eead4]"><Lbl>…</Lbl></label>` pattern. Persist
via `useLocalStorage("pg_instrumental", false)`. Send as a new boolean on `GenerateRequestBody`.
Label: **instrumental**; tooltip explains it demotes vocal tracks and notes the processed-vocal
caveat. Do **not** touch `AdvancedPanel.tsx` (unrelated bottom tab panel).

## Component 7 — Confession / warning ("with warnings")

When the checkbox is on, the existing honor+confess receipt (SDD) reports:
- how many vocal-classified tracks were still admitted as a last resort (pool too thin), and
- a one-time note that heavily-processed vocals (vocoder/talkback) may read as instrumental and slip
  through, and conversely some instrumental textures may be over-demoted.

Surfaced on the result, not a modal — the warning lands where the user sees the outcome.

## Testing

- **Generation fidelity (mandatory, playlist-testing):** a `gui_fidelity` **multi-pier** artist-mode
  test asserting a known high-`voice_prob` track is demoted out of the bridge when the flag is on and
  present when off. Never a single-seed / hand-built-override harness.
- **Loader test:** `voice_prob` reads back aligned to track_ids; missing sidecar → all-NaN, no crash.
- **Extraction smoke:** `--limit` run produces a well-formed sidecar; class-column order asserted
  against the model `.json`.
- **Policy routing test:** the flag survives `derive_runtime_config` into the runtime config (guards
  the inert-knob trap).

## Open items to verify during implementation

1. `voice_instrumental` model class-column order (from model `.json`).
2. Exact additive-penalty insertion point(s) in `beam.py` / `candidate_pool.py` and whether the
   existing soft-penalty helper is reusable as-is or needs a small generalization.
3. `penalty_weight` default calibration (tuning recipe → plan): steep enough to be decisive without
   being a de-facto hard gate.
4. Where `voice_prob` aggregation across frames is defined (mean vs percentile) — pick mean unless
   the smoke set argues otherwise.

## Alternatives considered

- **Hard filter (+warn / +relax):** rejected — violates never-fail (a 6-track playlist) or becomes a
  silent hard gate that readmits the exact tracks it should guard against.
- **Classify off persisted MuQ vectors (avoid the rescan):** rejected — no pretrained voice/
  instrumental head for MuQ; needs hand-labeled data + a trained head. Much larger project, worse
  first step.
- **Extend + `--force` the energy sidecar:** rejected in favor of an isolated sidecar to avoid any
  regression risk to the live pace axis, for identical compute.

## Rollout sequence (activate fixes; never merge inactive)

1. Add the `voice_instrumental` head + isolated sidecar extraction; `--limit` **validate-first**.
2. If validated, full library pass (canonical-only, background, resumable).
3. Wire loader → policy → beam/pool penalty → request models → GUI checkbox → confession.
4. Tests green; exercise end-to-end in the real GUI; **activate as the live path** behind the
   checkbox (default off is correct here — it is a user choice, not a fix being withheld).

## Future work / housekeeping (out of scope)

Sidecar backup files accumulate: `<artifact>/muq_sidecar.bak_2026*.npz` (~10 files) + a stray
`.tmp.npz`, and the energy/instrumental extractors mint a timestamped `.bak` on every merge. A
separate small housekeeping task should define a retention policy (keep last N backups, prune the
rest) and sweep the stray `.tmp`. Not part of this feature.
