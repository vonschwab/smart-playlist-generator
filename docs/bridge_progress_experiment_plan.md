# Bridge Progress Experiment Plan

## Problem Statement
Pier+Bridge ordering already finds strong candidates, but segment ordering can still feel like a "near-pier pile" rather than a deliberate arc. We want bridge segments that start close to Pier A, move toward a shared sonic middle, then end closer to Pier B. Current scoring uses transition quality, bridge similarity, and a monotonic progress constraint, but does not explicitly reward target progress along the segment.

## Hypothetical Solution
Introduce a dry-run/audit-only experiment that adds a progress-aware term to beam search scoring. This term rewards candidates whose projected progress toward Pier B matches a per-step target curve.

Two shapes:
- `linear`: target progress grows linearly with step index.
- `arc`: target progress follows a cosine ease-in/out curve, staying closer to Pier A at the beginning and closer to Pier B at the end.

The experiment keeps the existing monotonic constraint and transition/bridge scoring intact.

## Rationale
- Progress in sonic space is already computed (A->B projection). We can use it to shape the ordering, not just gate it.
- Explicit target progress is the minimal change that produces an audible arc without changing candidate discovery logic.
- Dry-run gating allows A/B testing without production impact.

## Planned Architecture
### Core Scoring
In `_beam_search_segment`:
- Compute `progress_by_idx` for each candidate in A->B space (already exists).
- Add experiment config (enabled/weight/shape).
- Compute `target_t` per step.
- Add `experiment_progress_weight * abs(cand_t - target_t)` as a penalty term.

### Configuration
New config block (dry-run/audit only):
```
playlists:
  ds_pipeline:
    pier_bridge:
      experiments:
        progress_arc:
          enabled: true
          weight: 0.25
          shape: "linear" | "arc"
```

### CLI Flags
Dry-run only flags (examples):
- `--pb-experiment-progress-arc`
- `--pb-experiment-progress-weight 0.35`
- `--pb-experiment-progress-shape arc`

### Audit/Diagnostics
Preflight audit payload should record:
- `experiments.progress_arc.enabled`
- `experiments.progress_arc.weight`
- `experiments.progress_arc.shape`

Segment diagnostics should indirectly show:
- Progress distributions for candidates and chosen paths.
- Changes in transition scores and progress deviations.

## Testing Methods
### A/B Run Plan
1) Baseline run (no experiment) with fixed piers.
2) Experiment run (progress arc enabled) with identical seeds and random seed.
3) Compare audit reports and playlists.

### Quantitative Checks
- Progress deviation per step (abs(cand_t - target_t)).
- Distribution of progress values in the chosen path.
- Transition score medians and minimums.
- Failure rates (no valid continuation, pool too small).

### Qualitative Review
Music-expert listening/inspection:
- Early tracks should feel anchored to Pier A.
- Middle tracks should feel like a shared sonic "middle."
- Late tracks should feel closer to Pier B.
- Verify no obvious regressions in flow or pacing.

## Risks and Guardrails
- Risk: over-constraining progress can reduce candidate diversity and force suboptimal transitions.
- Guardrail: keep experiment weight modest (0.2-0.4 range) and require dry-run/audit.
- If transition scores drop too much or failures increase, reduce weight or revert to linear shape.

## Documentation Notes
This experiment is for diagnostic purposes. If results are clearly superior, promote to an optional tuning knob in production, but keep the default path unchanged unless validated across multiple artists and modes.
