# Upstream Transition Alignment + Edge Repair Fallback Plan

## Summary

Replace the repair-first approach with upstream transition-metric alignment. Pier-bridge beam scoring, builder stats, reporter edge scores, and edge repair all use one final-edge metric so the search judges transitions the same way the audit reports them.

The repair pass remains useful, but only as an opt-in fallback after upstream scoring is aligned.

## Implementation Order

1. Add a shared transition metric module exposing:
   - `TransitionMetricContext`
   - `build_transition_metric_context(...)`
   - `score_transition_edge(context, prev_idx, cur_idx) -> dict`
   - `is_broken_transition(edge, transition_floor, centered_cos_floor) -> bool`
2. Route `pier_bridge/beam.py`, `pier_bridge_builder.py`, `reporter.py`, and `ds_pipeline_runner.py` through the shared helper.
3. Keep beam hard gates on shared `T < transition_floor`; add the `T_centered_cos < -0.5` catastrophic gate only when `center_transitions=True`.
4. Keep `min_edge_objective` compatible, but compare shared `T` values.
5. Add focused parity tests before adding repair behavior.
6. Add the opt-in `pier_bridge.edge_repair` fallback using the same shared metric helper.
7. Recompute final IDs, seed positions, edge scores, playlist stats, and repair logs immediately after a successful swap.
8. Update docs to frame T mismatch as a regression and repair as a last-mile guardrail.

## Shared Metric Contract

The shared edge dict includes at least:

- `T`
- `T_raw`
- `T_centered_cos`
- `H`
- `S`
- optional `G`

Missing-data fallback cases may omit or neutralize some components, but beam and reporter should not disagree for edges that have the same available metric inputs.

## Edge Repair Rules

Repair is default-off via `pier_bridge.edge_repair.enabled: false`.

Candidate eligibility excludes:

- seeds
- existing playlist indices
- duplicate track keys
- disallowed seed or pier artists
- title artifacts
- allowed-set violations

Conservative acceptance rules:

- Piers are never replaced.
- For bad edges into a pier, repair may replace the source interior track, never the pier.
- A swap is accepted only if both resulting adjacent edges clear `transition_floor`.
- A swap is rejected if either resulting edge has `T_centered_cos < -0.5`.
- Worst-of-two adjacent transition quality must improve by at least `0.05`.

If repair changes `final_indices`, beam-only audit components for swapped edges are stale and should be omitted or explicitly marked stale.

## Test Plan

- Beam `trans_score_in_beam` equals reporter `T` for centered transitions.
- `T_centered_cos < -0.5` is detected as catastrophic.
- Builder `edge_scores` match final reporter `edge_scores`.
- Repair no-ops when upstream beam already produces clean edges.
- Repair refuses artist, seed, duplicate, allowed-set, title-artifact, and pier violations.
- Repair handles destination-edge swaps and source-before-pier swaps.
- Repair recomputes IDs, stats, and logs after swaps.

Run:

```powershell
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_beam_vs_final_t_diagnostic.py tests/unit/test_selected_edge_audit.py -q
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_edge_repair.py -q
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q
C:\Windows\py.exe -3.13 -m pytest -q
```

Golden ordering changes are acceptable only when caused by the upstream metric fix and should be inspected before updating goldens.
