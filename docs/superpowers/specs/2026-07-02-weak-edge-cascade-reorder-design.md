# Weak-edge cascade reorder — design

**Date:** 2026-07-02
**Status:** approved (design)
**Scope:** reorder the pier-bridge weak-edge fixers into a least→most destructive
escalation, splitting variable-bridge length into an early add-only pass and a
final remove-only pass. Removal of a track becomes the true last resort.

## Motivation

Today the fixers run: `beam → variable-bridge (bidirectional length) → tail-DP →
assemble → break-glass repair`. Variable-bridge flexes length **both directions
in one early pass**, so it can *shorten a segment (delete a track)* at step 2 —
before tail-DP or repair ever get a chance to fix that edge by **swapping**, which
would have kept the track. We do the most destructive thing (remove content)
before the least destructive things (swap within the existing tracks). That is
backwards.

Rank the levers by how much they violate the listener's intent:

| Lever | Length | Content | Invasiveness |
|---|---|---|---|
| Swap (tail-DP, break-glass repair) | unchanged | same tracks, re-selected from pool | lowest |
| Add (lengthen) | +k | gains bridge tracks | medium |
| Remove (shorten / delete) | −k | **loses tracks, merges edges** | highest |

Removal is the only subtractive act, so it must be the last resort.

**Track count is not a target.** (Dylan, 2026-07-02: the requested track count is
arbitrary — it only sizes bridges relative to the number of artist piers; real
duration tracks average track length, and a future "Continue Playlist" feature
will append freely.) So there is **no length budget**: add and remove each do
what's best for edge quality, bounded only by the 90s compute budget — never by a
target count. All running-total / band bookkeeping is removed.

## The reordered cascade

Per generation:

1. **var-bridge ADD-only** (per-segment, pre-assembly)
2. **tail-DP** (per-segment, pre-assembly) — unchanged
3. **break-glass edge repair** (global, post-assembly) — unchanged
4. **remove-only / repair-by-deletion** (global, post-assembly) — NEW, last resort

### Pass 1 — variable-bridge ADD-only

`src/playlist/pier_bridge_builder.py` var-bridge block (~lines 1721–2019).
`src/playlist/pier_bridge/var_bridge.py::choose_segment_length` is already
length-range-general (`lo`, `hi`) and needs **no change**; the caller changes:

- `lo = nominal` (was `nominal − flex`) — **never shorten** in this pass.
- `hi = nominal + variable_bridge_flex`.
- **Remove** the length-budget machinery: `_vbl_band`, `_vbl_total_dev`, and the
  running-deviation clamp on `lo`/`hi`. Keep the per-segment flex cap
  (`variable_bridge_flex`), the deterministic max-flex-segments cap
  (`variable_bridge_max_flex_segments`, purely a time guard), `variable_bridge_min_edge`
  (the `good_enough` bottleneck), and `variable_bridge_epsilon` (the prefer-shortest
  anti-crutch: only accept a longer length if it beats the worst edge by ≥ eps).
- Update the `Var-bridge seg …` log line to drop `total_dev`.

Add stays conservative (each added track must earn ≥ eps on the worst edge) — not
to respect a count, but because a bridge track should justify its existence and
unbounded adding burns the time budget.

### Pass 4 — remove-only (repair-by-deletion), NEW

New module `src/playlist/repair/edge_delete.py`, function
`delete_broken_edges(indices, *, transition_ctx, floor, protected_positions,
max_deletions) -> DeleteResult(indices, delete_log)`. Invoked in
`pier_bridge_builder.py` immediately **after** the break-glass repair block
(after ~line 2913), operating on the repaired `indices`. Sibling of
`repair_playlist_edges`; reuses the same calibrated blended-T edge scoring
(`score_transition_edge` / the transition context repair already builds).

Algorithm (deterministic, worst-edge-first):

1. Score every adjacent edge. Find edges with `T < floor` (still broken after
   swap-based repair). If none, no-op (byte-identical to today's output).
2. For the worst still-broken edge `(a, b)`: consider deleting whichever endpoint
   is a **deletable interior track** (NOT in `protected_positions` — piers/seeds
   are never removed). Deleting node `x` (with neighbors `prev`, `next`) replaces
   edges `(prev→x, x→next)` with the single merged edge `(prev→next)`.
   - Evaluate the candidate deletion(s): compute the new local window min-edge
     after the merge.
   - **Never-worse:** accept a deletion only if the new local window min-edge
     **exceeds the broken edge's T** (strictly lifts the worst edge). If both
     endpoints are piers, or no deletion improves it, **leave the edge** (the true
     "nothing worked" outcome).
3. After an accepted deletion, re-scan and repeat, capped by `max_deletions`
   (deterministic bound; deletions are rare — they only fire on edges that
   survived add-only, tail-DP, and swap-repair).

Deletion cannot violate diversity: removing a track only *increases* the gap
between same-artist tracks, never decreases it — so no min-gap re-check is needed.
Emit a `delete_log` (mirrors repair's `swap_log`) for the diagnostics dict.

## Config (`src/playlist/pier_bridge/config.py`)

- New (default-on, per activate-fixes): `edge_delete_enabled: bool = True`,
  `edge_delete_floor: float = 0.30` (same floor as repair/tail-DP/var-bridge),
  `edge_delete_max_deletions: int = 4` (time/deterministic cap).
- **Removed** (length budget gone): `variable_bridge_band` and its use. If the key
  lingers in config it is ignored; note in the migration. Keep the other
  `variable_bridge_*` knobs.
- Add `edge_delete_*` to `config.example.yaml` and `config.yaml`.

## Consequence (explicitly accepted)

Making pass 1 add-only means var-bridge **no longer shortens an already-decent
segment to marginally improve an above-floor edge** (e.g. 0.50 → 0.60 by dropping
a meander). Shortening now happens *only* as a last resort on edges still below
0.30. This trades marginal edge-polish on healthy segments for the guarantee that
**content is never destroyed unless an edge is genuinely broken** — the intended
philosophy.

## Testing

- **var_bridge add-only:** with `lo = nominal`, `choose_segment_length` never
  returns a length below nominal; still lengthens when it earns ≥ eps; prefers the
  shortest length within eps of the best. (Unit on `choose_segment_length` +
  a builder-level check that no segment shrinks below nominal.)
- **edge_delete:** deletes the worse interior endpoint of a broken edge when the
  merge lifts the worst edge; **never deletes a protected (pier) position**;
  leaves the edge when neither deletion improves it (never-worse); no-op when no
  edge is below floor; respects `max_deletions`; produces a correct `delete_log`.
  Use synthetic 2-D vectors (the edge_repair test pattern).
- **Composition:** a pure-beam run where a segment is deliberately un-swappable
  (pool has no connector) confirms the edge survives repair and is fixed by
  deletion; a run with a healthy pool confirms deletion never fires.
- Regenerate the config-snapshot goldens (they gain `edge_delete_*`, lose
  `variable_bridge_band`) — deliberate, diff-audited.

## Out of scope

- The "Continue Playlist" feature (append tracks to an existing playlist) — future,
  reinforces that length is fluid but is its own project.
- Any change to tail-DP or break-glass repair internals (they are passes 2 and 3,
  unchanged). This spec only reorders and adds the deletion pass.
- Deleting piers/seeds — never allowed.
