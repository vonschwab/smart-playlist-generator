# Break-glass edge repair — design

**Date:** 2026-07-01
**Status:** approved (design), pending implementation plan
**Scope:** re-aim the existing post-order edge-repair pass (`src/playlist/repair/edge_repair.py`)
at weak edges. Phase 2 — the beam landing-edge investigation — is explicitly out of scope
(see "Deferred").

## Problem / evidence

An Alvvays artist run (2026-07-01, `C:\Users\Dylan\Desktop\tmp\alvvays_weak_edge_log.txt`)
emitted a worst edge of **T=0.003 / S=0.103** (Beat Happening "Drive Car Girl" → pier
Alvvays "Dives"), plus a weak cluster (0.219–0.329) in the following segment. A probe over
the full artifact (reproducing the log's 0.003 exactly) showed the library held **T=0.902**
landings into Dives (Hovvdy "Thru"; Porches "Intimate" 0.861; Beach House "She's So Lovely"
0.854 — Porches and Beach House were already in the playlist). The worst edge was avoidable;
the beam left ~0.9 on the table.

Why every existing defense missed it:

- **Edge repair is enabled but its trigger is dead.** `repair_playlist_edges` fires only on
  `is_broken_transition`, which (since the roam promotion removed the T hard gate) checks
  only `centered_cos < −0.5` — unreachable in MuQ's all-positive cosine space (this
  playlist's floor: +0.062). Hence `repair_applied: false` on a 0.003 edge.
- **Beam minimax** (`worst_edge_minimax_enabled`) still chose 0.003 over an available ~0.9 —
  either the landing hop isn't folded into `_state_min_edge`, or the good candidates never
  made the segment pool, or a path-shaping modifier (anti_center collapse-preventer, roam
  penalty, progress, genre penalty) out-competed them. Pinning which is **Phase 2**.
- **Var-bridge** triggered (`variable_bridge_min_edge=0.3`) but a length change cannot fix a
  bad neighbor choice.
- **Latent bug found:** `_candidate_refusal_reasons` never checks positional **min_gap** — a
  repair swap could violate artist spacing. Never mattered (repair never fired); must be
  fixed as part of this work.

## Decisions (Dylan)

- Repair is a **break-glass fallback**: fire when possible; if no better transition exists,
  leave the edge alone and emit as-is. Never worse, never fail, never block generation.
- When an edge is under threshold, **path-shaping modifiers do not get a vote** in choosing
  the replacement — pure transition quality wins. (The modifiers exist to shape the journey;
  one of them likely hid the good candidate.)
- Beam landing fix is a separate follow-up, informed by this pass's validation.

## Design

**1. Trigger.** Edge needs repair when `T < edge_repair_t_floor` (new `PierBridgeConfig`
knob + `config.yaml` key, default **0.30** — aligned with `variable_bridge_min_edge`) OR the
existing anti-alignment check (`centered_cos < edge_repair_centered_cos_floor`, kept).
The predicate lives in `edge_repair.py`; **`is_broken_transition` is untouched** (its
no-hard-gate semantics are a deliberate roam-design decision shared by other callers).
Process triggered edges **worst-first**, re-scoring adjacent edges after each accepted swap.

**2. Break-glass acceptance.** For each triggered edge, scan the candidate universe for a
replacement of the swappable slot; accept the best candidate with
`new_worst_T ≥ old_worst_T + edge_repair_margin` (existing knob, 0.05). If nothing clears
that bar, leave the edge alone. **Drop the current requirement that new edges fully clear
the floor** (`_all_edges_clear` on T): lifting 0.003 → 0.25 is accepted even though
0.25 < 0.30. Keep the centered-cos anti-alignment check on new edges.

**3. Candidate evaluation = pure transition quality.** Score candidates by
`min(T_in, T_out)` (the existing `_worst_t` of adjacent edges). By construction this ignores
anti_center, roam corridor penalty, progress monotonicity, genre penalty, and bridge score.
Hard constraints stay hard: duplicate track, pier/seed protection, seed-artist-in-interior
ban (`disallowed_artist_keys`), artist caps, allowed pool, title artifacts — **plus a new
positional min_gap check** (same-identity artist within `min_gap` positions of the swap slot
→ refuse, using the artist-identity resolver like the cap check does).

**Scope boundary:** pool admission gates (sonic floor, genre, onset) are upstream of the
universe and are NOT relaxed here. Repair searches `universe` (the deduped pier-bridge pool
passed at `pier_bridge_builder.py:2774`) as-is. If validation shows the good candidates were
pool-gated out entirely, that is Phase-2 evidence, not a change to this pass.

**4. Slot choice.** Existing logic unchanged: for edge `u→v` with `v` a pier, swap `u`
("source_before_pier" — exactly the Dives case); interior edges swap the destination `v`;
pier→pier edges are logged and skipped.

**5. Config / rollback (Layer 4).** Live default ON: `edge_repair_enabled` is already true;
`edge_repair_t_floor: 0.30` lands in `config.yaml` (and `config.example.yaml`). Rollback:
`edge_repair_t_floor: 0` reverts to anti-alignment-only (today's behavior). One INFO line
per executed repair (old/new track + old/new worst-T) and one summary line
(`edges_triggered / repaired / left_alone`); the existing `swap_log` / `repair_applied`
metrics plumbing is already wired through to the DS report.

**6. Testing & validation.**
- Unit (pure function, synthetic matrices): fires under floor; accepts best-effort partial
  lift; leaves alone when nothing clears margin; never returns a lower worst-T than input;
  min_gap / dedup / pier / disallowed-artist refusals hold; worst-first ordering; t_floor=0
  reproduces today's no-op.
- Validation: rerun the Alvvays generation. Expect the 0.003 landing lifted substantially
  (toward ~0.85–0.9 if Hovvdy/Porches/Beach House candidates are in the universe — the run
  will tell us), `below_floor → 0`, `repair_applied: true`, mean/p50 essentially unchanged.
- Golden pipeline JSONs containing sub-0.30 edges will legitimately change → regenerate
  deliberately, diff-audited (per the 2026-06-26 false-green lesson: regen ANY playlist-T
  golden affected).

## Deferred (Phase 2 — separate effort)

Beam landing-edge investigation: why the beam chose 0.003 over ~0.9 — landing-hop inclusion
in `_state_min_edge`, segment-pool membership of the top landing candidates, and which
path-shaping modifier out-competed them; then decide whether the beam itself should relax
modifiers when a prospective edge falls under a threshold. Also: outlier-pier placement
(endpoint preference) if repair + beam fix still leave pier-landing residue.
