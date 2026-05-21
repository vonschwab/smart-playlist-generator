# Playlist Ordering Tuning Recipe (updated 2026-05-21)

Opt-in ordering & track-quality knobs added in
`docs/superpowers/plans/2026-05-20-playlist-ordering-and-track-quality.md`,
plus the foundational `transition_weights` alignment learned via the
diagnostic audit on 2026-05-21.

---

## Symptoms that suggest these knobs may help

- High-T transitions still feel jarring (texture/density mismatch despite 0.9+ scores)
- Demo, live, or medley tracks appearing unprompted in playlists
- A few catastrophically bad edges (T < 0.20) per playlist despite high mean T
- "min_transition" in stats is much lower than mean_transition

---

## Step 1: Enable the diagnostic audit (always do this first)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true
```

Generate a playlist. The log will include a **"Selected-edge audit"** section with per-edge:
- `T`, `T_centered_cos`, `S` (sonic cosine), `G` (genre similarity)
- `bridge_score` (harmonic mean sim to both piers)
- `trans_beam` (beam's internal transition score — comparable to T)
- `progress_t`, `progress_jump`
- `local_sonic_cos` (raw cosine of the edge, uncentered)
- `local_pen`, `genre_pen` (penalties applied)
- `title_flags` (artifact flags for the destination track)
- `⚠` prefix on edges below the transition floor

Also watch for `WARNING: T-mismatch edge` lines. These are now regression signals: the beam and final reporter share the same transition metric, so a mismatch usually means stale audit data or a missing-data fallback, not normal tuning drift.

---

## Knob 0: Align `transition_weights` with `tower_weights` (BIGGEST IMPACT)

**Do this first.** Skip the other knobs until this is correct.

`transition_weights` controls how the rhythm/timbre/harmony towers are weighted *inside the beam's transition scoring*. `tower_weights` controls how those towers are weighted in the hybrid embedding used for candidate similarity and the rest of the pipeline.

**These must match.** When they don't, the beam scores transitions in a different feature space than the reporter (and the listener) uses to judge them. The beam will systematically approve edges that score poorly post-hoc.

```yaml
playlists:
  ds_pipeline:
    tower_weights:
      rhythm: 0.20
      timbre: 0.50
      harmony: 0.30
    transition_weights:        # MUST match tower_weights
      rhythm: 0.20
      timbre: 0.50
      harmony: 0.30
```

**Empirical impact on a representative seeded playlist (Geese / 5 piers / 30 tracks):**

| Metric | Rhythm-dominant (0.50/0.25/0.15) | Aligned (0.20/0.50/0.30) | Change |
|---|---|---|---|
| `min_transition` | 0.366 | 0.459 | +0.09 |
| `mean_transition` | 0.828 | 0.898 | +0.07 |
| `p10` | 0.567 | 0.709 | +0.14 |
| Worst edges | METZ→DinoJr (0.366), DinoJr→Ride (0.537) | Those edges gone | — |

**How to verify alignment:** Generate with `emit_selected_edge_audit: true` and compare `T` vs `trans_beam` per row. They should track together. Persistent large gaps (>0.3) indicate weights still diverge somewhere.

**Why timbre-dominant works for indie/rock libraries:** In a relatively homogeneous neighborhood (e.g., indie pop), most tracks share tempo and time-feel — rhythm is a poor discriminator. What listeners actually notice is production style, density, and tone color, which are timbre. The previous rhythm-dominant default approved loud-to-loud handoffs (METZ → Dinosaur Jr) that felt jarring in texture.

---

## Knob 1: Title-artifact penalty (Task 4)

**Use when:** The audit shows `title_flags` like `demo`, `live`, `medley` on bad-feeling tracks.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      title_artifact_penalty:
        enabled: true
        weights:
          demo: 0.10
          live: 0.05
          medley: 0.20
          remix: 0.10
          instrumental: 0.08
          version: 0.05
          take: 0.10
          outtake: 0.15
          alternate: 0.10
```

**Tuning:** Higher weights = stronger demotion. These values are starting points.
`0.10` demotes a demo track by roughly the same magnitude as a moderate bridge score difference.
`0.20` (medley) strongly discourages medleys unless no alternatives exist.

**Warning:** Setting weights above `0.30` may strand long narrow-style segments (One Each / artist mode). If generation starts failing, dial weights down by 50%.

---

## Knob 2: Scaled local-sonic-edge penalty (Task 5)

**Use when:** The audit shows many `local_sonic_cos` values below 0.10 and `local_pen` is tiny (< 0.03). The penalty is decorative in `legacy` mode.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      local_sonic_edge_penalty_enabled: true
      local_sonic_edge_penalty_threshold: 0.15
      local_sonic_edge_penalty_mode: scaled
      local_sonic_edge_penalty_scale: 2.0
```

**Tuning:**
- `threshold: 0.15` flags edges with raw sonic cosine below 0.15 (sonically anti-correlated or orthogonal).
- `scale: 2.0` produces penalties of 0.05–0.30 — comparable to bridge score differences.
- Verify in the audit that `local_pen` values are now non-trivial (0.05+) on triggering edges.
- Suggested experiment: start at `scale: 1.5`, observe `min_transition` in stats, increase to 2.0 if still getting jarring edges.

---

## Knob 3: Worst-edge lexicographic beam objective (Task 6)

**Use when:** `min_transition` in stats is dramatically lower than `mean_transition`, and a few single bad edges ruin otherwise good playlists.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      min_edge_objective: min_edge
```

**Effect:** The beam selects paths that maximize the worst edge, breaking ties by total score. Expected: `min_transition` rises noticeably; `mean_transition` may drop slightly.

**Warning:** This changes what the beam optimizes for. If playlists start feeling "safe but flat", revert to `total_score`.

---

## Knob 4: Edge repair fallback (last-mile guardrail)

**Use when:** The beam is already aligned (`T` and `trans_beam` match in the audit), but rare bad edges remain because the local candidate pool had a better adjacent replacement available after final assembly.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      edge_repair:
        enabled: true
        centered_cos_floor: -0.5
        margin: 0.05
        variety_guard:
          enabled: false
          threshold: 0.85
```

**Effect:** After the beam emits the playlist, repair scans broken edges using the same final transition metric. It may replace one non-seed, non-pier interior track only when both adjacent edges clear the transition floor, neither centered cosine is catastrophic, and the worst adjacent `T` improves by at least `margin`.

**Warning:** Keep this as a fallback, not the primary solution. If `WARNING: T-mismatch edge` appears, fix the metric regression first. Enable `variety_guard` only for dynamic/discover-style playlists where extra similarity between neighbors is acceptable.

---

## Reading the audit table

Example bad edge entry:
```
⚠ Edge #15: Hideous Sun Demon - Gimmicks -> Stove - Nightwalk
  T=0.092 T_centered_cos=-0.817 S=0.306 G=1.000 | bridge=0.55 trans_beam=0.25 title_flags=-
  progress_t=0.850 progress_jump=0.100 local_sonic_cos=0.030 local_pen=0.021 genre_pen=0.000 below_floor=True
```

Interpretation:
- `⚠` + `below_floor=True` — this edge fell below the transition floor in the final emitted playlist.
- `T_centered_cos=-0.817` — the underlying centered cosine is strongly anti-correlated. The rescaled T=0.092 obscures how bad this is.
- `bridge=0.55` — candidate was moderately positioned between both piers.
- `trans_beam=0.25` with `T=0.092` would now be a metric regression. In a healthy run, `trans_beam` should match `T` for the same edge unless the row is stale after repair or metric inputs were missing.
- `local_sonic_cos=0.030` — very low raw cosine. **Scaled local-sonic penalty would demote this significantly.**
- `local_pen=0.021` — current legacy penalty is tiny; confirms the scaled mode would help.

Likely fix for this edge: first investigate any `T`/`trans_beam` mismatch. If they match and are both low, tune upstream scoring (`local_sonic_edge_penalty_mode: scaled`, `min_edge_objective: min_edge`) before enabling edge repair as a fallback.
