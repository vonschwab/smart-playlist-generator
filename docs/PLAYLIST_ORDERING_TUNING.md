# Playlist Ordering Tuning Recipe (2026-05-20)

Opt-in ordering & track-quality knobs added in
`docs/superpowers/plans/2026-05-20-playlist-ordering-and-track-quality.md`.

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

Also watch for `WARNING: T-mismatch edge` lines — these indicate the beam scored an edge as acceptable but the final reporter scored it as below-floor (a known diagnostic for formulation gaps between beam and post-hoc scoring).

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
- `trans_beam=0.25` — the beam scored this edge as passable (0.25 > floor 0.20), but the final reporter scored it 0.092. This is a T-mismatch.
- `local_sonic_cos=0.030` — very low raw cosine. **Scaled local-sonic penalty would demote this significantly.**
- `local_pen=0.021` — current legacy penalty is tiny; confirms the scaled mode would help.

Likely fix for this edge: enable `local_sonic_edge_penalty_mode: scaled` with `scale: 2.0`.
