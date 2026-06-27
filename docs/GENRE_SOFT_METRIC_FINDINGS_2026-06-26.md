# Genre soft-cosine metric — calibration findings & verdict (2026-06-26)

**BLUF — the soft-cosine genre-pair metric does NOT beat the incumbent `max` metric, and is NOT
activated.** After the sonic centered-transition fix landed (discriminating sigmoid), the genre-pair
penalty has very little room to act, and in that room `max@.30` already captures the available value.
Soft is inert on cohesive seeds, its one diverse-seed win did not replicate (1 of 4 wide trios),
it degrades the worst *sonic* edge on some diverse seeds, and it adds a new failure mode
(over-penalizing within-genre subgenre variation). **Decision: keep `max` as the default genre-pair
metric; do not make `soft` the default; do not delete the `max` provider.** This supersedes the
"make soft default + delete max" plan in `GENRE_SOFT_CALIBRATION_PLAN.md` and the roadmap in
`HANDOFF_2026-06-25_genre_finish_roadmap.md`.

The single most valuable finding is unrelated to soft-vs-max — see **§5 (the Embassy reframe)**.

---

## 1. What was tested

- **Mechanism (held on `worktree-roam-corridors-engine`, rebased onto the sonic sigmoid fix):**
  `SoftGenrePairSimProvider` (soft cosine over the hub-damped taxonomy graph, full weighted
  signature incl. broad tags) + a graded pairwise edge penalty, selectable via
  `genre_pair_metric: max|soft` and `genre_pair_penalty_mode: floor|graded`. Knobs ride the
  `roam` dict (the penalty only fires under roam).
- **Configs per seed (7-cell sweep):** `OFF`, `max@.30` (the shipped flat-0.5 floor),
  `soft+graded` × floor {0.45, 0.55, 0.65} (strength 2.0), `soft+flat` × floor {0.45, 0.55} (pen 0.5).
- **Corpus (8 seeds):** RD (cohesive precision case), Metallica / Madvillain (genre-distinct solos),
  4 cross-genre **trios** (Codeine+Herbie+Aphex; BH+BoC+YLT; Madvillain+BillEvans+Deerhunter;
  Aphex+Sufjan+ModestMouse; Herbie+BoC+YLT).
- **Instruments:** (a) aggregate sweep — worst-genre-adjacency lift `dGcohW` + worst-sonic-edge `minT`
  + feasibility/budget, measured against the **new sigmoid** with roam enabled; (b) **impostor probe**
  (`scripts/research/impostor_probe.py`) — the surgical test of soft's one structural advantage;
  (c) **tracklist dumps** for an ear audition.
- **Harness:** `scripts/research/slider_differentiation_eval.py` (`--genre-cal`, `--dump`), routed
  through the real policy layer (`derive_runtime_config`). `gcoh*` = adjacent soft-genre cosine of the
  OUTPUT playlist, measured with the *actual* soft provider the beam uses.

## 2. Aggregate results — `softG@0.55` (the candidate) across the 4 wide trios

| wide trio | `dGcohW` (genre, worst adj.) | `minT` Δ (sonic worst edge) | read |
|---|---|---|---|
| Codeine + Herbie + Aphex | **+0.140** | +0.040 | the one win |
| Madvillain + Bill Evans + Deerhunter | −0.002 | +0.043 | no genre help |
| Aphex + Sufjan + Modest Mouse | +0.000 | +0.031 | no genre help |
| Herbie + BoC + YLT | +0.016 | **−0.096** | **sonic cratered** |

Genre lift on **1 of 4**; a real worst-sonic-edge **degradation on another**. Knife-edge confirmed:
on Herbie+BoC+YLT, floor 0.45 *worsened* genre (−0.143) and floor 0.55 *worsened* sonic (−0.096).

## 3. Cohesive / solo seeds — `max` wins or both inert

- **The Radio Dept.** (cohesive): `max@.30` lifts worst genre adjacency **+0.080** (0.505→0.585) by
  demoting truly-disjoint edges. **Soft is byte-identical to OFF** (both modes, floors 0.45/0.55;
  strength 0.5/1.0/2.0 all identical) — it fires 470–1904× but never tips a selection. At 0.65 it
  *over-fires* (5230 hits) and *degrades* (−0.017). → **max wins, soft inert.**
- **Metallica** (genre-isolated): no candidate shares any tag → provider sims are `None` → **both
  metrics structurally inert** (0 hits). Fine; nothing to do.
- **Madvillain**: max 716 hits, soft 8855–29358 hits, but **output unchanged for all** (gcohW 0.346,
  minT 0.226 everywhere). Both inert on the selected path.

## 4. Impostor probe — soft's one structural advantage is real but rare + mixed

Soft's *only* thing max can't do: catch a **shared-one-tag impostor** (max saturates to 1.0 on one
shared tag → never demotes; soft scores it ~0.44 → demotes). Probe = among each seed's sonic top-60,
count edges with `max≥0.80 & soft<0.45` (soft's *unique* demotes):

- max & soft **agree ~90%** on the sonic-close pool (corr **0.86–0.99**). Unique demotes: **0–3 per seed**.
- Quality is **mixed**: true catches (Caribou→Herbie, electronic vs jazz; Hiroshi Yoshimura→Aphex)
  *and* likely false-positives (**Common + A Tribe Called Quest → Madvillain** — all hip-hop, soft
  over-penalizes *subgenre* variation; **Broadcast → Beach House** — a legit dreampop neighbor).

So soft's edge over max is real but (a) rare, (b) taste-dependent, (c) sometimes wrong.

## 5. THE KEY REFRAME — the genre-pair penalty is the wrong lever for "include the great neighbor"

Probing "why isn't **The Embassy** (the canonical RD real-neighbor) in the RD playlist?" exposed the
most important lesson of the session. Sonic rank of each artist's best track to a RD track, out of
40,752 non-RD tracks (rank 0 = closest):

| artist | sonic rank | soft→RD | in RD playlist? |
|---|---|---|---|
| Beach House | 31 | 0.90 | ✅ |
| **The Embassy** | **34** | **0.62** | ❌ **absent (all configs)** |
| M83 | 198 | 0.90 | ✅ |
| Pains of Being Pure at Heart | 233 | 1.00 | ✅ |
| Oneohtrix Point Never | 385 | 0.44 | ❌ |
| Ducks Ltd. | 407 | 0.68 | ✅ |
| Club 8 | 445 | 0.90 | ❌ |
| Sambassadeur | 848 | 0.96 | ❌ |

The Embassy is the **34th-closest** track in the library to RD *and* a genre neighbor (soft 0.62), yet
absent — while tracks at rank 198 / 233 / 407 are present. **The genre-pair penalty can only DEMOTE
off-axis edges; it can never PROMOTE a genre-good neighbor.** Embassy is genre-good, so the penalty
(max *or* soft) never touches it. Whether Embassy gets a bridge slot is decided by **sonic proximity
+ the beam's per-segment transition optimization** — a different lever entirely. (This is also why
soft's RD playlist == OFF: there was nothing for genre to do.)

**Implication:** if the goal is "the playlist contains the best sonic+genre neighbors," the lever is
**sonic admission / beam selection** (or a genre *promotion* signal that doesn't exist yet) — NOT the
genre-pair metric we spent this session calibrating. *Open, not yet diagnosed:* whether Embassy is
admitted-but-not-selected (legit beam optimization) or quietly excluded — needs a real INFO
generation log (gate-tally + per-segment pool lines), per the read-the-logs discipline.

## 6. Why soft had so little room (the root cause)

The sonic centered-transition fix (the discriminating sigmoid) made the **sonic objective dominate**
selection. The genre-pair penalty is now a minor tiebreaker. In that small room the *egregious*
genre-disjoint edges (which `max` catches well, as a coarse "share any tag?" detector) carry most of
the value; soft's *gradation* has little additional room and what little it has is taste-dependent and
occasionally wrong. No re-calibration of a **blanket pairwise floor** escapes this.

## 7. Decision & what it means for the branch

- **Keep `max` as the default genre-pair metric** (`genre_pair_metric: max`, floor 0.30, flat). It is
  robust, never degrades, and delivers the modest cohesive-seed lift. It is *good at a coarse job*,
  not merely least-worst.
- **Do NOT** make soft the default; **do NOT** delete the `max` provider.
- The soft mechanism on the branch is a **tested-and-rejected scaffold** — per "never merge an inactive
  fix," it should not merge as a feature. Remove the soft code (cleanest) or keep it explicitly as an
  off-by-default experiment.
- Branch was rebased onto the sonic fix for testing; since soft isn't shipping, the **sonic fix should
  still land via its own branch**, and this branch is not merged as a feature.

## 8. Reusable tooling left behind

- `scripts/research/impostor_probe.py` — max-vs-soft disagreement among sonic-close candidates. The
  right instrument for any future genre-metric A/B (the aggregate coherence sweep is too coarse).
- `slider_differentiation_eval.py --genre-cal` — floor×strength×mode sweep (OFF/max/soft) with
  `dGcohW` / `minT` / `pair_hits`, roam-enabled, policy-routed.
- `slider_differentiation_eval.py --dump` — ordered `artist — title` tracklists with adjacent
  soft-genre sims + worst-edge marker, for ear auditions.

## 9. If we ever revisit: the higher-ceiling thread

Not soft-as-blanket-floor. Two focused options, in priority order:
1. **The Embassy lever** (§5): make the beam/admission surface the best *sonic+genre* neighbors
   (rank-34 Embassy should not lose to rank-407 tracks). This is the quality lever that actually
   moves playlists.
2. **Surgical impostor demotion**: a penalty that fires *only* on the impostor signature
   (sonic-close AND genre-far, e.g. Caribou→Herbie) and leaves within-genre variation alone — the
   targeted version of what soft tried to do as a blanket floor.
