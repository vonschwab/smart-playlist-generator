# Genre-Blur Collapse — Attack Design (umbrella)

**Date:** 2026-06-28
**Status:** Strategy approved (brainstorming). Decomposed into 3 sequenced sub-projects; SP1 gets its own detailed spec next.

## The problem (precise)

Long bridges sag into the dense, **blurred-average sonic space *adjacent* to the seeds' genre** — not the seeds' actual niche. Jangle-pop seeds → hazy dreampop reverb (Cocteau Twins, Candy Claws); electronic seeds → four-on-the-floor; every genre has its dense attractor. The result is **homogenization**: every playlist converges to its genre's center-of-mass instead of representing real similarity to *these* seeds.

**This is NOT the same as legitimate breadth.** Yo La Tengo genuinely spans noise-pop to drone, so an ambient YLT track is the *seed's own* material — valid. Genre-blur collapse is different: the dense material is the *genre-average the seed merely lives near*, masquerading as similarity. The distinction is the whole game, and our past metrics couldn't see it.

**Why it happens — the scoring is the root cause, not a side effect.** The beam rewards (a) smooth transitions and (b) being "between" the piers. The genre-blur center is internally smooth *and* central-to-everything, so it scores well on both. The beam drifts there *because* of the score, not despite it. Nothing in the score rewards "this track is distinctively what the seeds are." This is why a collapsed playlist can show an excellent worst edge (the blur transitions smoothly into itself) — smoothness *is* the blur's strength.

## The seed-character signal (drives everything)

Operationally distinguishing "real seed-similarity" from "genre-blur", per the brainstorm:
- **Sonic-primary.** Genre tags are unreliable — scene-mates get tagged differently — so the MERT sonic embedding is the trustworthy signal and tags are at most a light tiebreak, never load-bearing.
- Both candidate framings ("seed-specific vs genre-central" and "rare vs common") collapse, once de-genred, into **one notion**: *close to the actual seeds, and NOT just sitting at the dense center everything is close to.* **Seed-proximity + anti-centrality.**

Two approaches to realize anti-centrality sonically, to be **compared empirically** (SP2):
- **(A) Mutual proximity / hubness correction.** Blur tracks are *hubs* — near everything — so raw cosine inflates their similarity to any seed. Mutual proximity re-scales by how *mutual* the closeness is, deflating hubs. Published fix for exactly this; `roam_mutual_proximity` already implements it, just not in the main beam/admission path.
- **(B) Seed-relative difference.** Score `sim(cand, seeds) − sim(cand, local-dense-center)`; reward tracks closer to the seeds than to the blur. More direct, but needs the "center" defined and is nearer the density approach that already misfired ([[project_beam_redundancy_negative_result]]).

## The objective metric (the breakthrough — and the prerequisite)

We have been **flying blind on collapse all session** because every metric we tried was wrong: transition-smoothness *rewards* the blur; the drift/density metric conflated YLT's legitimate breadth with collapse; and **raw seed-similarity is circular** — blur tracks *are* moderately seed-similar, and both A/B *reduce* those tracks, so any metric downstream of "similarity to seeds" is rigged toward the blur.

The independent metric is **cross-seed distinctiveness** — the "Candy Claws on both playlists" observation, operationalized:

> Pick a corpus of **different-niche seeds adjacent to the same blur** (e.g. a jangle band, a dreampop band, a shoegaze band — all near the dreampop-haze). Generate each one's playlist. Measure how much they **collapse onto the same tracks / the same sonic region** (shared-track overlap + inter-interior sonic similarity across the *different* seeds). **Lower overlap = less collapse = the winner.** Non-circular because it references *difference between seeds' outputs*, never "similarity to the seeds."

**Guardrails (so we don't get fooled again):**
- **Quality floor.** Distinctiveness is gameable by going random. The win must hold *while* each playlist keeps seed-similarity above baseline and its worst edge listenable — distinct-and-still-good, not distinct-at-any-cost.
- **Blind audition is ground truth.** The overlap metric is the cheap automatic proxy; before trusting it to pick a winner, confirm on a few cases (blinded, with a decoy — the sonic-audition pattern) that the lower-overlap playlist actually *sounds* more seed-true and less like genre-wallpaper.

## Decomposition (metric → scoring → structure)

**SP1 — Cross-seed collapse harness (FIRST).** The objective metric above: the seed corpus, the inter-playlist-overlap measure, the quality-floor guards, the blind-audition hook. Prerequisite — we cannot compare A vs B without it — and valuable standalone (our first trustworthy collapse number). Models on `scripts/research/sonic_audition_*`. *Gets its own detailed spec (corpus choice, exact overlap formula, guard thresholds, audition format) next.*

**SP2 — Seed-character scoring fix.** Build BOTH (A) mutual-proximity and (B) seed-relative behind config flags in the beam+admission; run through SP1; the harness (+ audition) picks the winner; **delete the loser** (no lingering dead path). The winner is the scoring fix.

**SP3 — Mini-piers (structural).** Insert high-seed-character, non-artist neighbors as extra piers in long bridges (>5–6 tracks; a 4-pier 30-track playlist runs `[9,9,8]` today — verified), using SP2's winning metric for "high-character" selection, so the spans shorten and *can't* sag. Built last: depends on the metric SP2 settles, and we want to know whether the scoring fix *alone* substantially helps before adding the invasive topology change. Addresses why mini-pier v1 failed ([[project_mini_pier_v1_failed_validation]]) — that gate was *local*; this is structural + length-triggered.

## Out of scope
- Genre-tag-based collapse detection (tags unreliable — sonic-primary by decision).
- Reviving the abandoned absolute-density / beam-redundancy lever (negative result; superseded by the seed-relative / mutual-proximity framing).
- Variable bridge length (shipped separately; complementary — it smooths *landings*, this fixes *drift*).

## Self-review
- **Placeholders:** none at the strategy level. SP1's concrete specifics (corpus list, overlap formula, thresholds, audition UI) are explicitly deferred to SP1's own spec — correct decomposition, not a gap.
- **Consistency:** sonic-primary throughout; the metric is non-circular by construction (cross-seed, never seed-similarity); A and B are both anti-centrality realizations of the one signal; sequencing (metric→scoring→structure) matches the dependency order.
- **Scope:** explicitly decomposed into 3 SPs because it spans multiple subsystems (eval, scoring, topology); each is independently buildable/testable; SP1 is the methodology prerequisite per evaluation-methodology discipline.
- **Ambiguity:** "collapse" pinned to *genre-adjacent blur masquerading as similarity* (NOT legitimate seed breadth); "seed-character" pinned to *sonic seed-proximity + anti-centrality*; "wins" pinned to *lower cross-seed overlap under a quality floor, audition-confirmed*.
