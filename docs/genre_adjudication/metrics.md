# Gold-corpus eval metrics & protocol

Defines how a genre adjudicator is scored against `corpus.yaml` (50 ratified releases:
20 failure / 20 control / 10 sparse). This is the **Phase-0 "metrics documented" exit criterion**
for the Claude-adjudication roadmap; it is consumed by the **Phase-2 blind A/B gate** (nothing
writes the authority until an adjudicator clears it). Follows the `evaluation-methodology` skill.

## Unit of comparison
Per release, compare a candidate's proposed **genre set** to `gold_genres` at **observed-leaf**
granularity. Facets (`gold_facets`) are scored separately, never mixed into the genre set.
**Canonicalize both sides** via `src/genre/graph_adapter.py::canonicalize_tag` before comparing
(normalizes hyphen/space, resolves aliases) so `soul-jazz` == `soul jazz`, etc.

## Per-release metrics (then aggregate as DISTRIBUTIONS, never just means)
Let P = proposed set, G = `gold_genres`.
- **Precision** = |P ∩ G| / |P| — de-bloat: penalizes over-inclusion.
- **Recall** = |P ∩ G| / |G|.
- **F1**.
- **Noise rate** = |P \ G| / |P| — the headline metric given the library-wide over-tagging.
- **File-tag preservation** = fraction of `must_preserve` ⊆ P — the **floor metric**; a dropped
  `must_preserve` genre is the worst single failure (the "never drop the user's correct tag" rule).
- **Taxonomy-gap rate** = fraction of releases whose gold needs a `taxonomy_gaps` term (coverage signal).
- **Escalation correctness** (if the adjudicator escalates): was it a genuine gap / ambiguity?

**Reporting:** min / p10 / p50 / p90 per metric, **per bucket** (the worst release defines trust —
north star #5). State N and pool. Means alone are not acceptable (they hide floor failures).

## What each bucket tests
- **control (20)** — gold is a clean *subset* of the current observed-leaf set. Expect **high precision
  + high file-tag-preservation**: the adjudicator must *prune without adding errors*. Regression guard.
- **failure (20)** — current authority is wrong/over-tagged (e.g. Portishead missing `trip-hop`, Mulatu
  `afrobeat`, Sabbath missing metal, Tim Hecker `emo`). Expect Claude to **fix**: recall gain + noise
  drop **vs the current published observed-leaf set**.
- **sparse (10)** — legacy-only / thin evidence (0–2 observed leaves). Expect Claude to **produce the
  correct identity from little/no evidence** (the `model_prior` own-knowledge path); escalation allowed.

## Phase-2 blind A/B protocol
Three arms, scored against gold, **blind** (scorer cannot see which arm is which):
- **A** — Claude adjudicator output.
- **B** — current published authority (observed-leaf set per release).
- **C** — decoy (e.g. the full *bloated* authority incl. inferred, or random canonical genres) to
  measure discrimination — if the scorer can't separate C, the metric is broken.

**Gate:** A must beat B on **noise rate** AND not regress **file-tag preservation**, on the failure +
control buckets, before any write-back. Report distributions + per-bucket + N. Iterate the contract on
fail; do not proceed.

## Independence / circularity guardrails (the thing that makes this valid)
- Gold is **Dylan-ratified**, drafted from raw evidence + music knowledge, **never** from an adjudicator
  or the enriched-genre space. The corpus is **held out** from any prompt-tuning.
- Never score genre quality inside the same space being adjudicated (the past "QC said fine while tags
  were junk" failure). The gold labels are the independent arm.
- Residual risk (logged): Claude drafted the gold, so it may anchor toward Claude's later answers —
  mitigated by Dylan's edits and by the failure bucket's *objective* ground truth (file tags that were
  demonstrably right). Treat any A-beats-B margin on the *control* bucket with extra caution.

## How to compute (when the harness is built)
For each `corpus.yaml` entry: canonicalize P and `gold_genres`; compute the metrics above; verify
`must_preserve ⊆ P`; exclude `taxonomy_gaps` terms from the canonical-match denominator (count them in
gap-rate instead). Adapt the existing `scripts/research/genre_audition_*` harness rather than rebuilding.
