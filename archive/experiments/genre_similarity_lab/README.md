# Genre Similarity Lab

This workspace is for prototyping the next-generation genre similarity and playlist cohesion system. It is separate from production code so we can iterate quickly and measure impact before integrating.

## Objectives
- Build a richer genre representation (aliasing, source-aware weights, TF-IDF, learned relations).
- Refresh the ensemble scorer (size-normalized pairing, configurable weights, confidence-aware thresholds).
- Prototype a learned genre–genre matrix from local co-occurrence and blend it with the curated matrix.
- Add an evaluation harness to tune weights/thresholds against labeled pairs and sanity checks.
- Explore segment-aware playlist ordering (transition fit, energy/key arcs) using existing multi-segment sonic data.

## Directory Layout
- `PLAN.md` — concrete milestones and tasks.
- `notebooks/` — optional scratch space (kept empty here; use locally).
- `src/` — prototypes (normalization, weighting, learned matrix, evaluation harness, playlist ordering experiments).

## How to use
1. Follow `PLAN.md` tasks in order; each task is bounded and testable.
2. Run prototypes against the existing SQLite DB (`data/metadata.db`) or small extracts.
3. Capture findings (metrics, edge cases) back into `PLAN.md` or a short `notes.md`.

## Constraints
- Keep everything here non-invasive to production code until validated.
- Default to ASCII in code; use comments sparingly to explain non-obvious logic.
