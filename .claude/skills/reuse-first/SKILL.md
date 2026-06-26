---
name: reuse-first
description: Use BEFORE writing new code, adding a function/module, or adding a dependency in this repo. Walks a reuse ladder — existing repo code, then stdlib/native, then an already-installed dependency — tuned to this project's utilities and god-class hotspots, before any new code is written.
---

# Reuse first

The best change reuses what already exists. Before writing new code, walk this
ladder and stop at the first rung that holds. This is *not* an excuse to skip
understanding the problem, and it never overrides this repo's instrumentation
discipline (see "Never minimize away" below).

## Rung 0 — Understand first

Read the code the change touches and trace the real flow before picking a rung.
A small diff you don't understand is guessing, not reuse. (Pairs with CLAUDE.md:
read the generation logs / grep before answering.)

## Rung 1 — Is it already in this repo?

`grep` before writing. High-value reuse targets that get reinvented:

- generic helpers: `src/artist_utils.py`, `src/string_utils.py`,
  `src/logging_utils.py`, `src/playlist/utils.py`
- identity / normalization: `src/ai_genre_enrichment/normalization.py`
- genre reads: `src/genre/authority.py` (and the `genre-data-authority` skill —
  every genre consumer must read through authority.py)
- runtime / mode config: `src/playlist_gui/policy.py::derive_runtime_config`
  (never hand-roll mode strings — standing gotcha)
- energy: `src/playlist/energy_loader.py`

## Rung 2 — Hotspot rule

If the change lands in a god-class — `src/playlist/pier_bridge_builder.py`,
`src/playlist_generator.py`, `src/playlist/pipeline/` (package),
`src/playlist_gui/worker.py` — extract a helper. Do not grow the monolith
(CLAUDE.md Hotspots).

## Rung 3 — Stdlib / native?

`itertools`, `functools`, `dataclasses`, `pathlib`, `statistics`,
`collections`. Prefer numpy/scipy vectorization over hand-rolled loops — both
are already dependencies.

## Rung 4 — Already-installed dependency?

Check `pyproject.toml` / `web/package.json` before reaching for a new
dependency. Adding a dependency is a deliberate act — confirm rungs 1–3 can't
cover it first, then flag the addition explicitly.

## Rung 5 — Only then write new code

Write the minimum that works, following the patterns already in the file.

## Never minimize away

Reuse-first is about *not duplicating*, not about stripping a feature. Keep, in
full, regardless of rung:

- diagnostic logging, quality metrics, opt-in audit reports (Design Principles 20–21)
- input validation, error handling, graceful fallbacks (Principle 25)
- config knobs / tunability (Principle 23)
- anything the user explicitly asked to keep

## Output

After the change, note in 1–2 lines what existing code/stdlib/dep you reused, or
why new code was unavoidable.
