# Codex Project Instructions

These instructions apply to Codex work in this repository. They supplement the
default review prompts below and carry forward the durable project principles
from `CLAUDE.md`.

## Review Stance

@codex review for security regressions, authz/authn, and error-handling edge cases. Be thorough and list all findings you can.

@codex review focusing on: correctness, test gaps, and backward compatibility. If something is ambiguous, ask me questions in-thread.

@codex review for performance concerns (hot paths, allocations) and concurrency/thread-safety.

## Project Context

- Use `README.md` for the listener-facing feature catalog.
- Treat `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` as the most authoritative implementation document.
- Use `audit/07-roadmap.md` for active roadmap and audit references; do not re-derive already-cited findings.
- Python 3.11+ is required. Install with `pip install -e .[gui]` for users and `pip install -e .[gui,dev]` for contributors.
- GUI work uses PySide6. Older references to PyQt are stale.

## Product Principles

- Playlists should feel intentional and curated, not random.
- Playlists should have an arc across energy, mood, texture, era, or density.
- The listener's own library, seeds, and history are the source of truth.
- Discovery should surprise without disorienting.
- The worst transition matters; optimize for no broken moments, not just good averages.
- The system should respect explicit seeds, recent listening, and the user's local collection.

## Architectural Commitments

- Preserve sonic and genre fusion. Sonic-only and genre-only are valid modes, but the core value is their interaction.
- Treat rhythm, timbre, and harmony as independent sonic dimensions.
- Preserve multi-genre signatures. Do not collapse nuanced genre sets into one broad label.
- Use computed artist identity for diversity, dedupe, collaborations, ensembles, and seed-artist exclusion.
- Enforce diversity constraints as hard constraints where the engine expects them.
- Weight rare taste signals more strongly than generic labels.
- Keep recency filtering in candidate-pool construction, never post-order validation.
- Keep the app local-first. External APIs enrich and export; they must not gate runtime generation.

## Current Best Methods

- Pier-bridge with beam search is the current best topology for multi-seed playlists.
- Vector mode, IDF weighting, and coverage bonus are the current best genre-arc approach.
- Tower PCA with rhythm 0.20, timbre 0.50, harmony 0.30 is the current best sonic decomposition.
- Independent mode presets `strict`, `narrow`, `dynamic`, `discover`, and `off` for genre and sonic axes are first-class UX and compatibility requirements.

## Engineering Discipline

- Prefer continuous gradients over hard cliffs when changing scoring.
- Diagnostic logging is part of the feature. If a scoring component cannot be measured, it should not ship.
- Quality metrics are first-class output: transition stats, weakest-edge reports, and distinct-artist counts matter.
- New behavior should be opt-in and backward-compatible by default unless the user explicitly asks otherwise.
- Prefer tunable config over hardcoded behavior. Document new knobs and tuning recipes.
- Pre-compute heavy work; keep generation hot paths fast. N+1 SQL and repeated artifact decodes are bugs.
- Edge cases need graceful fallbacks and actionable errors, not crashes.

## Safety Rules

- Treat `data/metadata.db` like production data. Do not write, migrate, or alter it without explicit user instruction and a second confirmation. Back it up before approved writes.
- Music library files are permanently read-only. Never write, move, rename, or delete audio files.
- Do not reintroduce post-order recency filtering.
- Do not relax lint/type gates by broadening ignores without flagging it.
- GUI test results are useful but not sufficient evidence by themselves; verify important GUI behavior through code-path inspection or direct exercise.

## Hotspots

Read these files carefully before editing; prefer extracting helpers over adding to monoliths:

- `src/playlist/pier_bridge_builder.py` - DJ Bridge beam search and pool union.
- `src/playlist_generator.py` - top-level orchestration.
- `src/playlist/pipeline.py` - single-seed DS pipeline.
- `src/playlist_gui/main_window.py` - GUI front end.
- `src/playlist_gui/worker.py` - GUI IPC worker.
