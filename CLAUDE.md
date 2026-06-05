# Playlist Generator — project guide

For the listener-facing feature catalog, see `README.md`. Newest, most authoritative implementation doc is `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`. Active roadmap and audit findings: `audit/07-roadmap.md` (don't re-derive — evidence is cited as `[A#3]`, `[P#1]`, etc.).

## Environment

- **Python 3.11+** required (pinned in `pyproject.toml`; README's "3.8+" is stale).
- **Install:** `pip install -e .[web]` (users) or `pip install -e .[web,dev]` (contributors — adds pytest, ruff, mypy, pre-commit). The legacy PySide6 GUI needs `pip install -e .[gui]`.
- **Test:** `pytest`. Markers: `smoke`, `integration`, `golden`, `gui`, `slow`. Use `-m "not slow"` for fast feedback.
- **Lint / types:** `ruff check` (E, F rules) and `mypy`. The `extend-ignore` list and `[[tool.mypy.overrides]]` modules are intentional — each entry has a comment. Don't relax without flagging.
- **CLI:** `python main_app.py --artist "..." --tracks 30`. Full reference: `docs/GOLDEN_COMMANDS.md`.
- **GUI:** `python tools/serve_web.py` (browser GUI, default port 8770) — the maintained front-end. The legacy PySide6 app (`python -m playlist_gui.app`, install `pip install -e .[gui]`) is deprecated.
- **Doctor:** `python tools/doctor.py`.

## Key paths

- `data/metadata.db` — SQLite track database
- `data/artifacts/beat3tower_32k/data_matrices_step1.npz` — pre-computed sonic + genre matrices
- `data/genre_similarity.yaml` — genre taxonomy overrides
- `config.yaml` — gitignored; copy from `config.example.yaml`

---

## Design principles

Four layers: **destination → architectural commitments → current best methods → engineering discipline**. The first two are durable. The third is replaceable when something better emerges. The fourth governs how we change anything without breaking the rest.

### Layer 1 — North star: what a playlist should *be*

The destination. These don't change if we rewrite the engine tomorrow.

1. **A playlist should feel intentional, like someone curated it.** Not a shuffle, not a random walk through "similar tracks." There's a hand on it.
2. **A playlist has an arc.** It moves somewhere over its duration — energy, mood, texture, era, density. A flat line is a failure mode even if every track is individually good.
3. **A playlist reflects this listener's taste, not music in general.** The user's seeds, library, and history are the gravity. Generic "popular and similar" is the failure mode.
4. **A playlist surprises without disorienting.** Familiar enough to feel like home, fresh enough to discover something. Whiplash and monotony are the twin failure modes.
5. **The worst edge defines the experience.** A single jarring transition breaks a 30-track playlist. Quality means floor quality, not average. Optimize for "no broken moments," not "good on average."
6. **The user should feel listened-to.** What they just heard, what they own, what they seeded — these are respected. The system serves the user; it doesn't impose on them.

### Layer 2 — Architectural commitments

What we believe is *necessary* to reach the north star. Stronger than any one algorithm.

7. **Sonic ⊗ genre fusion is necessary.** Sonic alone misses cultural context; genre alone misses sonic feel. The fusion is the value prop.
8. **Sonic feel is multi-dimensional.** Rhythm, timbre, and harmony each carry independent musical meaning. Collapsing them into a single embedding loses signal.
9. **Multi-genre signatures must be preserved.** A track tagged "shoegaze + dreampop + slowcore" is not "indie rock." Reducing to a single label destroys taste fidelity.
10. **Identity is computed, not raw.** Ensemble suffixes ("Trio", "Quartet"), collaborations ("X feat. Y"), and "The"-prefixes are normalized before any counting. Underpins diversity, dedup, and seed-artist exclusion.
11. **Diversity is part of the experience, enforced as a hard constraint.** Min-gap, per-artist cap, seed-artist disallowed in bridge interiors. Enforce; don't recommend.
12. **Rare > common when expressing taste.** Niche signals (shoegaze, slowcore) outweigh popular labels (indie rock). The user's taste is the ground truth, not the global average.
13. **Recency awareness.** Respect "I just heard this." Recency exclusion lives in candidate-pool construction (pre-order), never post-order.
14. **Local-first.** Desktop app on the user's library. External APIs (Last.fm, MusicBrainz, Discogs, Plex) enrich offline and export at the end — they never gate runtime generation.

### Layer 3 — Current best methods (so far)

What we believe works *today*. Replaceable in principle, but they encode hard-won lessons — don't replace them casually.

15. **Pier-bridge with beam search** is our current best playlist topology. Seeds anchor structure; bridges are beam-searched per segment; progress is monotonic in sonic space.
16. **Vector mode + IDF + coverage bonus** is our current best genre-arc method (Phase 2 + Phase 3). Solved hub-genre collapse and saturation.
17. **Tower-weighted blend with weights rhythm 0.20 / timbre 0.50 / harmony 0.30** is our current best sonic decomposition (timbre dominant). Harmony uses **2DFTM** (2D Fourier Transform Magnitude of the chromagram, 96-dim, key-invariant) — validated 2026-06-03 via blind A/B; replaces the legacy 20-dim chroma-median tower that encoded absolute key. Blend is 162-dim (9 + 57 + 96).
18. **`transition_weights` align with `tower_weights`** (also rhythm 0.20 / timbre 0.50 / harmony 0.30). The beam's per-edge transition score and the reporter's post-hoc T must be computed in the same feature balance, or the beam approves edges the reporter scores poorly. v4.1 fixed a long-standing rhythm-dominant default that produced 0.4+ beam-vs-reporter gaps on timbre-mismatched edges.
19. **Four independent mode axes** are our current best UX for cohesion-vs-discovery: `cohesion_mode` (beam tightness: strict/narrow/dynamic/discover), `genre_mode` (genre pool gating: strict/narrow/dynamic/discover/off), `sonic_mode` (sonic pool gating: strict/narrow/dynamic/off), `pace_mode` (rhythm gating: strict/narrow/dynamic/off). Per-mode pier-bridge knobs (`bridge_floor_<mode>`, `weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`.

### Layer 4 — Engineering discipline

How we evolve the system without breaking the layers above.

19. **Continuous gradients beat hard cliffs.** Centered baselines, smooth squashing, weighted (not binary) scoring. Saturation hides influence; ties hide signal.
20. **Diagnostic logging is part of the feature.** Per-step values, winner-changed counts, saturation indicators, opt-in audit reports under `docs/run_audits/`. If a scoring component can't be measured, it doesn't ship.
21. **Quality metrics are first-class output.** Every generation produces transition stats (min / mean / p10 / p90), weakest-edge report, distinct-artist count.
22. **Opt-in, backward-compatible by default.** New behavior ships behind a config flag with legacy defaults preserved.
23. **Tunability over hardcoded behavior.** Knobs go in `config.yaml`. When a new knob is added, document the tuning recipe.
24. **Pre-compute heavy work; warm path stays fast.** N+1 SQL in generation (audit `[P#1]`) and uncached artifact decodes (`[P#3]`) are bugs, not perf nits.
25. **Edge cases get graceful fallbacks.** <4 artist tracks → error with suggested mode. Infeasible bridge → progressively relax (`bridge_floor` → pool size → beam width → `min_gap`). Don't crash.

---

## Hotspots

Oversized monoliths from earlier phases. Read carefully before editing; prefer extracting helpers over adding to them.

| File | Size | Role |
|------|------|------|
| `src/playlist/pier_bridge_builder.py` | ~5.3k LOC | DJ Bridge beam search + pool union |
| `src/playlist_generator.py` | ~4.1k LOC | Top-level orchestrator |
| `src/playlist/pipeline.py` | ~1.9k LOC | Single-seed DS pipeline (treated as load-bearing) |
| `src/playlist_gui/main_window.py` | ~2k LOC | GUI front-end |
| `src/playlist_gui/worker.py` | ~1.4k LOC | GUI IPC worker |

GUI/backend coupling is already clean (audit `[A#5]`). The architecture problem is god-classes within layers, not coupling between them.

## Project-specific gotchas

- **`data/metadata.db` is irreplaceable — treat it like production.** A full re-analysis takes days. Never write to, migrate, or alter the database without explicit user instruction followed by a second confirmation. Before any write operation, back up the file (`metadata.db.bak` with a timestamp). When in doubt, stop and ask.
- **Music library files are permanently read-only.** The audio files on disk are never written, moved, renamed, or deleted — ever. Read access only, no exceptions.
- **`qtbot` is a no-op stub** (`tests/conftest.py:60-70`, audit `[T#1]`). GUI tests using `qtbot.mouseClick`, `qtbot.waitSignal`, etc. silently pass. Don't trust GUI test results as evidence the GUI works — exercise the feature.
- **Don't re-introduce post-order recency filtering.** Recency lives pre-order, in pool construction. The v3.4 fix exists for a reason — seed tracks at pier positions may be recently played but are explicitly requested.
- **Don't change `transition_weights` without also changing `tower_weights` (or vice versa).** The two weight sets must stay aligned. See `docs/PLAYLIST_ORDERING_TUNING.md` "Knob 0" for the empirical evidence. To diagnose suspected divergence, enable `pier_bridge.emit_selected_edge_audit: true` and compare `T` vs `trans_beam` per row.
- **The 0.20/0.50/0.30 tower weighting is baked into the `tower_weighted` artifact at build time.** The current production artifact uses **2DFTM harmony** (key-invariant 96-dim 2D Fourier Transform Magnitude, validated 2026-06-03) — blend is 162-dim (rhythm 9 + timbre 57 + harmony 96). If you ever re-run the full library analysis from scratch, re-fold via `scripts/fold_2dftm_into_artifact.py` (which also re-runs `scripts/extract_harmony_2dftm_sidecar.py` first). The older `scripts/rebuild_sonic_tower_weighted.py` only re-bakes weights, not features — don't use it alone after a fresh re-analysis.
- **Sonic is not one axis.** Timbre + harmony describe color/texture; rhythm describes pace/energy. `pace_mode` controls rhythm independently (`docs/PLAYLIST_ORDERING_TUNING.md` Knob 5). `dynamic` preserves current behavior; `strict`/`narrow` engage rhythm admission and bridge gates.
- **Raw genre-conflict gate (`candidate_pool.genre_conflict_min_confidence`) is null by default.** Setting it positive against the 764-dim raw artifact vocabulary with identity affinity rejects ~50% of candidates, including legitimate ones. The soft penalty (`genre_conflict_penalty_strength`) still demotes off-axis tracks. See the v4.1 changelog.
- **Segment-pool one-per-artist collapse is OFF by default in `config.yaml`** (`pier_bridge.collapse_segment_pool_by_artist: false`). The beam enforces per-segment artist diversity on its own; the upstream pool collapse only starves the projection of bridging candidates. Don't re-enable without understanding the trade-off.
- **`[[tool.mypy.overrides]]` is for clean modules that should stay clean.** Don't add new modules there to silence errors — type the code instead.
- **`cohesion_mode` drives the beam; the other three slider axes drive pool composition.** All four axes (`cohesion_mode`, `genre_mode`, `sonic_mode`, `pace_mode`) live at `playlists.<axis>` in `config.yaml`. The pier-bridge per-mode knobs (`bridge_floor_<mode>`, `weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`. The old `playlists.ds_pipeline.mode` key is gone — use `cohesion_mode` instead.
- **`sonic_only` mode no longer exists.** The closest equivalent is `genre_mode: off` (disables genre pool gating) combined with `cohesion_mode: discover` (relaxed beam).
- **Genre mode is CLI-only in the current GUI** (commit `2e0000b`). Restoration is roadmap item Tier-2.4. Don't claim parity until it lands.

---

## SQLite schema reference

**Rule:** Always run `SELECT name FROM sqlite_master WHERE type='table'` and `PRAGMA table_info(<table>)` before writing any query against an unfamiliar table. Never guess column names.

### `data/metadata.db`

Primary key convention: TEXT keys everywhere, no integer surrogate IDs.

| Table | Key columns |
|-------|-------------|
| `tracks` | `track_id TEXT` (PK), `artist TEXT`, `album TEXT`, `album_id TEXT`, `artist_key TEXT`, `norm_artist TEXT`, `norm_title TEXT`, `file_path TEXT` |
| `track_genres` | `track_id TEXT`, `genre TEXT`, `source TEXT`, `weight REAL` |
| `track_effective_genres` | `track_id TEXT`, `genre TEXT`, `source TEXT`, `priority INTEGER`, `weight REAL` |
| `albums` | `album_id TEXT` (PK), `title TEXT`, `artist TEXT` |
| `album_genres` | `album_id TEXT`, `genre TEXT`, `source TEXT` |
| `artists` | `artist_name TEXT` (PK — **not** `artist_id` or `id`) |
| `artist_genres` | `artist TEXT`, `genre TEXT`, `source TEXT` — `artist` is the plain name string, **not** a FK integer |

Common patterns:
```sql
-- Genres for an artist (use artist_genres directly)
SELECT DISTINCT genre FROM artist_genres WHERE artist = 'Acetone';

-- Genres for all tracks by an artist (via track join)
SELECT DISTINCT tg.genre
FROM track_genres tg JOIN tracks t ON t.track_id = tg.track_id
WHERE t.artist = 'Acetone';

-- Genres for an album
SELECT genre FROM album_genres WHERE album_id = ?;
```

### `data/ai_genre_enrichment.db`

| Table | Key columns |
|-------|-------------|
| `enriched_genre_signatures` | `release_key TEXT` (PK), `normalized_artist TEXT`, `normalized_album TEXT`, `album_id TEXT`, `signature_json TEXT` |
| `enriched_genres` | `enriched_genre_id INTEGER` (PK), `release_key TEXT`, `genre TEXT`, `confidence REAL`, `status TEXT` |
| `ai_genre_user_overrides` | `release_key TEXT` (PK), `genres_add_json TEXT`, `genres_remove_json TEXT` |
| `ai_genre_source_pages` | `source_page_id INTEGER` (PK), `release_key TEXT`, `source_url TEXT`, `source_domain TEXT` |
| `ai_genre_suggestions` | `suggestion_id INTEGER` (PK), `check_id INTEGER`, `genre TEXT`, `confidence REAL` |

`release_key` format: `"normalized_artist::normalized_album"` where normalization strips Unicode punctuation (P/S categories → space, lowercased). Example: `"acetone::york blvd"` (period stripped).

**Open bug:** `genre_resolver.py::_release_key` uses `normalize_source_tag` (no punctuation stripping), producing `"acetone::york blvd."` — mismatches the stored key. Albums with trailing punctuation silently miss enrichment lookup. Fix: align `_release_key` to use `normalize_release_name` from `normalization.py`.
