# Phase 4 — Publish adjudicated genres to the authority

**Date:** 2026-06-17
**Branch:** `worktree-phase1-album-adjudicator`
**Status:** design approved, pending plan
**Predecessors:** Phases 0–3 (gold corpus → `album-adjudicator-v1` contract → scorer/eval gate → resumable bulk runner). Roadmap: `docs/GENRE_RESOURCES_AND_CLAUDE_ADJUDICATION_ROADMAP_2026-06-15.md`.

## Goal

Take the completed Pass-1 adjudication (1,242 targeted albums in the shadow DB
`data/adjudication_pass1.db`) and land it in the production genre authority
(`release_effective_genres` in `metadata.db`) so playlist generation actually uses
the tightened, de-bloated genre identities — without re-deriving or disturbing the
~2,186 albums we did **not** adjudicate.

This is the first time the Claude adjudicator writes to production. It is gated by a
conservative trust policy and the mandatory metadata.db backup discipline.

## Inputs (current state, measured)

- `data/adjudication_pass1.db`: 1,242 complete adjudications, 0 failed.
  - 1,077 standard (Haiku) + 165 thorough (Sonnet second pass on ≤2-genre results).
  - "Best result" per album = thorough where present, else standard.
- Disposition of the 1,242 best-results:
  - **112 escalated** → human review (blocking).
  - **1,130 non-escalated** → auto-publish:
    - 682 pure de-bloat (proposed ⊆ prior observed_leaf).
    - 215 sparse (0 prior observed_leaf — anything beats empty).
    - 233 introduce ≥1 genre not in prior authority → auto-publish **and** listed in a
      non-blocking diff report.
- Taxonomy gaps across the 1,242: **119 unique terms, 330 occurrences** (see Sub-phase A).

## Trust policy (decided)

Auto-publish every non-escalated adjudication. Human-review only the escalated set.
Rationale and safety:

1. **The file-tag floor already routes the only irreversible loss to review.** Dropping
   a user's own embedded file tag is the one unrecoverable error; `enforce_file_tag_floor`
   force-escalates exactly those, so they land in the 112, never auto-published.
2. **metadata.db is backed up (timestamped) before the publish write**, with a second
   confirmation, per CLAUDE.md data-safety rules.
3. **Reversible.** Prior authority + every adjudication with full provenance are retained
   (the shadow DB is never deleted). A systematic error is fixed by re-materialize + re-publish.
4. **Residual risk is bounded and audited.** The only non-escalated risk is an *invented*
   genre (proposed ∉ prior tags, Mulatu→afrobeat class). The 233 such albums go to a
   non-blocking diff report for post-hoc skim — no review queue.

## Pipeline order (data-first — non-negotiable)

```
A. taxonomy growth (gap terms)         # new terms must exist before they can resolve
   → re-canonicalize adjudication      # picks up newly-added genre_ids
B. materialize adjudication → sidecar  # surgical: only adjudicated albums
C. two lanes (auto 1,130 / review 112) + diff report (233)
D. publish() → release_effective_genres  (backup first)
   → rebuild artifact → verify end-to-end
```

If we publish before growing the taxonomy, terms like `ethio-jazz`, `chicago soul`,
`kankyo ongaku`, `dance-rock` canonicalize to `unknown` and are silently dropped from the
published observed-leaf set.

---

## Sub-phase A — Taxonomy growth (gap terms)

**Governed by the `taxonomy-growth` skill.** Do not hand-edit the YAML; run the loop.

Triage of the 119 gap terms into the skill's buckets:

- **Real genre adds** (own records, with parents/`similar_to`): `chicago soul` (11),
  `future jazz` (7), `kankyo ongaku` (6), `aor` (4), `rare groove` (3), `dance-rock` (3),
  `rumba` (3), `horror rock` (3), `countrypolitan`, `country soul`, `exotica`,
  `minneapolis sound`, `dark jazz`, `library music`, `deep funk`, `cape jazz`,
  `south african jazz`, `brazilian jazz`, `world jazz`, `garage psych`, `heavy psych`,
  `indie dance`, `future funk`, `afro-soul`, `electro-disco`, `pop rap`, `p-funk`,
  `rap-rock`, … (final list set during the loop with placement judgment guardrails).
- **Aliases of existing canonical** (spelling/format variants → `alias_proposal`):
  `neo-classical`→neoclassical, `r and b`/`r and b soul`/`rnb/swing`→rhythm and blues,
  `avantgarde`→avant-garde, `bossanova`→bossa nova, `alt country`→alternative country,
  `prog rock`→progressive rock, `bubblegum pop`→bubblegum, `psych folk`→psychedelic folk,
  `jazz-funk` → resolve the documented DB↔YAML coherence bug (DB alias maps to jazz fusion;
  YAML returns unknown — align so runtime steering resolves it).
- **Facet routes** (belong in the facet vocabulary, not genres): `lo-fi` (39), `drone` (27),
  `abstract`, `minimal`, `orchestral`, `ballad`, `pastoral`, `instrumental`, `c86`,
  `african`, `symphonic`, `tribal`, `modal`, `suite`. Most already classify as facets;
  confirm and add any missing.
- **Rejects** (typos / compounds / umbrellas the adjudicator should not emit): `indie`
  (13, already rejected), `electronicnica`/`indie-electronicnic` (typos), `funk / soul`,
  `rock pop`, `alternative pop/rock`, `jazzy hip-hop`, `post-hardcore punk`, etc.

**Gates (skill):** pre-analysis script → batch script (`validate_proposal` → N OK / 0 FAIL)
→ isolated-copy ingest test → **human approval** → live ingest (`0.X.0-…-grown`) →
PASS-N edge upgrade for same-batch forward refs → tests → commit (one pass per commit) →
status note. Output: `data/layered_genre_taxonomy.yaml` bumped past `0.12.1`.

After growth, re-run canonicalization over the 1,242 so the gap rate drops before publish.
Terms still unknown after growth (genuine deferrals) are recorded for a later SP3a pass and
simply omitted from the published set (not invented).

---

## Sub-phase B — Adjudication materializer (shared apply path)

**New module:** `src/ai_genre_enrichment/adjudication_materializer.py`. **TDD.**

One unit, used by both lanes. **What it does:** turn one album's adjudication response
into the layered genre/facet rows the sidecar holds, and write them for that album only.

```
materialize_adjudication(
    store,                 # SidecarStore (writes genre_graph_release_genre_assignments)
    *, release_id, album_id, artist, album,
    response,              # validated adjudicator response (genres[], facets[], ...)
    taxonomy,
) -> AdjudicationMaterializeSummary
```

**How it works (reuse, don't reinvent):**
- Route each proposed genre term through the existing `classify_layered_term(taxonomy, term)`.
  Facet/reject/review/family/leaf are handled exactly as the hybrid materializer does — so
  facet-leaks (`lo-fi`, `drone`) land in the facet table, not genres; unknown terms are skipped
  (recorded, not invented).
- For each leaf: emit `observed_leaf`, then walk `taxonomy.parents_for_genre` →
  `inferred_parent` and `taxonomy.families_for_genre` → `inferred_family`. This mirrors
  `compute_layered_assignment_rows`; factor the shared expansion so both call one helper.
- Replace that album's rows in `genre_graph_release_genre_assignments` (surgical — the store's
  `replace_layered_assignments_for_release` already does per-release replace). Every
  non-adjudicated album is untouched (grandfathered).
- Provenance per row: `source="claude_adjudicator"`, `prompt_version`, model, confidence from
  the response.

**Depends on:** the layered taxonomy, the sidecar store, `classify_layered_term`, the shared
leaf-expansion helper. **Does not** touch `release_effective_genres` (that's `publish()`'s job)
and **does not** know about lanes/escalation (the runner decides which albums to call it for).

**Why the sidecar, not overrides or a direct authority write:** the sidecar's
`genre_graph_release_genre_assignments` is the canonical place layered assignments live and the
single source `publish()` reads. Writing there keeps `publish()` the only writer of
`release_effective_genres` (genre-data-authority One Rule) and leaves the artifact builder
unchanged. The adjudicator is the new pipeline stage superseding hybrid-fusion materialization
for adjudicated albums.

---

## Sub-phase C — Two lanes + diff report

**New script:** `scripts/research/apply_adjudication.py` (resumable, like the bulk runner).

- **Auto lane:** for each of the 1,130 non-escalated best-results, call
  `materialize_adjudication`. Idempotent; safe to re-run.
- **Review lane (112 escalated): lightweight CLI/report, not a React view.** Decided:
  a CLI surface that, per escalated album, prints prior observed_leaf vs proposed set +
  `escalate_reason` + any `dropped_file_tags`, and accepts **accept / edit / reject**.
  - accept → materialize as-is.
  - edit → materialize a corrected genre set (typed inline).
  - reject → leave the album's existing authority untouched (no materialize).
  - Decisions persist to a small table/file so the review is resumable and reusable for the
    remaining 2,186 albums later. Rationale: 112 is small, album-grain semantics don't fit the
    term-grain review queue, and the React+NDJSON path trips the web-gui traps for a one-off.
- **Diff report (non-blocking):** the 233 "added ≥1 new genre" albums →
  `docs/genre_adjudication/phase4_added_genres_report.md` (album / prior / proposed / new terms).
  No queue, no gate; a post-hoc skim so a systematic invented-genre error can't hide.

**Ordering within C:** auto lane and review lane both write to the sidecar; run auto first,
then review, then regenerate the diff report from the final materialized state.

---

## Sub-phase D — Publish, rebuild, verify

1. **Backup** `metadata.db` → `metadata.db.bak.<timestamp>`. Second confirmation before write.
2. **`publish(metadata_db, sidecar_db, dry_run=True)`** first — inspect `PublishStats`
   (graph_albums, legacy_albums, unlinked, collisions). Then `dry_run=False` to commit.
3. **Rebuild artifact** `data/artifacts/beat3tower_32k/data_matrices_step1.npz` via
   `scripts/build_beat3tower_artifacts.py` (genre_source already `graph`). Follow the
   artifact-backup discipline; never touch MERT shards/sidecar.
4. **Verify end-to-end:**
   - Spot-check a known album: Kendrick — *To Pimp a Butterfly* should read the tight
     `jazz rap / conscious hip hop / g-funk / funk / neo-soul` observed_leaf, not the prior
     8-tag bloat.
   - Confirm `display_genre_names_for_track` (authority path) returns the new set.
   - Run one multi-pier generation through the `gui_fidelity` harness; confirm it loads and no
     regression in transition stats.

## Testing strategy

- **Materializer (B):** TDD unit tests — facet-leak routes to facets; leaf expands to
  parent+family; unknown term skipped (not invented); per-release replace touches only the
  target album; provenance stamped. Use a fixture taxonomy, not the live YAML.
- **Apply runner (C):** unit test the lane split (escalated vs non-escalated), the
  invented-genre classification feeding the diff report, and review-decision persistence
  (accept/edit/reject → resumable).
- **Publish (D):** existing `tests/unit/test_genre_publish.py` covers `publish()`; add a case
  that an adjudication-materialized album appears in `release_effective_genres` with the tight set.
- **No live metadata.db writes in tests** — use temp DBs.

## Data safety

- metadata.db: backup + second confirmation before publish (CLAUDE.md). Dry-run before commit.
- Taxonomy YAML: isolated-copy test before every live ingest; one pass per commit.
- MERT shards/sidecar: never touched.
- Surgical principle: only adjudicated albums are materialized; all others grandfathered.
  Never wholesale re-derive the library to deploy this.

## Out of scope (future)

- The remaining 2,186 albums (well-tagged majority) — same formula, resumable, later pass.
- Flipping the live adjudicator config from `gpt-4o-mini` to the Claude adjudicator (Phase 5).
- SP3a coverage feedback loop for terms still unknown after Sub-phase A.

## Open items

- Final genre-add vs alias vs reject split for the 119 gap terms is set during the
  taxonomy-growth loop (Sub-phase A), not pre-committed here.
- Exact persistence shape for review decisions (small table in the shadow DB vs a YAML
  decisions file) — decided at plan time; must be resumable.
