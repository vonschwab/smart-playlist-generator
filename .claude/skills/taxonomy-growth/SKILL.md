---
name: taxonomy-growth
description: How to build, wire, and maintain the layered genre taxonomy graph (data/layered_genre_taxonomy.yaml) via the SP3a growth loop. Use whenever ingesting a GPT placement handoff, adding/editing genre records or edges, writing a batch or edge-upgrade script, or processing the deferred-edge queue. Encodes the same-batch-forward-reference rule and the PASS-N split that the single-snapshot validator forces.
---

# Genre taxonomy growth & maintenance

The layered taxonomy is a graph DB of genre records (`data/layered_genre_taxonomy.yaml`) that the genre-enrichment program (SP1–SP5) is making the authoritative vocabulary. This skill is the **operational playbook for growing it**: turning a GPT placement handoff into validated records that ingest cleanly and stay internally consistent.

Program context (the "why") lives in memory: `project_genre_enrichment_program`, `project_enriched_genre_authority`. This skill is the "how." Read the Core Loop and the Same-Batch rule before touching the taxonomy.

## Where things live

| Thing | Path |
|---|---|
| Taxonomy graph (production, git-tracked) | `data/layered_genre_taxonomy.yaml` |
| Growth engine | `src/ai_genre_enrichment/graph_growth.py` |
| Loader / model / `normalize_taxonomy_name` | `src/ai_genre_enrichment/layered_taxonomy.py` |
| Batch + edge-upgrade scripts (one-off, outside repo tree) | `C:\tmp\sp3a_taxonomy_handoff\` |
| Incoming GPT handoffs | `C:\Users\Dylan\Downloads\*_handoff*.md` |
| Status notes back to GPT | `C:\tmp\sp3a_taxonomy_handoff\*_status_for_gpt.md` |

The YAML is git-tracked (unlike `metadata.db`), so a bad ingest is recoverable — but still **isolated-copy test before every live write** and **commit one pass per commit**.

## Core Loop

Each tranche of placements goes through this loop. Steps 4, 6, 7 are human-gated.

1. **Read the handoff.** GPT proposes per-term: `PROPOSE` (a record), `DUPLICATE OF` (an alias), or `RECLASSIFY`. Note every `DECISION_NOTE` flagging same-batch or not-yet-placed references.
2. **Pre-analysis (script it, don't eyeball).** For every parent/`similar_to`/`canonical_target` target in the tranche, check existence against the live taxonomy (`canonical names ∪ alias names`). Bucket each into: **exists** (keep), **same-batch** (trim → PASS-N restore), **missing-external** (trim → deferred queue). See [Same-Batch rule](#the-same-batch-forward-reference-rule).
3. **Build the batch script** (`build_<batch>_batch.py`) from the template below. Generates a human-review YAML; runs `validate_proposal` preflight. Expect `N OK, 0 FAIL`.
4. **Isolated-copy ingest test** → report record-count delta, version, new-records-by-kind, the PASS-N edge queue. **Present for human review. Do not live-ingest yet.**
5. **Live ingest** (after approval): `append_approved_to_taxonomy` into `data/layered_genre_taxonomy.yaml`, bump version to `0.X.0-<batch>-grown`.
6. **PASS-N edge-upgrade** (after approval): restore the same-batch edges trimmed in step 2 via an edge-upgrade script, bump to `0.X.Y-<batch>-edge-upgrade`. Isolated-copy test it first.
7. **Deferred queue:** process any *previously*-deferred edges this batch just unblocked as their own PASS-N.
8. **Tests + commit** (one commit per pass). Then **status note back to GPT**.

## Build-script template

Helpers + `GrowthProposal` are the whole API. Model new scripts on `build_group1_pass2_batch.py`.

```python
_PROJECT_ROOT = _find_project_root(Path.cwd().resolve())   # cwd, NOT __file__ —
sys.path.insert(0, str(_PROJECT_ROOT))                     # scripts live in C:\tmp
from src.ai_genre_enrichment import graph_growth as gg

def edge(target, edge_type, weight, confidence):
    return {"target": target, "edge_type": edge_type, "weight": weight, "confidence": confidence}

def genre_proposal(name, kind, specificity, status, parents, similar_to, alias_variants, rationale):
    return gg.GrowthProposal(name=name, kind=kind, status=status,
        specificity_score=specificity, parent_edges=parents, similar_to=similar_to,
        alias_variants=alias_variants, term_kind_confirm="genre", rationale=rationale)

def facet_proposal(name, facet_type, specificity, status, alias_variants, rationale):
    return gg.GrowthProposal(name=name, kind="facet", status=status,
        specificity_score=specificity, parent_edges=[], similar_to=[],
        alias_variants=alias_variants, term_kind_confirm="facet",   # "facet" for facets
        facet_type=facet_type, rationale=rationale)

def alias_proposal(name, target, rationale):
    return gg.GrowthProposal(name=name, kind="alias", status="alias_only",
        specificity_score=0.0, parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="genre", canonical_target=target, rationale=rationale)  # "genre" even for alias
```

Then in `main()`: `gg.write_proposals(OUT_PATH, [(GrowthCandidate(term=p.name, album_frequency=0), p) for p in all_proposals])`, re-load the YAML and patch every row to `decision: keep` (this batch is human-reviewed, nothing is "pending"), and run `gg.validate_proposal(taxonomy, p)` for each as preflight.

**`GrowthProposal` fields:** `name, kind, status, specificity_score, parent_edges, similar_to, alias_variants, term_kind_confirm, rationale, facet_type, canonical_target`.

- `term_kind_confirm` = `"facet"` for facets, `"genre"` for **everything else** (genre/subgenre/umbrella/**and alias**).
- `parent_edges` only on leaf/umbrella kinds; **facets must have none and no `similar_to`** (validator rejects).
- A leaf (`genre`/`subgenre`/`microgenre`) needs **≥1 parent edge** or validation fails.

## Data model reference (authoritative — from `enums:` block in the YAML)

| Enum | Values |
|---|---|
| `kind` | family, umbrella, genre, subgenre, microgenre, facet, alias, reject |
| `status` | active, review, deprecated, alias_only, rejected |
| `facet_type` | mood, texture, instrumentation, production, era, region, function, vocal, scene, format, rhythm |
| `edge_type` | is_a, family_context, scene_adjacent, fusion_of, style_modifier, bridge_to, alias_of |

**Only `is_a` and `family_context` count as parent edges** (`parent_edge_types`) — they drive specificity inheritance and `_parent_target_error`. The others are lateral.

**Edge shapes we actually use (match these or the graph gets inconsistent):**

| Intent | edge_type | weight | confidence | notes |
|---|---|---|---|---|
| Strong taxonomic parent | `is_a` | 0.70–0.80 | 0.85 | null |
| Softer family membership | `family_context` | 0.25–0.65 | 0.70–0.85 | null |
| Hand-curated lateral adjacency | `scene_adjacent` | ~0.45 | 0.85 | null |
| `similar_to` (auto-converted on ingest) | `bridge_to` | **0.40** | **0.60** | **"similar_to (growth)"** |

**`similar_to` is not a separate channel** — on ingest `_proposal_record` appends each `similar_to` target as a `bridge_to` edge in `parent_edges` (weight 0.4 / conf 0.6 / notes "similar_to (growth)"). That's why a `similar_to` to a same-batch sibling must be trimmed just like a parent edge: the loader resolves it by canonical name and can't see a record created in the same run.

**Default parent-edge fill** (if you omit weight/confidence in `edge()`): weight 0.55, confidence 0.8, edge_type `family_context`. Always set them explicitly.

## The Same-Batch Forward-Reference rule

**This is the #1 recurring gotcha. Internalize it.**

`validate_proposal` and the loader resolve every target name against a **single starting snapshot**. A record created later in the *same* batch does not exist yet at validation time. So any edge whose target is **also new in this tranche** fails — even though it'll be valid once both land.

**The fix (always the same):** trim the forward edge from the PASS-1 batch, ingest, then **restore it in a follow-up PASS-N edge-upgrade** once both endpoints exist. Document the trim inline in the proposal's `rationale` so the human-review YAML shows exactly what was deferred and why.

Precedent: R&B→PASS 2, doom/stoner→PASS 3, G1P1→PASS 2/4, G1P2→PASS 6. The pattern never changes.

### PASS-N edge-upgrade script

Model on `pass6_group1_pass2_edge_upgrade.py`. A flat `UPGRADES` list of `(source, target, edge_type, weight, confidence, notes)`, then:

1. **Validate first:** each `source` resolves via `taxonomy.genre_by_name(normalize_taxonomy_name(source))`; each `target` passes `_parent_target_error(taxonomy, target)`. Abort on any failure.
2. **Idempotent apply:** dedupe by **`(normalize_taxonomy_name(target), edge_type)`** — not target alone (a source can have both an `is_a` and a `bridge_to` to the same target). Skip edges already present.
3. Bump `taxonomy_version`, write with `yaml.safe_dump(..., sort_keys=False, allow_unicode=True)`.
4. Support `--dry-run` and `--isolated-copy PATH`.

Restored `similar_to` edges get the **bridge_to 0.40/0.60/"similar_to (growth)"** shape. A restored *parent* edge keeps its original `family_context`/`is_a` weight from the proposal.

## DUPLICATE OF / alias resolution

When GPT says `DUPLICATE OF: <target>`, resolve by checking the target:

| Situation | Action |
|---|---|
| Target is canonical, **no existing alias** for the term | Create a `kind: alias` record (`alias_proposal`) |
| Target already has that alias | **No-op** — omit it (don't double-add) |
| Term is already canonical under that exact name (self-referential) | **No-op** — no record needed |
| Target **doesn't exist** | Stop — re-review; don't invent the alias |

Always verify the target exists before emitting an alias. `alias_variants` on a canonical proposal auto-create alias records too (same `_alias_record` helper), skipping any variant equal to the record name.

**`_alias_record` limitation:** it hardcodes `alias_policy: {"type": "plain"}` and `notes: "Spelling variant (growth)."`. No `GrowthProposal` field overrides either. If an alias needs nuance (e.g. "scoped to this corpus's co-occurrence"), it can only live in the proposal `rationale` (visible in the human-review YAML), not on the landed record.

## Deferred-edge queue

Edges waiting on a not-yet-placed term. Maintain the running queue **in the status note to GPT** (it's the cross-session memory for this). After each tranche:

1. **Add** this batch's missing-external trims (e.g. `broken beat → acid jazz`, `uk garage → dubstep`) to the queue.
2. **Drain** any queued edges this batch just unblocked, as their own PASS-N. (E.g. G1P1 landing `reggae`/`hard rock`/`nu jazz` unblocked 9 edges → PASS 5.)
3. The queue is empty only when every waiting edge's target exists.

## Placement judgment guardrails

When GPT's `kind`/placement is wrong, or you're placing directly, apply these (each is a hard-won user directive):

- **Umbrellas are broad context, low specificity (≈0.24–0.42), spread parentage.** No single child-branch gets a strong parent weight, so the umbrella never overpowers precise children (e.g. `dance` spread across electronic/pop/r&b/funk so it doesn't dominate house/disco/dance-pop).
- **Don't turn instrument-led terms into genre leaves.** `piano jazz`, `jazz trumpet`, `jazz guitar` → instrumentation **facets** (or aliases to one), not subgenres — unless there's a genuine scene/style tradition beyond the instrument.
- **Don't collapse distinct genres into co-occurrence aliases.** `uk garage` ≠ `garage rock`; a corpus-driven `garage → garage rock` alias must not absorb it. Place it as its own genre.
- **Specificity ladder (rough):** umbrella 0.24–0.42 · genre 0.48–0.66 · subgenre 0.62–0.82. Broad/contextual or noisy terms → `status: review`, not `active`.

## Trap Catalog (every one cost real debugging)

| Trap | Symptom | Fix |
|---|---|---|
| Same-batch forward ref left in PASS-1 | validation FAIL on a target that "should" exist | Trim → PASS-N restore. See the rule above. |
| `read_proposals` ≠ list of `GrowthProposal` | `AttributeError: 'ProposalEntry' object has no attribute 'kind'` passing it to `append_approved_to_taxonomy` | It returns `list[ProposalEntry]`; ingest wants `[e.proposal for e in entries if e.decision == "keep"]`. |
| `append_approved_to_taxonomy(..., dry_run=...)` | `TypeError: unexpected keyword` | No `dry_run` param. Signature is `(path, approved, *, new_version)`. For dry runs, ingest into a temp copy. |
| Anchoring root on `__file__` | `RuntimeError: could not locate project root` | Scripts live in `C:\tmp`, outside the repo. Use `_find_project_root(Path.cwd().resolve())` and run from repo root. |
| Dedup by target only | duplicate or missing edges | Dedupe by `(target, edge_type)` — a source can hold multiple edge types to one target. |
| Adding a term breaks a "goes-to-review" test | `test_graduate_reviewed_writes_to_yaml`-style failure | `classify_source_tags` now resolves the term directly. Use `"xyzzy unknown genre"` as the genuinely-unknown source tag. |
| Trusting `test_ai_genre_hybrid_cli.py` | 3 pre-existing failures even on unmodified taxonomy | Not your regressions. Always deselect all three: `--deselect tests/unit/test_ai_genre_hybrid_cli.py::test_hybrid_enrich_one_apply_persists_accepted_signature --deselect tests/unit/test_ai_genre_hybrid_cli.py::test_hybrid_enrich_one_apply_can_include_provisional_lastfm_terms --deselect tests/unit/test_ai_genre_hybrid_evidence.py::test_lastfm_only_is_rejected_noise`. Note the path prefix is `tests/unit/`, not `tests/`. |
| `test_specific_lastfm_only_terms_are_provisional_when_release_evidence_exists` fails | `AssertionError: Left contains one more item: 'folk'` (or similar canonical term) | Pre-existing. Deselect: `--deselect tests/unit/test_ai_genre_hybrid_evidence.py::test_specific_lastfm_only_terms_are_provisional_when_release_evidence_exists`. Verify it fails on unmodified taxonomy (git stash) before treating as a regression. |
| Adding a term that was previously an "unmapped genre" test fixture | 5 different failure modes in `test_graph_growth.py` — empty candidate set, DuplicateError on append, "No growth candidates found", rc==1 on ingest | Replace the fixture with `"xyzzy-growth-test"` (or other unmappable nonsense). Also check aliases: `"vapor wave"` → `"xyzzy-gt"` (distinct normalization required; the alias `"xyzzy growth test"` normalizes identically to `"xyzzy-growth-test"` and would be skipped). |
| Facet with parent_edges / similar_to | validator rejects | Facets are modifiers — no edges. Put relationships on the genre side. |
| `microgenre` kind in `validate_proposal` | `FAIL 'name': Unsupported kind: microgenre` | The validator doesn't support `microgenre`. Ingest works fine (`_proposal_record` produces `role=leaf, kind=microgenre`). Skip in preflight: `if p.kind in ("reject", "microgenre"): skipped += 1; continue`. |
| Facet used as `similar_to` target | `FAIL 'name': parent target 'X' is a facet…` | `similar_to` converts to `bridge_to` edges, which are parent-edge types. Facets cannot be parent-edge targets. Drop the entry entirely — no PASS-N restoration (not a same-batch issue; a type constraint). |
| Alias name used as `similar_to` target (e.g., `jazz-funk`) | `FAIL 'name': parent edge target does not exist: 'jazz-funk'` | Conditional aliases and exact-hyphen aliases are not canonical names. `_parent_target_error` can't find them. Drop the entry and note it in the rationale. |
| Pre-existing record same name as proposal | `FAIL 'name': a taxonomy record named/slugged like 'X' already exists` | The term is already in the taxonomy (may be a different kind — e.g., proposed as umbrella but exists as facet). Skip the proposal; check the existing record's kind and decide whether the existing placement is appropriate. |
| `reject` kind proposals passed to `validate_proposal` | `FAIL 'name': Unsupported kind: reject` | The validator doesn't support `reject` kind. Skip reject proposals in the preflight loop: `if p.kind == "reject": skipped += 1; continue`. |
| `reject` record missing `reject_reason` on live ingest | `ValueError: Rejected record is missing reject_reason: <name>` | Every `reject`-kind record needs a `reject_reason` from the enum in the YAML `enums:` block: `label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise`. Patch the YAML rows before ingest. Common mappings: Discogs/retail category labels → `retail_bucket`; noisy source-tag garbage → `source_noise`; misspelled/broken terms → `malformed`. |
| `write_proposals` YAML is a bare list, not a dict | `TypeError: list indices must be integers or slices, not str` when doing `data["proposals"]` | `write_proposals` writes a top-level **list** of `{term, album_frequency, cooccurring_tags, examples, decision, proposal}` dicts. Patch with `for row in data:`, not `for row in data["proposals"]:`. |
| `LayeredTaxonomy` has no `._genres` | `AttributeError: 'LayeredTaxonomy' object has no attribute '_genres'` | Use the public API: iterate via `taxonomy.genres` or look up by name with `taxonomy.genre_by_name(normalize_taxonomy_name(name))`. Never access private `_genres`. |

## Validation & safety checklist

- [ ] Pre-flight: `validate_proposal` → `N OK, 0 FAIL` before any ingest.
- [ ] Isolated-copy ingest test (temp copy of the YAML) before **every** live write — report count delta, version, new-by-kind, PASS-N queue.
- [ ] Human approval gate before live ingest **and** before each PASS-N.
- [ ] One pass = one commit. Version string in the commit matches the bumped `taxonomy_version`.
- [ ] Run the enrichment unit tests after ingest; re-import the sidecar if downstream consumes it.
- [ ] Never edit `metadata.db` as part of this — taxonomy growth touches only the YAML.

## Status-note-back-to-GPT protocol

After a tranche closes, write `<batch>_status_for_gpt.md` so the next GPT session has accurate state. Include:

- **What landed exactly as proposed** vs **where you diverged and why** (GPT calibrates on the divergences).
- **Alias tables, kept separate:** "alias variants created alongside canonical records" vs "plain standalone aliases added" (`canonical target | alias`). Conflating them confuses GPT — an alias's canonical is the target, not the term.
- **Current taxonomy state:** version + record counts by kind.
- **Deferred queue** (what's waiting on which unplaced term) and **priority terms** for the next pass (those that unblock queued edges).
- **Warnings** that must survive (e.g. "`uk garage` must NOT collapse into `garage`").

## Version naming

- Ingest pass: `0.X.0-<batch>-grown` (e.g. `0.5.0-group1-pass2-grown`).
- Edge-upgrade / deferred-drain pass: `0.X.Y-<batch>-edge-upgrade` (e.g. `0.5.1-group1-pass2-edge-upgrade`).

Current live version is in the YAML's `taxonomy_version`; `git log` is the pass history. Don't hardcode "current" counts here.

## Maintenance protocol ("keep this current")

This SKILL.md is the index of how we grow the graph, not a one-time doc. Update it when:
1. A new trap bites → add a Trap Catalog row.
2. The `graph_growth` API changes (new `GrowthProposal` field, signature change) → fix the template and field list.
3. A new edge-shape convention is established → add it to the edge-shapes table.
4. A placement guardrail is set by the user → add it to Placement judgment.
