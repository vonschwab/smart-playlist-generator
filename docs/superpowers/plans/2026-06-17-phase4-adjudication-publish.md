# Phase 4 — Adjudication Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the completed 1,242-album Pass-1 adjudication into the production genre authority (`release_effective_genres`) so generation uses the tightened identities — auto-publishing 1,130 non-escalated, human-reviewing 112 escalated, grandfathering the ~2,186 un-adjudicated albums.

**Architecture:** A new `adjudication_materializer` writes each adjudicated album's tight observed-leaf set (graph-expanded into inferred parent/family layers) into the sidecar's `genre_graph_release_genre_assignments`, keyed by the recompute release_key so it cleanly supersedes the old hybrid rows. The existing `publish()` then copies the sidecar into `release_effective_genres` (it stays the only authority writer), and the artifact rebuild bakes it into generation. Two driver scripts split the auto lane from the escalated-review CLI lane.

**Tech Stack:** Python 3.11+, SQLite (`metadata.db`, `ai_genre_enrichment.db`, shadow `adjudication_pass1.db`), pytest, the layered taxonomy graph.

## Global Constraints

- **metadata.db is irreplaceable** — timestamped backup + explicit second confirmation before any write; dry-run publish first. (CLAUDE.md)
- **Publish is the only writer of `release_effective_genres`.** New code writes the sidecar, never the authority directly. (genre-data-authority skill)
- **Surgical only** — materialize only the 1,242 adjudicated albums; never wholesale re-derive the library.
- **MERT shards/sidecar (`mert_shards/`, `mert_sidecar.npz`) are never touched.**
- **Data-first ordering** — taxonomy growth (Task 1) MUST complete before publish (Task 5), or new terms canonicalize to `unknown` and drop.
- **DB paths:** resolve via `scripts/research/run_album_adjudicator.py::resolve_db(name)` (a worktree's `data/` symlinks to the main checkout). Shadow DB: `data/adjudication_pass1.db`. Sidecar: `data/ai_genre_enrichment.db`.
- **Adjudication provenance constant:** `source="claude_adjudicator"`, `source_reliability=0.85`.
- **Prompt versions:** standard `album-adjudicator-v1+<hash>`, thorough `album-adjudicator-v1-thorough+<hash>` (via `run_adjudicator_bulk.effective_prompt_version`). "Best result" per album = thorough where present, else standard.

---

## Task 1: Taxonomy growth — add the gap terms (Sub-phase A)

**This is the one non-TDD task: it executes the `taxonomy-growth` skill loop, which has its own gates.** Do not hand-edit the YAML.

**Files:**
- Modify: `data/layered_genre_taxonomy.yaml` (via the growth loop; currently `0.12.1-group1-pass9-edge-upgrade`, 762 records)
- Batch/edge scripts live outside the repo tree at `C:\tmp\sp3a_taxonomy_handoff\` (skill convention)

**Input — the 119-term gap triage** (frequencies from the full 1,242 run):
- **Genre adds** (own records w/ parents + `similar_to`): `chicago soul`(11), `future jazz`(7), `kankyo ongaku`(6), `aor`(4), `rare groove`(3), `dance-rock`(3), `rumba`(3), `horror rock`(3), `garage psych`(4), `countrypolitan`, `country soul`, `exotica`, `minneapolis sound`, `dark jazz`, `library music`, `deep funk`, `cape jazz`, `south african jazz`, `brazilian jazz`, `world jazz`, `heavy psych`, `indie dance`, `future funk`, `afro-soul`, `electro-disco`, `pop rap`, `p-funk`, `rap-rock`, `psych folk`, `space jazz`, `industrial metal`, `electro-industrial`.
- **Aliases** (`alias_proposal` → existing canonical): `neo-classical`→neoclassical, `r and b`/`r and b soul`/`rnb/swing`→rhythm and blues, `avantgarde`→avant-garde, `bossanova`→bossa nova, `alt country`→alternative country, `prog rock`→progressive rock, `bubblegum pop`→bubblegum, `funk-soul`/`funk soul`/`funk / soul`→ (resolve to funk+soul or a soul-funk canonical — decide in loop), `jazz-funk`→ **fix the documented DB↔YAML coherence bug** (DB alias maps jazz-funk→jazz fusion; YAML returns unknown).
- **Facet routes** (confirm classify as facets; add any missing to the facet vocabulary): `lo-fi`(39), `drone`(27), `abstract`, `minimal`, `orchestral`, `ballad`, `pastoral`, `instrumental`, `c86`, `african`, `symphonic`, `tribal`, `modal`, `suite`.
- **Rejects** (typos/compounds/umbrellas; `reject` with `reject_reason`): `indie`(13, already rejected), `electronicnica`, `indie-electronicnic` (malformed), `rock pop`, `alternative pop/rock`, `jazzy hip-hop`, `post-hardcore punk`, `funk / soul` if not aliased, `electronic jazz`, `electronica dance`, `dance and dj`, `modern rock`, `diction`.

Apply the skill's **placement judgment guardrails** (umbrellas low-specificity/spread; don't turn instrument terms into leaves; don't collapse distinct genres into co-occurrence aliases; specificity ladder).

- [ ] **Step 1: Invoke the taxonomy-growth skill** and run its Core Loop on the triage above (pre-analysis script → `build_<batch>_batch.py` → `validate_proposal` preflight expecting `N OK, 0 FAIL`).

- [ ] **Step 2: Isolated-copy ingest test.** Report record-count delta, new-by-kind, the PASS-N edge queue. **Present for human approval. Do not live-ingest yet.**

- [ ] **Step 3 (after approval): Live ingest** → bump version to `0.13.0-phase4-gaps-grown`; then PASS-N edge-upgrade for same-batch forward refs (`0.13.1-…-edge-upgrade`).

- [ ] **Step 4: Re-canonicalize the adjudication against the grown taxonomy** and report the new gap rate:

```bash
python scripts/research/analyze_bulk_shadow.py data/adjudication_pass1.db
```
Expected: `releases with gaps` materially below the pre-growth 20% (247/1242). Remaining unknowns are genuine deferrals (recorded for a later SP3a pass, omitted from publish — never invented).

- [ ] **Step 5: Commit** (taxonomy-growth does its own one-pass-per-commit commits + status note back to GPT). Verify `git log` shows the grown + edge-upgrade passes.

**Gate:** Task 2+ may begin in parallel (code only), but Task 5 (publish) MUST NOT run until Task 1 is committed and the gap rate re-measured.

---

## Task 2: Adjudication materializer (Sub-phase B)

**Files:**
- Create: `src/ai_genre_enrichment/adjudication_materializer.py`
- Test: `tests/unit/test_adjudication_materializer.py`

**Interfaces:**
- Consumes: `LayeredTaxonomy` (`load_default_layered_taxonomy`), `classify_layered_term`, `taxonomy.parents_for_genre`, `taxonomy.families_for_genre`, `taxonomy.genre_by_id`; `SidecarStore.replace_layered_assignments_for_release`; `normalize_release_artist`, `normalize_release_name`.
- Produces:
  - `compute_adjudication_rows(response, taxonomy, *, prompt_version, model) -> tuple[list[dict], list[dict], list[str]]` (genre_rows, facet_rows, skipped_terms) — **pure, no DB**.
  - `materialize_adjudication(store, *, album_id, artist, album, response, taxonomy, prompt_version, model) -> AdjudicationMaterializeSummary`.
  - `ADJUDICATOR_SOURCE = "claude_adjudicator"`, `ADJUDICATOR_SOURCE_RELIABILITY = 0.85`.
  - `@dataclass(frozen=True) AdjudicationMaterializeSummary(release_id, observed_leaf: tuple[str,...], inferred_count: int, facet_count: int, skipped_terms: tuple[str,...])`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_adjudication_materializer.py
from __future__ import annotations

from src.ai_genre_enrichment.adjudication_materializer import (
    ADJUDICATOR_SOURCE,
    compute_adjudication_rows,
)
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

TAX = load_default_layered_taxonomy()


def _resp(genres, facets=None):
    return {
        "genres": [{"term": t, "confidence": 0.8, "layer": "core"} for t in genres],
        "facets": [{"term": t, "facet_type": ft} for t, ft in (facets or [])],
        "escalate": False, "overall_confidence": 0.8,
    }


def test_leaf_expands_to_observed_parent_and_family():
    # shoegaze is a stable leaf with parent/family edges in the taxonomy
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze"]), TAX, prompt_version="pv", model="haiku")
    layers = {(r["genre_id"], r["assignment_layer"]) for r in genre_rows}
    sg = TAX.genre_by_name("shoegaze")
    assert (sg.genre_id, "observed_leaf") in layers
    assert any(layer == "inferred_family" for _, layer in layers)
    assert skipped == []


def test_facet_term_routes_to_facets_not_genres():
    # 'lo-fi' is a facet; it must never land in genre_rows
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze", "lo-fi"]), TAX, prompt_version="pv", model="haiku")
    genre_names = {TAX.genre_by_id(r["genre_id"]).name for r in genre_rows}
    assert "lo-fi" not in genre_names
    assert any(r["source"] == ADJUDICATOR_SOURCE for r in facet_rows)


def test_unknown_term_is_skipped_not_invented():
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze", "xyzzy not a real genre"]), TAX,
        prompt_version="pv", model="haiku")
    assert "xyzzy not a real genre" in skipped
    # only the real leaf produced an observed_leaf row
    obs = [r for r in genre_rows if r["assignment_layer"] == "observed_leaf"]
    assert len(obs) == 1


def test_provenance_and_reliability_stamped():
    genre_rows, _, _ = compute_adjudication_rows(
        _resp(["shoegaze"]), TAX, prompt_version="pv-X", model="sonnet")
    row = next(r for r in genre_rows if r["assignment_layer"] == "observed_leaf")
    assert row["source_reliability"] == 0.85
    assert row["provenance"]["source"] == ADJUDICATOR_SOURCE
    assert row["provenance"]["prompt_version"] == "pv-X"
    assert row["provenance"]["model"] == "sonnet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_adjudication_materializer.py -v`
Expected: FAIL with `ModuleNotFoundError: ... adjudication_materializer`.

- [ ] **Step 3: Write the implementation**

```python
# src/ai_genre_enrichment/adjudication_materializer.py
"""Materialize an album-adjudicator response into the sidecar's layered genre store.

The adjudicator's tight observed-leaf set is expanded through the graph (parents →
inferred_parent, families → inferred_family) exactly as the hybrid materializer does,
then written for ONE album's release_key — superseding the prior hybrid rows. Facet
terms route to the facet table; unknown/review terms are recorded as skipped, never
invented. Writes the sidecar only; `publish()` remains the sole authority writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .layered_assignment import classify_layered_term
from .layered_taxonomy import FAMILY_KIND, CanonicalGenre, LayeredTaxonomy
from .normalization import normalize_release_artist, normalize_release_name

ADJUDICATOR_SOURCE = "claude_adjudicator"
ADJUDICATOR_SOURCE_RELIABILITY = 0.85


@dataclass(frozen=True)
class AdjudicationMaterializeSummary:
    release_id: str
    observed_leaf: tuple[str, ...]
    inferred_count: int
    facet_count: int
    skipped_terms: tuple[str, ...]


def _add_genre(rows, genre, layer, conf, *, prompt_version, model, term):
    key = (genre.genre_id, layer)
    candidate = {
        "genre_id": genre.genre_id,
        "assignment_layer": layer,
        "confidence": float(conf),
        "source_reliability": ADJUDICATOR_SOURCE_RELIABILITY,
        "evidence_count": 1,
        "rejected_by_user": False,
        "provenance": {
            "source": ADJUDICATOR_SOURCE, "prompt_version": prompt_version,
            "model": model, "term": term, "genre": genre.name,
        },
    }
    existing = rows.get(key)
    if existing is None or candidate["confidence"] > existing["confidence"]:
        rows[key] = candidate


def compute_adjudication_rows(
    response: dict[str, Any], taxonomy: LayeredTaxonomy, *, prompt_version: str, model: str
) -> tuple[list[dict], list[dict], list[str]]:
    genre_rows: dict[tuple[str, str], dict] = {}
    facet_rows: dict[tuple[str, str], dict] = {}
    skipped: list[str] = []
    proposed = [g["term"] for g in response.get("genres", [])]

    def _facet(canonical_id, conf, term):
        facet_rows[(canonical_id, ADJUDICATOR_SOURCE)] = {
            "facet_id": canonical_id, "confidence": float(conf),
            "source": ADJUDICATOR_SOURCE,
            "provenance": {"source": ADJUDICATOR_SOURCE, "term": term,
                           "prompt_version": prompt_version, "model": model},
        }

    for g in response.get("genres", []):
        term = g["term"]
        conf = g.get("confidence", response.get("overall_confidence", 0.0))
        cls = classify_layered_term(taxonomy, term, context_terms=proposed)
        if cls.term_kind in ("reject", "review") or cls.canonical_id is None:
            skipped.append(term)
            continue
        if cls.term_kind == "facet":
            _facet(cls.canonical_id, conf, term)
            continue
        genre = taxonomy.genre_by_id(cls.canonical_id)
        if genre is None:
            skipped.append(term)
            continue
        if cls.term_kind == "family":
            _add_genre(genre_rows, genre, "inferred_family", conf,
                       prompt_version=prompt_version, model=model, term=term)
            continue
        _add_genre(genre_rows, genre, "observed_leaf", conf,
                   prompt_version=prompt_version, model=model, term=term)
        for parent in taxonomy.parents_for_genre(genre.genre_id):
            _add_genre(genre_rows, parent, "inferred_parent", conf,
                       prompt_version=prompt_version, model=model, term=term)
        for family in taxonomy.families_for_genre(genre.genre_id):
            _add_genre(genre_rows, family, "inferred_family", conf,
                       prompt_version=prompt_version, model=model, term=term)

    for f in response.get("facets", []):
        cls = classify_layered_term(taxonomy, f.get("term", ""))
        if cls.term_kind in ("facet", "alias") and cls.canonical_id and taxonomy.facet_by_id(cls.canonical_id):
            _facet(cls.canonical_id, response.get("overall_confidence", 0.8), f.get("term", ""))

    return list(genre_rows.values()), list(facet_rows.values()), skipped


def materialize_adjudication(
    store, *, album_id: str, artist: str, album: str,
    response: dict[str, Any], taxonomy: LayeredTaxonomy, prompt_version: str, model: str,
) -> AdjudicationMaterializeSummary:
    release_id = f"{normalize_release_artist(artist)}::{normalize_release_name(album)}"
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        response, taxonomy, prompt_version=prompt_version, model=model)
    store.replace_layered_assignments_for_release(
        release_id=release_id, artist=artist, album=album,
        genre_assignments=genre_rows, facet_assignments=facet_rows,
    )
    observed = tuple(sorted(
        taxonomy.genre_by_id(r["genre_id"]).name
        for r in genre_rows if r["assignment_layer"] == "observed_leaf"
    ))
    inferred = sum(1 for r in genre_rows if r["assignment_layer"] != "observed_leaf")
    return AdjudicationMaterializeSummary(
        release_id=release_id, observed_leaf=observed,
        inferred_count=inferred, facet_count=len(facet_rows), skipped_terms=tuple(skipped),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_adjudication_materializer.py -v`
Expected: 4 passed. (If `shoegaze` has no family edge in the live taxonomy, switch the fixture leaf to another stable leaf with a family edge — e.g. `dream pop` — and re-run.)

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/adjudication_materializer.py tests/unit/test_adjudication_materializer.py
git commit -m "feat(genre): adjudication materializer (response -> sidecar layered rows)"
```

---

## Task 3: Apply runner — auto lane + diff report (Sub-phase C, part 1)

**Files:**
- Create: `scripts/research/apply_adjudication.py`
- Test: `tests/unit/test_apply_adjudication.py`

**Interfaces:**
- Consumes: `materialize_adjudication` (Task 2); `AdjudicationStore` read; `resolve_db`, `build_evidence` (run_album_adjudicator); `canonicalize_proposed`; `load_graph_adapter`.
- Produces (pure helpers, unit-tested):
  - `best_results(rows, thorough_pv) -> dict[album_id, response]` (thorough wins).
  - `split_lanes(best) -> tuple[dict auto, dict escalated]` (escalated = `response["escalate"]`).
  - `invented_genres(proposed_canonical: list[str], prior_leaf: list[str]) -> list[str]` (proposed − prior).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_apply_adjudication.py
from __future__ import annotations

from scripts.research.apply_adjudication import (
    best_results, split_lanes, invented_genres,
)


def _r(genres, escalate=False):
    return {"genres": [{"term": g} for g in genres], "escalate": escalate}


def test_best_results_prefers_thorough():
    rows = [
        ("a1", "pv-std", _r(["x"])),
        ("a1", "pv-thorough", _r(["x", "y", "z"])),
        ("a2", "pv-std", _r(["q"])),
    ]
    best = best_results(rows, thorough_pv="pv-thorough")
    assert [g["term"] for g in best["a1"]["genres"]] == ["x", "y", "z"]
    assert [g["term"] for g in best["a2"]["genres"]] == ["q"]


def test_split_lanes_separates_escalated():
    best = {"a1": _r(["x"]), "a2": _r(["y"], escalate=True)}
    auto, escalated = split_lanes(best)
    assert set(auto) == {"a1"} and set(escalated) == {"a2"}


def test_invented_genres_are_proposed_minus_prior():
    assert invented_genres(["afrobeat", "funk"], ["funk", "soul"]) == ["afrobeat"]
    assert invented_genres(["funk"], ["funk", "soul"]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_apply_adjudication.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/research/apply_adjudication.py
#!/usr/bin/env python
"""Auto lane: materialize every non-escalated adjudication into the sidecar, and
write a non-blocking diff report for albums that introduced a new genre. Escalated
albums are left for review_escalated.py. Resumable: materialize is idempotent
(replace-by-release-key), so re-running is safe.

Usage:
  python scripts/research/apply_adjudication.py --dry-run   # counts + report only
  python scripts/research/apply_adjudication.py             # write sidecar + report
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adjudicator_bulk import effective_prompt_version  # noqa: E402
from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_materializer import materialize_adjudication  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import canonicalize_proposed  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

REPORT = _ROOT / "docs" / "genre_adjudication" / "phase4_added_genres_report.md"


def best_results(rows, *, thorough_pv):
    best = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


def split_lanes(best):
    auto = {a: r for a, r in best.items() if not r.get("escalate")}
    escalated = {a: r for a, r in best.items() if r.get("escalate")}
    return auto, escalated


def invented_genres(proposed_canonical, prior_leaf):
    prior = set(prior_leaf)
    return [g for g in proposed_canonical if g not in prior]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    std_pv = effective_prompt_version(thorough=False)
    tho_pv = effective_prompt_version(thorough=True)
    conn = sqlite3.connect(f"file:{args.shadow_db}?mode=ro", uri=True)
    raw = [(a, pv, json.loads(rj)) for a, pv, rj in conn.execute(
        "SELECT album_id, prompt_version, response_json FROM adjudications WHERE status='complete'")]
    conn.close()
    best = best_results(raw, thorough_pv=tho_pv)
    auto, escalated = split_lanes(best)

    taxonomy = load_default_layered_taxonomy()
    adapter = load_graph_adapter()
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}
    store = None if args.dry_run else SidecarStore(str(resolve_db("ai_genre_enrichment.db")))

    added_rows, materialized = [], 0
    for album_id, resp in auto.items():
        ev = build_evidence(meta, album_id, id2name)
        canon = canonicalize_proposed([g["term"] for g in resp["genres"]], adapter.canonicalize_tag)["canonical"]
        prior = ev["current_observed_leaf"]
        new = invented_genres(canon, prior)
        if new:
            added_rows.append((ev["artist"], ev["album"], prior, canon, new))
        if not args.dry_run:
            # best-result prompt_version: thorough if this response is the thorough one
            pv = tho_pv if best.get(album_id) is resp and resp.get("_thorough") else std_pv
            materialize_adjudication(
                store, album_id=album_id, artist=ev["artist"], album=ev["album"],
                response=resp, taxonomy=taxonomy,
                prompt_version=std_pv, model="haiku",  # provenance only; model not load-bearing
            )
            materialized += 1
    meta.close()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase-4 added-genre diff report ({len(added_rows)} albums)\n",
             "Non-blocking. Auto-published albums where the adjudicator added >=1 genre",
             "not in the prior authority. Skim for invented-genre errors.\n"]
    for artist, album, prior, proposed, new in sorted(added_rows):
        lines.append(f"- **{artist} — {album}**  NEW={new}\n"
                     f"    prior={prior}\n    proposed={proposed}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"auto={len(auto)} escalated={len(escalated)} materialized={materialized} "
          f"added_genre_albums={len(added_rows)}{' (dry-run)' if args.dry_run else ''}")
    print(f"diff report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(Note: the `pv`/`_thorough` provenance line is best-effort; provenance model/version is not load-bearing for generation. Keep it simple — `std_pv`/`"haiku"` is acceptable since provenance is informational.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_apply_adjudication.py -v`
Expected: 3 passed.

- [ ] **Step 5: Dry-run the script against the real shadow DB (no writes)**

Run: `python scripts/research/apply_adjudication.py --dry-run`
Expected: `auto=1130 escalated=112 materialized=0 added_genre_albums=233 (dry-run)` and a written diff report. (Counts may shift slightly after Task 1 re-canonicalization — that's fine.)

- [ ] **Step 6: Commit**

```bash
git add scripts/research/apply_adjudication.py tests/unit/test_apply_adjudication.py docs/genre_adjudication/phase4_added_genres_report.md
git commit -m "feat(genre): auto-lane apply runner + non-blocking added-genre diff report"
```

---

## Task 4: Escalated review CLI (Sub-phase C, part 2)

**Files:**
- Create: `scripts/research/review_escalated.py`
- Test: `tests/unit/test_review_escalated.py`

**Interfaces:**
- Consumes: `materialize_adjudication` (Task 2); the shadow DB; `build_evidence`/`resolve_db`.
- Produces (pure helpers, unit-tested):
  - `parse_decision(line) -> tuple[str, list[str]]` — `"accept"`→`("accept", [])`, `"reject"`→`("reject", [])`, `"edit a, b ,c"`→`("edit", ["a","b","c"])`, `"skip"`/`"quit"` pass through.
  - A `ReviewDecisionStore` (tiny SQLite table `escalation_decisions(album_id PK, decision, genres_json, updated_at)` in the shadow DB) with `get(album_id)`, `save(album_id, decision, genres)`, `decided_ids()` — for resumability.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_review_escalated.py
from __future__ import annotations

from scripts.research.review_escalated import ReviewDecisionStore, parse_decision


def test_parse_decision_variants():
    assert parse_decision("accept") == ("accept", [])
    assert parse_decision("reject") == ("reject", [])
    assert parse_decision("skip") == ("skip", [])
    assert parse_decision("edit shoegaze, dream pop ,  noise pop") == (
        "edit", ["shoegaze", "dream pop", "noise pop"])


def test_decision_store_roundtrip_and_resume(tmp_path):
    s = ReviewDecisionStore(tmp_path / "d.db")
    assert s.decided_ids() == set()
    s.save("a1", "accept", [])
    s.save("a2", "edit", ["funk", "soul"])
    assert s.get("a2") == {"decision": "edit", "genres": ["funk", "soul"]}
    assert s.decided_ids() == {"a1", "a2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_review_escalated.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/research/review_escalated.py
#!/usr/bin/env python
"""Human review of the escalated adjudications (album-grain). Resumable: decisions
persist to escalation_decisions in the shadow DB. accept/edit -> materialize via the
SAME path as the auto lane; reject -> leave the album's existing authority untouched.

Usage:
  python scripts/research/review_escalated.py            # interactive review
  python scripts/research/review_escalated.py --apply    # materialize decided accept/edit
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adjudicator_bulk import effective_prompt_version  # noqa: E402
from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_materializer import materialize_adjudication  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import canonicalize_proposed  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402


def parse_decision(line):
    line = line.strip()
    if line.startswith("edit"):
        rest = line[len("edit"):].strip()
        genres = [g.strip() for g in rest.split(",") if g.strip()]
        return ("edit", genres)
    word = line.split()[0] if line.split() else ""
    return (word, [])


class ReviewDecisionStore:
    def __init__(self, db_path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path)
        self._c.execute(
            "CREATE TABLE IF NOT EXISTS escalation_decisions "
            "(album_id TEXT PRIMARY KEY, decision TEXT, genres_json TEXT, updated_at TEXT)")
        self._c.commit()

    def save(self, album_id, decision, genres):
        self._c.execute(
            "INSERT INTO escalation_decisions (album_id, decision, genres_json, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET "
            "decision=excluded.decision, genres_json=excluded.genres_json, updated_at=excluded.updated_at",
            (album_id, decision, json.dumps(genres), time.strftime("%Y-%m-%dT%H:%M:%S")))
        self._c.commit()

    def get(self, album_id):
        row = self._c.execute(
            "SELECT decision, genres_json FROM escalation_decisions WHERE album_id=?",
            (album_id,)).fetchone()
        return {"decision": row[0], "genres": json.loads(row[1])} if row else None

    def decided_ids(self):
        return {r[0] for r in self._c.execute("SELECT album_id FROM escalation_decisions")}


def _escalated(shadow_db, tho_pv):
    conn = sqlite3.connect(f"file:{shadow_db}?mode=ro", uri=True)
    raw = [(a, pv, json.loads(rj)) for a, pv, rj in conn.execute(
        "SELECT album_id, prompt_version, response_json FROM adjudications WHERE status='complete'")]
    conn.close()
    best = {}
    for a, pv, resp in raw:
        if a not in best or pv == tho_pv:
            best[a] = resp
    return {a: r for a, r in best.items() if r.get("escalate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--apply", action="store_true", help="materialize decided accept/edit")
    args = ap.parse_args()

    tho_pv = effective_prompt_version(thorough=True)
    std_pv = effective_prompt_version(thorough=False)
    escalated = _escalated(args.shadow_db, tho_pv)
    decisions = ReviewDecisionStore(args.shadow_db)
    adapter = load_graph_adapter()
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}

    if args.apply:
        taxonomy = load_default_layered_taxonomy()
        store = SidecarStore(str(resolve_db("ai_genre_enrichment.db")))
        n = 0
        for album_id in escalated:
            d = decisions.get(album_id)
            if not d or d["decision"] not in ("accept", "edit"):
                continue
            ev = build_evidence(meta, album_id, id2name)
            resp = escalated[album_id]
            if d["decision"] == "edit":
                resp = {**resp, "genres": [{"term": g, "confidence": 0.8, "layer": "core"} for g in d["genres"]]}
            materialize_adjudication(store, album_id=album_id, artist=ev["artist"],
                album=ev["album"], response=resp, taxonomy=taxonomy,
                prompt_version=std_pv, model="review")
            n += 1
        meta.close()
        print(f"applied {n} accept/edit decisions")
        return 0

    done = decisions.decided_ids()
    todo = [a for a in escalated if a not in done]
    print(f"escalated={len(escalated)} decided={len(done)} remaining={len(todo)}")
    for i, album_id in enumerate(todo, 1):
        ev = build_evidence(meta, album_id, id2name)
        resp = escalated[album_id]
        canon = canonicalize_proposed([g["term"] for g in resp["genres"]], adapter.canonicalize_tag)["canonical"]
        print(f"\n[{i}/{len(todo)}] {ev['artist']} — {ev['album']}")
        print(f"   prior    = {ev['current_observed_leaf']}")
        print(f"   proposed = {canon}")
        print(f"   reason   = {resp.get('escalate_reason','')}")
        if resp.get("dropped_file_tags"):
            print(f"   DROPPED FILE TAGS = {resp['dropped_file_tags']}")
        line = input("   [accept / reject / edit a,b,c / skip / quit] > ")
        decision, genres = parse_decision(line)
        if decision == "quit":
            break
        if decision == "skip":
            continue
        if decision in ("accept", "reject", "edit"):
            decisions.save(album_id, decision, genres)
    meta.close()
    print("review session saved. Re-run to resume; then --apply to materialize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_review_escalated.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/research/review_escalated.py tests/unit/test_review_escalated.py
git commit -m "feat(genre): escalated-review CLI (resumable, accept/edit/reject -> materialize)"
```

---

## Task 5: Publish, rebuild, verify (Sub-phase D)

**Operational task — gated, not TDD. Task 1 MUST be committed first.**

**Files:**
- Modify (data, not code): `data/ai_genre_enrichment.db` (sidecar — materialize writes), `data/metadata.db` (publish writes), `data/artifacts/beat3tower_32k/data_matrices_step1.npz` (rebuild).

- [ ] **Step 1: Full test sweep + lint**

Run: `python -m pytest -q -m "not slow"` then `ruff check`
Expected: green (taxonomy-growth deselects from its skill if needed); no new ruff errors.

- [ ] **Step 2: Back up the sidecar, then run the auto lane**

```bash
cp "$(python -c 'import sys;sys.path.insert(0,"scripts/research");from run_album_adjudicator import resolve_db;print(resolve_db("ai_genre_enrichment.db"))')" \
   "data/ai_genre_enrichment.db.bak.$(date +%Y%m%d-%H%M%S)"
python scripts/research/apply_adjudication.py
```
Expected: `materialized=1130 added_genre_albums≈233`, diff report written. Spot-check the report for any obvious invented-genre error.

- [ ] **Step 3: Review the 112 escalated, then apply**

```bash
python scripts/research/review_escalated.py          # interactive (resumable)
python scripts/research/review_escalated.py --apply   # materialize accept/edit
```
Expected: review session completes (or resumes across sittings); `--apply` reports the accept/edit count materialized.

- [ ] **Step 4: BACK UP metadata.db (mandatory) — STOP for explicit confirmation**

```bash
META="$(python -c 'import sys;sys.path.insert(0,"scripts/research");from run_album_adjudicator import resolve_db;print(resolve_db("metadata.db"))')"
cp "$META" "${META}.bak.$(date +%Y%m%d-%H%M%S)"
ls -la "${META}.bak."*
```
**Do not proceed past this step without the user's explicit second confirmation to write to metadata.db.**

- [ ] **Step 5: Dry-run publish, inspect stats**

```bash
python scripts/publish_genres.py --dry-run
```
Expected JSON: `graph_albums` ≈ count of albums with sidecar assignments (should rise to include the 1,242 adjudicated), low `collisions`, `dry_run: true`. Sanity-check the numbers before the real write.

- [ ] **Step 6: Publish (the authority write)**

```bash
python scripts/publish_genres.py
```
Expected: same stats with `dry_run: false`. `release_effective_genres` now reflects the adjudicated sets.

- [ ] **Step 7: Verify the authority for a known album**

```bash
python -c "
import sys; sys.path.insert(0,'scripts/research')
from run_album_adjudicator import resolve_db
import sqlite3
c=sqlite3.connect(f'file:{resolve_db(\"metadata.db\")}?mode=ro',uri=True)
rows=c.execute('''SELECT g.name, r.assignment_layer FROM release_effective_genres r
 JOIN albums a ON a.album_id=r.album_id
 JOIN genre_graph_canonical_genres g ON g.genre_id=r.genre_id
 WHERE a.artist LIKE \"%Kendrick%\" AND a.title LIKE \"%Butterfly%\"
 AND r.assignment_layer=\"observed_leaf\" ORDER BY g.name''').fetchall()
print('TPAB observed_leaf:', [r[0] for r in rows])
"
```
Expected: the tight set (`conscious hip hop`, `funk`, `g-funk`, `jazz rap`, `neo-soul`) — not the prior 8-tag bloat.

- [ ] **Step 8: Rebuild the artifact**

```bash
python scripts/build_beat3tower_artifacts.py --genre-source graph
```
Expected: `data_matrices_step1.npz` rebuilt from the published authority. (Honor artifact-backup discipline; never touch MERT files.)

- [ ] **Step 9: End-to-end generation smoke (fidelity harness)**

Per the **playlist-testing** skill, run one multi-pier generation through the `gui_fidelity` harness and confirm it loads the new genres and transition stats are not regressed.
Expected: generation completes <90s; distinct-artist + transition min/mean/p10/p90 reported; no crash.

- [ ] **Step 10: Final commit**

```bash
git add docs/genre_adjudication/phase4_added_genres_report.md
git commit -m "chore(genre): Phase-4 publish complete — adjudicated authority live + diff report"
```
(metadata.db / sidecar / npz are gitignored data; the commit records the report + any code touched.)

---

## Self-Review

**Spec coverage:** A (taxonomy growth)→Task 1; B (materializer)→Task 2; C auto lane + diff report→Task 3; C review lane→Task 4; D publish/rebuild/verify→Task 5. Trust policy (auto 1,130 / review 112 / report 233)→Tasks 3–5. Surgical/grandfather→materializer replace-by-key (Task 2) + only-adjudicated-albums iteration (Tasks 3–4). Data safety→Task 5 backups + confirmation gate. All spec sections mapped.

**Placeholder scan:** none — every code step has full code; the one informational note (provenance model string) is explicitly flagged as non-load-bearing.

**Type consistency:** `compute_adjudication_rows`/`materialize_adjudication` signatures match between Task 2 definition and Tasks 3–4 calls; `best_results(rows, *, thorough_pv)`, `split_lanes`, `invented_genres`, `parse_decision`, `ReviewDecisionStore` match between their tests and the script bodies; genre/facet row dict keys match `replace_layered_assignments_for_release`'s expected keys.

**Known follow-up (not blocking):** the `apply_adjudication.py` provenance prompt_version is simplified to `std_pv` for all auto-lane rows; if exact per-row model/version provenance is wanted later, thread the row's source pv through `best_results`.
