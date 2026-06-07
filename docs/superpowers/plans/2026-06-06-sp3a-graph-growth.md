# SP3a — Graph Growth Pre-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the layered genre taxonomy to cover the library's real collected-tag vocabulary — AI proposes full placements for genuinely-unmapped genres, the user approves them in an editable YAML, and an ingest command validates and appends them to `data/layered_genre_taxonomy.yaml`.

**Architecture:** A new pure-logic module `src/ai_genre_enrichment/graph_growth.py` (candidate gathering, alias-collapse, AI placement proposal, proposal-file I/O, structural validation, taxonomy append) backed by one new read-only `SidecarStore` query, wired to two CLI subcommands (`graph-propose-growth`, `graph-ingest-growth`). Growth is additive to a git-versioned YAML; the loader's own `_validate_taxonomy` is the final backstop on re-import.

**Tech Stack:** Python 3.11, stdlib `sqlite3`, `yaml`, pytest. Reuses `src/ai_genre_enrichment/layered_taxonomy.py` (`load_layered_taxonomy`, `_record_id`, `normalize_taxonomy_name`, `LayeredTaxonomy`), `layered_assignment.classify_layered_term`, `storage.SidecarStore`, `client.OpenAIEnrichmentClient`, `routing.WebMode`.

Spec: `docs/superpowers/specs/2026-06-06-sp3a-graph-growth-design.md`.

---

## File Structure

- **Create** `src/ai_genre_enrichment/graph_growth.py` — all growth logic: dataclasses (`GrowthCandidate`, `GrowthProposal`), `gather_growth_candidates`, `collapse_variants`, `growth_proposal_response_format`, `_build_taxonomy_context`, `propose_placement`, `write_proposals`/`read_proposals`, `validate_proposal`, `append_approved_to_taxonomy`.
- **Modify** `src/ai_genre_enrichment/storage.py` — add `SidecarStore.all_collected_tags()`.
- **Modify** `scripts/ai_genre_enrich.py` — add `graph-propose-growth` and `graph-ingest-growth` subcommands.
- **Create** `tests/unit/test_graph_growth.py` — unit tests for every logic unit.
- **Create** `scripts/growth_candidate_report.py` — read-only real-data smoke (Task 10).

---

## Task 1: `SidecarStore.all_collected_tags()`

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_growth.py
from src.ai_genre_enrichment.storage import SidecarStore


def _page_with_tags(store, release_key, artist, album, source_type, tags):
    page_id = store.upsert_source_page(
        release_key=release_key, normalized_artist=artist, normalized_album=album,
        album_id=None, source_url=f"{source_type}://{release_key}/{album}",
        source_type=source_type, identity_status="confirmed",
        identity_confidence=0.9, evidence_summary="t",
    )
    store.replace_source_tags(page_id, tags)


def test_all_collected_tags_returns_release_scoped_rows(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _page_with_tags(store, "acetone::york blvd", "acetone", "york blvd",
                    "lastfm_tags", ["slowcore", "indie rock"])
    rows = store.all_collected_tags()
    got = {(r["release_key"], r["normalized_tag"]) for r in rows}
    assert ("acetone::york blvd", "slowcore") in got
    assert ("acetone::york blvd", "indie rock") in got
    # carries artist/album for example strings
    sample = next(r for r in rows if r["normalized_tag"] == "slowcore")
    assert sample["normalized_artist"] == "acetone"
    assert sample["normalized_album"] == "york blvd"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py::test_all_collected_tags_returns_release_scoped_rows -v`
Expected: FAIL — `AttributeError: 'SidecarStore' object has no attribute 'all_collected_tags'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/ai_genre_enrichment/storage.py` (next to `release_keys_with_source_type`):

```python
    def all_collected_tags(self) -> list[sqlite3.Row]:
        """Every collected source tag joined to its release, for growth analysis.

        Returns rows with: release_key, normalized_artist, normalized_album,
        normalized_tag. One row per (page, tag); callers aggregate.
        """
        with self.connect() as conn:
            return list(conn.execute(
                """
                SELECT p.release_key, p.normalized_artist, p.normalized_album,
                       t.normalized_tag
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                WHERE t.normalized_tag IS NOT NULL AND t.normalized_tag != ''
                """
            ).fetchall())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py::test_all_collected_tags_returns_release_scoped_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): SidecarStore.all_collected_tags for growth analysis (SP3a task 1)"
```

---

## Task 2: `GrowthCandidate` + `gather_growth_candidates`

Aggregate collected tags to distinct terms, keep only genuinely-unmapped genres (`classify_layered_term` → `term_kind == "review"` and `canonical_id is None`) at/above the album-frequency threshold, with evidence.

**Files:**
- Create: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py
from src.ai_genre_enrichment import graph_growth
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


def test_gather_candidates_keeps_unmapped_genres_above_threshold(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    # "vaporwave" is (assume) unmapped; appears on 3 releases -> candidate.
    for i in range(3):
        _page_with_tags(store, f"a{i}::alb{i}", f"a{i}", f"alb{i}",
                        "lastfm_tags", ["vaporwave", "ambient"])
    # "rock" is a known family -> dropped. "vaporwave" on 3 albums -> kept.
    cands = graph_growth.gather_growth_candidates(store, taxonomy, min_album_freq=3)
    terms = {c.term for c in cands}
    assert "vaporwave" in terms
    vw = next(c for c in cands if c.term == "vaporwave")
    assert vw.album_frequency == 3
    assert "ambient" in vw.cooccurring_tags          # co-occurring evidence
    assert len(vw.examples) >= 1                       # example "artist — album"


def test_gather_candidates_drops_below_threshold_and_mapped(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    _page_with_tags(store, "x::y", "x", "y", "lastfm_tags", ["vaporwave", "rock"])
    cands = graph_growth.gather_growth_candidates(store, taxonomy, min_album_freq=3)
    terms = {c.term for c in cands}
    assert "vaporwave" not in terms   # only 1 album < 3
    assert "rock" not in terms        # mapped family, never a candidate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k gather_candidates -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.graph_growth'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_genre_enrichment/graph_growth.py
"""Graph growth pre-pass: propose + ingest new genres into the layered taxonomy.

See docs/superpowers/specs/2026-06-06-sp3a-graph-growth-design.md.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .layered_assignment import classify_layered_term
from .layered_taxonomy import LayeredTaxonomy


@dataclass
class GrowthCandidate:
    term: str
    album_frequency: int
    cooccurring_tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)


def gather_growth_candidates(
    store,
    taxonomy: LayeredTaxonomy,
    *,
    min_album_freq: int = 3,
    max_examples: int = 3,
    max_cooccurring: int = 8,
) -> list[GrowthCandidate]:
    """Distinct genuinely-unmapped genres at/above the album-frequency threshold.

    A candidate is a tag whose classify_layered_term yields term_kind == 'review'
    AND canonical_id is None (genuinely unknown). Mapped genres, aliases, facets,
    rejects, and review-but-known terms are excluded. Ranked by album_frequency.
    """
    rows = store.all_collected_tags()

    # release -> set(tags); tag -> set(releases); tag -> example releases
    tags_by_release: dict[str, set[str]] = defaultdict(set)
    releases_by_tag: dict[str, set[str]] = defaultdict(set)
    example_by_tag: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        tag = r["normalized_tag"]
        rk = r["release_key"]
        tags_by_release[rk].add(tag)
        releases_by_tag[tag].add(rk)
        if len(example_by_tag[tag]) < max_examples:
            label = f"{r['normalized_artist']} — {r['normalized_album']}"
            if label not in example_by_tag[tag]:
                example_by_tag[tag].append(label)

    candidates: list[GrowthCandidate] = []
    for tag, releases in releases_by_tag.items():
        freq = len(releases)
        if freq < min_album_freq:
            continue
        classification = classify_layered_term(taxonomy, tag)
        if classification.term_kind != "review" or classification.canonical_id is not None:
            continue  # mapped / alias / facet / reject / known-review -> not a candidate

        cooccur: Counter[str] = Counter()
        for rk in releases:
            for other in tags_by_release[rk]:
                if other != tag:
                    cooccur[other] += 1
        candidates.append(GrowthCandidate(
            term=tag,
            album_frequency=freq,
            cooccurring_tags=[t for t, _ in cooccur.most_common(max_cooccurring)],
            examples=list(example_by_tag[tag]),
        ))

    candidates.sort(key=lambda c: (-c.album_frequency, c.term))
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k gather_candidates -v`
Expected: PASS (both). If a future taxonomy version maps `vaporwave`, pick any term not in `data/layered_genre_taxonomy.yaml` for the fixture.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): gather unmapped-genre growth candidates (SP3a task 2)"
```

---

## Task 3: `collapse_variants` (alias-collapse)

Merge near-duplicate candidate terms (shared normalized token-set, or one is the other with spaces removed) into a single representative, recording the rest as `variants`.

**Files:**
- Modify: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py

def test_collapse_variants_merges_spacing_variants():
    cands = [
        graph_growth.GrowthCandidate(term="synthwave", album_frequency=10),
        graph_growth.GrowthCandidate(term="synth wave", album_frequency=4),
        graph_growth.GrowthCandidate(term="vaporwave", album_frequency=6),
    ]
    merged = graph_growth.collapse_variants(cands)
    by_term = {c.term: c for c in merged}
    # "synthwave"/"synth wave" collapse to the higher-frequency representative
    assert "synthwave" in by_term
    assert "synth wave" not in by_term
    assert "synth wave" in by_term["synthwave"].variants
    # frequencies combine; vaporwave untouched
    assert by_term["synthwave"].album_frequency == 14
    assert "vaporwave" in by_term
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k collapse_variants -v`
Expected: FAIL — no attribute `collapse_variants`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ai_genre_enrichment/graph_growth.py

def _variant_key(term: str) -> str:
    """Spacing/separator-insensitive key: 'synth wave' and 'synthwave' match."""
    return "".join(ch for ch in term.casefold() if ch.isalnum())


def collapse_variants(candidates: list[GrowthCandidate]) -> list[GrowthCandidate]:
    """Merge spacing/separator variants into the highest-frequency representative.

    The representative keeps its term + the union of examples/co-occurring tags;
    the merged-away spellings are recorded in `variants` (alias suggestions).
    Combined album_frequency is summed (upper bound; exact de-dup of overlapping
    releases is unnecessary for ranking).
    """
    groups: dict[str, list[GrowthCandidate]] = defaultdict(list)
    for cand in candidates:
        groups[_variant_key(cand.term)].append(cand)

    merged: list[GrowthCandidate] = []
    for members in groups.values():
        members.sort(key=lambda c: (-c.album_frequency, c.term))
        rep = members[0]
        variants = [m.term for m in members[1:]]
        cooccur: list[str] = list(rep.cooccurring_tags)
        examples: list[str] = list(rep.examples)
        for m in members[1:]:
            for t in m.cooccurring_tags:
                if t not in cooccur:
                    cooccur.append(t)
            for e in m.examples:
                if e not in examples:
                    examples.append(e)
        merged.append(GrowthCandidate(
            term=rep.term,
            album_frequency=sum(m.album_frequency for m in members),
            cooccurring_tags=cooccur,
            examples=examples,
            variants=variants,
        ))
    merged.sort(key=lambda c: (-c.album_frequency, c.term))
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k collapse_variants -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): collapse spacing/separator variants (SP3a task 3)"
```

---

## Task 4: `GrowthProposal` + schema + `propose_placement`

AI proposes a full placement for one candidate. The OpenAI call is injected (a client exposing `_call_openai`) so tests mock it.

**Files:**
- Modify: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py
import json


class _FakeResp:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def _call_openai(self, prompt, response_format, *, instructions):
        self.calls.append(prompt)
        return _FakeResp(self._payload)


def test_propose_placement_returns_structured_proposal():
    taxonomy = load_default_layered_taxonomy()
    cand = graph_growth.GrowthCandidate(
        term="vaporwave", album_frequency=14,
        cooccurring_tags=["chillwave", "ambient"], variants=["vapor wave"],
    )
    payload = {
        "name": "vaporwave", "kind": "subgenre", "status": "active",
        "specificity_score": 0.8,
        "parent_edges": [{"target": "electronic", "edge_type": "family_context",
                          "weight": 0.55, "confidence": 0.8}],
        "similar_to": ["ambient"],
        "alias_variants": ["vapor wave"],
        "term_kind_confirm": "genre",
        "rationale": "Plunderphonic electronic microgenre.",
    }
    client = _FakeClient(payload)
    proposal = graph_growth.propose_placement(cand, taxonomy, client=client)
    assert proposal.name == "vaporwave"
    assert proposal.kind == "subgenre"
    assert proposal.parent_edges[0]["target"] == "electronic"
    assert proposal.term_kind_confirm == "genre"
    # the candidate's evidence reached the model
    assert "vaporwave" in client.calls[0]
    assert "chillwave" in client.calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k propose_placement -v`
Expected: FAIL — no attribute `propose_placement`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ai_genre_enrichment/graph_growth.py
import json
from copy import deepcopy
from typing import Any

from .layered_taxonomy import FAMILY_KIND, normalize_taxonomy_name
from .routing import WebMode

GROWTH_PROPOSAL_INSTRUCTIONS = """
You place a new music genre into an existing hierarchical genre taxonomy.
Given a candidate genre term, its evidence (how often it appears and which
genres co-occur with it), and the relevant existing taxonomy names, propose
where it belongs.

Rules:
- Only propose a placement if the term is a real GENRE/subgenre. If it is a
  descriptor/facet (mood, instrument, era, region, format) set term_kind_confirm
  to "facet"; if it is noise/non-music set it to "noise".
- parent_edges must reference EXISTING taxonomy names exactly as given. Choose
  1-2 parents (a family via "family_context", or a broader genre via "is_a").
- specificity_score: ~0.05 for broad families, ~0.5 for mid genres, ~0.8-0.9 for
  narrow microgenres.
- Do not invent edges to names not in the provided context.
""".strip()


_GROWTH_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "kind", "status", "specificity_score", "parent_edges",
                 "similar_to", "alias_variants", "term_kind_confirm", "rationale"],
    "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": ["genre", "subgenre"]},
        "status": {"type": "string", "enum": ["active", "review"]},
        "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "parent_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "edge_type", "weight", "confidence"],
                "properties": {
                    "target": {"type": "string"},
                    "edge_type": {"type": "string",
                                  "enum": ["is_a", "family_context"]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "similar_to": {"type": "array", "items": {"type": "string"}},
        "alias_variants": {"type": "array", "items": {"type": "string"}},
        "term_kind_confirm": {"type": "string",
                              "enum": ["genre", "facet", "noise"]},
        "rationale": {"type": "string"},
    },
}


def growth_proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "graph_growth_proposal",
        "schema": deepcopy(_GROWTH_PROPOSAL_SCHEMA),
        "strict": True,
    }


@dataclass
class GrowthProposal:
    name: str
    kind: str
    status: str
    specificity_score: float
    parent_edges: list[dict] = field(default_factory=list)
    similar_to: list[str] = field(default_factory=list)
    alias_variants: list[str] = field(default_factory=list)
    term_kind_confirm: str = "genre"
    rationale: str = ""


def _build_taxonomy_context(taxonomy: LayeredTaxonomy, candidate: GrowthCandidate) -> list[str]:
    """Bounded context: all families/umbrellas + genres sharing a token with the
    candidate or its co-occurring tags. Keeps the prompt small but relevant."""
    tokens = set(normalize_taxonomy_name(candidate.term).split())
    for t in candidate.cooccurring_tags:
        tokens.update(normalize_taxonomy_name(t).split())
    names: list[str] = []
    for genre in taxonomy.genres:
        if genre.kind in {FAMILY_KIND, "umbrella"}:
            names.append(genre.name)
        elif tokens & set(normalize_taxonomy_name(genre.name).split()):
            names.append(genre.name)
    return sorted(dict.fromkeys(names))


def propose_placement(
    candidate: GrowthCandidate,
    taxonomy: LayeredTaxonomy,
    *,
    client,
    web_mode: WebMode | str = WebMode.OFF,
) -> GrowthProposal:
    """Ask the model to place one candidate. `client` exposes `_call_openai`."""
    from .client import _extract_response_json

    context_names = _build_taxonomy_context(taxonomy, candidate)
    prompt = json.dumps({
        "candidate_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": candidate.cooccurring_tags,
        "spelling_variants": candidate.variants,
        "examples": candidate.examples,
        "existing_taxonomy_names": context_names,
    }, ensure_ascii=False, sort_keys=True)
    raw = client._call_openai(
        prompt, growth_proposal_response_format(),
        instructions=GROWTH_PROPOSAL_INSTRUCTIONS,
    )
    data = _extract_response_json(raw)
    return GrowthProposal(
        name=str(data["name"]),
        kind=str(data["kind"]),
        status=str(data.get("status") or "active"),
        specificity_score=float(data["specificity_score"]),
        parent_edges=list(data.get("parent_edges") or []),
        similar_to=list(data.get("similar_to") or []),
        alias_variants=list(data.get("alias_variants") or candidate.variants),
        term_kind_confirm=str(data.get("term_kind_confirm") or "genre"),
        rationale=str(data.get("rationale") or ""),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k propose_placement -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): AI placement proposal + strict schema (SP3a task 4)"
```

---

## Task 5: Proposal YAML I/O (`write_proposals` / `read_proposals`)

**Files:**
- Modify: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py

def test_proposal_file_round_trip(tmp_path):
    cand = graph_growth.GrowthCandidate(
        term="vaporwave", album_frequency=14,
        cooccurring_tags=["chillwave"], examples=["a — b"], variants=["vapor wave"],
    )
    proposal = graph_growth.GrowthProposal(
        name="vaporwave", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=["ambient"], alias_variants=["vapor wave"],
        term_kind_confirm="genre", rationale="x",
    )
    path = tmp_path / "proposals.yaml"
    graph_growth.write_proposals(path, [(cand, proposal)])
    entries = graph_growth.read_proposals(path)
    assert len(entries) == 1
    e = entries[0]
    assert e.term == "vaporwave"
    assert e.decision == "pending"           # default decision
    assert e.proposal.name == "vaporwave"
    assert e.proposal.parent_edges[0]["target"] == "electronic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k proposal_file_round_trip -v`
Expected: FAIL — no attribute `write_proposals`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ai_genre_enrichment/graph_growth.py
from dataclasses import asdict
from pathlib import Path

import yaml


@dataclass
class ProposalEntry:
    term: str
    album_frequency: int
    cooccurring_tags: list[str]
    examples: list[str]
    decision: str
    proposal: GrowthProposal


def write_proposals(path, items: list[tuple[GrowthCandidate, GrowthProposal]]) -> None:
    """Write candidate+proposal pairs to an editable review YAML."""
    entries = []
    for cand, proposal in items:
        entries.append({
            "term": cand.term,
            "album_frequency": cand.album_frequency,
            "cooccurring_tags": list(cand.cooccurring_tags),
            "examples": list(cand.examples),
            "decision": "pending",
            "proposal": asdict(proposal),
        })
    Path(path).write_text(
        yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def read_proposals(path) -> list[ProposalEntry]:
    """Read a (possibly user-edited) proposal YAML back into entries."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    entries: list[ProposalEntry] = []
    for row in raw:
        p = row.get("proposal") or {}
        entries.append(ProposalEntry(
            term=str(row.get("term") or ""),
            album_frequency=int(row.get("album_frequency") or 0),
            cooccurring_tags=list(row.get("cooccurring_tags") or []),
            examples=list(row.get("examples") or []),
            decision=str(row.get("decision") or "pending"),
            proposal=GrowthProposal(
                name=str(p.get("name") or row.get("term") or ""),
                kind=str(p.get("kind") or "subgenre"),
                status=str(p.get("status") or "active"),
                specificity_score=float(p.get("specificity_score") or 0.5),
                parent_edges=list(p.get("parent_edges") or []),
                similar_to=list(p.get("similar_to") or []),
                alias_variants=list(p.get("alias_variants") or []),
                term_kind_confirm=str(p.get("term_kind_confirm") or "genre"),
                rationale=str(p.get("rationale") or ""),
            ),
        ))
    return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k proposal_file_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): editable proposal YAML read/write (SP3a task 5)"
```

---

## Task 6: `validate_proposal` (structural validation)

**Files:**
- Modify: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py

def _valid_proposal(name="brand new genre", parent="electronic"):
    return graph_growth.GrowthProposal(
        name=name, kind="subgenre", status="active", specificity_score=0.7,
        parent_edges=[{"target": parent, "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="x",
    )


def test_validate_proposal_accepts_valid():
    taxonomy = load_default_layered_taxonomy()
    assert graph_growth.validate_proposal(taxonomy, _valid_proposal()) == []


def test_validate_proposal_rejects_dangling_parent():
    taxonomy = load_default_layered_taxonomy()
    errs = graph_growth.validate_proposal(
        taxonomy, _valid_proposal(parent="no such family zzz"))
    assert any("parent" in e.lower() for e in errs)


def test_validate_proposal_rejects_duplicate_name():
    taxonomy = load_default_layered_taxonomy()
    existing = taxonomy.genres[0].name  # already in the taxonomy
    errs = graph_growth.validate_proposal(taxonomy, _valid_proposal(name=existing))
    assert any("exist" in e.lower() for e in errs)


def test_validate_proposal_rejects_non_genre_and_bad_specificity():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_proposal()
    p.term_kind_confirm = "noise"
    assert any("genre" in e.lower() for e in graph_growth.validate_proposal(taxonomy, p))
    p2 = _valid_proposal()
    p2.specificity_score = 1.5
    assert any("specificity" in e.lower() for e in graph_growth.validate_proposal(taxonomy, p2))


def test_validate_proposal_requires_a_parent():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_proposal()
    p.parent_edges = []
    assert any("parent" in e.lower() for e in graph_growth.validate_proposal(taxonomy, p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k validate_proposal -v`
Expected: FAIL — no attribute `validate_proposal`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ai_genre_enrichment/graph_growth.py
from .layered_taxonomy import _record_id


def _name_exists(taxonomy: LayeredTaxonomy, name: str) -> bool:
    norm = normalize_taxonomy_name(name)
    if taxonomy.genre_by_name(norm) is not None:
        return True
    if taxonomy.facet_by_name(norm) is not None:
        return True
    return taxonomy.genre_by_id(_record_id(name)) is not None


def validate_proposal(taxonomy: LayeredTaxonomy, proposal: GrowthProposal) -> list[str]:
    """Return a list of structural errors ([] means the proposal is safe to add)."""
    errors: list[str] = []
    name = (proposal.name or "").strip()
    if not name:
        errors.append("Proposal has an empty name.")
        return errors
    if proposal.term_kind_confirm != "genre":
        errors.append(f"Not a genre (term_kind_confirm={proposal.term_kind_confirm}); skip.")
    if proposal.kind not in {"genre", "subgenre"}:
        errors.append(f"Unsupported kind: {proposal.kind}")
    if not (0.0 <= float(proposal.specificity_score) <= 1.0):
        errors.append(f"specificity_score out of range: {proposal.specificity_score}")
    if _name_exists(taxonomy, name):
        errors.append(f"A taxonomy record named/sluged like '{name}' already exists.")
    if not proposal.parent_edges:
        errors.append("A new leaf genre needs at least one parent edge.")
    for edge in proposal.parent_edges:
        target = str(edge.get("target") or "").strip()
        norm = normalize_taxonomy_name(target)
        if taxonomy.genre_by_name(norm) is None and taxonomy.facet_by_name(norm) is None:
            errors.append(f"parent edge target does not exist: '{target}'")
    for target in proposal.similar_to:
        norm = normalize_taxonomy_name(target)
        if taxonomy.genre_by_name(norm) is None and taxonomy.facet_by_name(norm) is None:
            errors.append(f"similar_to target does not exist: '{target}'")
    return errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k validate_proposal -v`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): structural validation of growth proposals (SP3a task 6)"
```

---

## Task 7: `append_approved_to_taxonomy`

Append validated proposals as new records to the taxonomy YAML, bump the version, and confirm the result re-loads with the new genres present.

**Files:**
- Modify: `src/ai_genre_enrichment/graph_growth.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py
import shutil
from src.ai_genre_enrichment.layered_taxonomy import (
    DEFAULT_TAXONOMY_PATH, load_layered_taxonomy)


def test_append_approved_adds_genre_and_reloads(tmp_path):
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    taxonomy = load_layered_taxonomy(tax_path)
    proposal = graph_growth.GrowthProposal(
        name="vaporwave", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=["vapor wave"],
        term_kind_confirm="genre", rationale="microgenre",
    )
    result = graph_growth.append_approved_to_taxonomy(
        tax_path, [proposal], new_version="0.3.0-grown-test")
    assert result.appended == 1
    # Re-load: the new genre is present and resolves a parent family.
    grown = load_layered_taxonomy(tax_path)
    assert grown.version == "0.3.0-grown-test"
    gid = graph_growth._record_id("vaporwave")
    assert grown.genre_by_id(gid) is not None
    assert grown.genres  # still valid taxonomy (loader _validate_taxonomy passed)
    # alias variant registered
    assert grown.exact_alias_target_for_name("vapor wave") is not None
```

(If `electronic` is not a record name in the current taxonomy, substitute an
existing family name — check with
`python -c "from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy as L; print([g.name for g in L().genres if g.kind=='family'])"`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k append_approved -v`
Expected: FAIL — no attribute `append_approved_to_taxonomy`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ai_genre_enrichment/graph_growth.py

@dataclass
class AppendResult:
    appended: int
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _genre_record(proposal: GrowthProposal) -> dict:
    parent_edges = [
        {
            "target": e["target"],
            "edge_type": e.get("edge_type", "family_context"),
            "weight": float(e.get("weight", 0.55)),
            "confidence": float(e.get("confidence", 0.8)),
            "notes": None,
        }
        for e in proposal.parent_edges
    ]
    # similar_to becomes bridge_to edges in the same parent_edges channel the
    # loader reads (it resolves any parent_edges target by name).
    for target in proposal.similar_to:
        parent_edges.append({
            "target": target, "edge_type": "bridge_to",
            "weight": 0.4, "confidence": 0.6, "notes": "similar_to (growth)",
        })
    return {
        "name": proposal.name,
        "kind": proposal.kind,
        "role": "leaf",
        "status": proposal.status or "active",
        "facet_type": None,
        "specificity_score": float(proposal.specificity_score),
        "canonical_target": None,
        "parent_edges": parent_edges,
        "secondary_roles": [],
        "reject_reason": None,
        "alias_policy": None,
        "source_policy": "growth",
        "possible_context_target": None,
        "notes": proposal.rationale or "Added via SP3a graph growth.",
    }


def _alias_record(variant: str, canonical_name: str) -> dict:
    return {
        "name": variant,
        "kind": "alias",
        "role": "alias",
        "status": "alias_only",
        "facet_type": None,
        "specificity_score": None,
        "canonical_target": canonical_name,
        "parent_edges": [],
        "secondary_roles": [],
        "reject_reason": None,
        "alias_policy": {"type": "plain"},
        "source_policy": None,
        "possible_context_target": None,
        "notes": "Spelling variant (growth).",
    }


def append_approved_to_taxonomy(taxonomy_path, approved: list[GrowthProposal],
                                *, new_version: str) -> AppendResult:
    """Append approved proposals as records to the taxonomy YAML and bump version.

    Caller is responsible for having validated each proposal first. New genre
    records are appended before their alias records so name targets resolve.
    """
    path = Path(taxonomy_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = data.setdefault("records", [])
    appended = 0
    for proposal in approved:
        records.append(_genre_record(proposal))
        for variant in proposal.alias_variants:
            if variant and normalize_taxonomy_name(variant) != normalize_taxonomy_name(proposal.name):
                records.append(_alias_record(variant, proposal.name))
        appended += 1
    data["taxonomy_version"] = new_version
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return AppendResult(appended=appended)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k append_approved -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): append approved proposals into taxonomy YAML (SP3a task 7)"
```

---

## Task 8: CLI `graph-propose-growth`

Gather candidates → collapse variants → AI-propose each → write the proposal YAML.

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py
from src.ai_genre_enrichment import graph_growth as gg


def test_cli_propose_growth_writes_file(tmp_path, monkeypatch):
    # sidecar with one unmapped term on >=3 albums
    side = tmp_path / "sidecar.db"
    store = SidecarStore(side)
    store.initialize()
    for i in range(3):
        _page_with_tags(store, f"a{i}::b{i}", f"a{i}", f"b{i}",
                        "lastfm_tags", ["vaporwave", "ambient"])
    meta = tmp_path / "metadata.db"   # discovery uses metadata; not needed here
    import sqlite3
    sqlite3.connect(meta).close()

    # Stub the AI proposal so no network is hit.
    def fake_propose(candidate, taxonomy, *, client, web_mode="off"):
        return gg.GrowthProposal(
            name=candidate.term, kind="subgenre", status="active",
            specificity_score=0.8,
            parent_edges=[{"target": "electronic", "edge_type": "family_context",
                           "weight": 0.55, "confidence": 0.8}],
            similar_to=[], alias_variants=candidate.variants,
            term_kind_confirm="genre", rationale="x")
    monkeypatch.setattr(gg, "propose_placement", fake_propose)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = tmp_path / "proposals.yaml"
    from scripts import ai_genre_enrich as cli
    rc = cli.main([
        "--sidecar-db", str(side), "--metadata-db", str(meta),
        "graph-propose-growth", "--out", str(out), "--min-album-freq", "3",
    ])
    assert rc == 0
    entries = gg.read_proposals(out)
    assert any(e.term == "vaporwave" for e in entries)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k cli_propose_growth -v`
Expected: FAIL — `argument command: invalid choice: 'graph-propose-growth'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/ai_genre_enrich.py`, register the subcommand in `build_parser()` (near the other `graph-*` parsers):

```python
    propose_growth = sub.add_parser(
        "graph-propose-growth",
        help="Gather unmapped-genre candidates and AI-propose taxonomy placements")
    propose_growth.add_argument("--out", required=True,
                                help="Path to write the editable proposal YAML")
    propose_growth.add_argument("--min-album-freq", type=int, default=3)
    propose_growth.add_argument("--limit", type=int, default=None,
                                help="Cap number of candidates proposed (cost control)")
    propose_growth.add_argument("--web-mode", choices=["off", "auto", "required"],
                                default="off")
    propose_growth.add_argument("--model", default=DEFAULT_MODEL)
    propose_growth.add_argument("--openai-api-key")
```

Dispatch in `main()` (next to the other `graph-*` dispatches):

```python
    if args.command == "graph-propose-growth":
        return cmd_graph_propose_growth(args)
```

Add the command function:

```python
def cmd_graph_propose_growth(args: argparse.Namespace) -> int:
    import os
    from src.ai_genre_enrichment import graph_growth
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    candidates = graph_growth.collapse_variants(
        graph_growth.gather_growth_candidates(
            store, taxonomy, min_album_freq=args.min_album_freq))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("No growth candidates found.")
        graph_growth.write_proposals(args.out, [])
        return 0

    api_key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            api_key = Config().openai_api_key
        except (FileNotFoundError, AttributeError):
            api_key = None
    client = OpenAIEnrichmentClient(model=args.model, api_key=api_key,
                                    web_mode=args.web_mode)

    items = []
    total = len(candidates)
    for idx, cand in enumerate(candidates, start=1):
        try:
            proposal = graph_growth.propose_placement(
                cand, taxonomy, client=client, web_mode=args.web_mode)
            items.append((cand, proposal))
            print(f"[{idx}/{total}] proposed {cand.term} -> {proposal.name} "
                  f"(kind={proposal.kind}, parents={[e.get('target') for e in proposal.parent_edges]})")
        except Exception as exc:  # one bad proposal shouldn't lose the batch
            print(f"[{idx}/{total}] FAILED {cand.term}: {type(exc).__name__}: {exc}")
    graph_growth.write_proposals(args.out, items)
    print(f"Wrote {len(items)} proposal(s) to {args.out}. Review then run graph-ingest-growth.")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k cli_propose_growth -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): graph-propose-growth CLI (SP3a task 8)"
```

---

## Task 9: CLI `graph-ingest-growth`

Read the reviewed proposal YAML → keep `decision: keep` → validate → append → re-import; `--dry-run` writes nothing.

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_graph_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_graph_growth.py

def test_cli_ingest_growth_appends_kept_only(tmp_path):
    import shutil, sqlite3
    from src.ai_genre_enrichment.layered_taxonomy import (
        DEFAULT_TAXONOMY_PATH, load_layered_taxonomy)
    from scripts import ai_genre_enrich as cli

    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    side = tmp_path / "sidecar.db"; SidecarStore(side).initialize()

    # one keep, one reject
    proposals_path = tmp_path / "proposals.yaml"
    keep = gg.GrowthProposal(
        name="vaporwave", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="x")
    rej = gg.GrowthProposal(
        name="aaron", kind="subgenre", status="active", specificity_score=0.5,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.5, "confidence": 0.5}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="noise")
    gg.write_proposals(proposals_path, [
        (gg.GrowthCandidate(term="vaporwave", album_frequency=9), keep),
        (gg.GrowthCandidate(term="aaron", album_frequency=4), rej),
    ])
    # user edits: keep vaporwave, reject aaron
    import yaml
    rows = yaml.safe_load(proposals_path.read_text(encoding="utf-8"))
    rows[0]["decision"] = "keep"
    rows[1]["decision"] = "reject"
    proposals_path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")

    rc = cli.main([
        "--sidecar-db", str(side),
        "graph-ingest-growth", "--proposals", str(proposals_path),
        "--taxonomy-path", str(tax_path), "--new-version", "0.3.0-grown-test",
    ])
    assert rc == 0
    grown = load_layered_taxonomy(tax_path)
    assert grown.genre_by_id(gg._record_id("vaporwave")) is not None
    assert grown.genre_by_id(gg._record_id("aaron")) is None   # rejected


def test_cli_ingest_growth_dry_run_writes_nothing(tmp_path):
    import shutil
    from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
    from scripts import ai_genre_enrich as cli
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    before = tax_path.read_text(encoding="utf-8")
    side = tmp_path / "sidecar.db"; SidecarStore(side).initialize()
    proposals_path = tmp_path / "proposals.yaml"
    keep = gg.GrowthProposal(
        name="vaporwave", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="x")
    gg.write_proposals(proposals_path, [(gg.GrowthCandidate(term="vaporwave", album_frequency=9), keep)])
    import yaml
    rows = yaml.safe_load(proposals_path.read_text(encoding="utf-8"))
    rows[0]["decision"] = "keep"
    proposals_path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")

    rc = cli.main([
        "--sidecar-db", str(side),
        "graph-ingest-growth", "--proposals", str(proposals_path),
        "--taxonomy-path", str(tax_path), "--new-version", "0.3.0-x", "--dry-run",
    ])
    assert rc == 0
    assert tax_path.read_text(encoding="utf-8") == before   # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph_growth.py -k cli_ingest_growth -v`
Expected: FAIL — `argument command: invalid choice: 'graph-ingest-growth'`.

- [ ] **Step 3: Write minimal implementation**

Register in `build_parser()`:

```python
    ingest_growth = sub.add_parser(
        "graph-ingest-growth",
        help="Validate + append decision:keep proposals into the taxonomy YAML")
    ingest_growth.add_argument("--proposals", required=True)
    ingest_growth.add_argument("--taxonomy-path", default=None,
                               help="Taxonomy YAML to grow (default: the packaged one)")
    ingest_growth.add_argument("--new-version", required=True,
                               help="taxonomy_version to stamp after growth")
    ingest_growth.add_argument("--dry-run", action="store_true")
```

Dispatch in `main()`:

```python
    if args.command == "graph-ingest-growth":
        return cmd_graph_ingest_growth(args)
```

Command function:

```python
def cmd_graph_ingest_growth(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment import graph_growth
    from src.ai_genre_enrichment.layered_taxonomy import (
        DEFAULT_TAXONOMY_PATH, load_layered_taxonomy)

    tax_path = args.taxonomy_path or str(DEFAULT_TAXONOMY_PATH)
    taxonomy = load_layered_taxonomy(tax_path)
    entries = graph_growth.read_proposals(args.proposals)
    kept = [e for e in entries if e.decision == "keep"]
    if not kept:
        print("No proposals marked decision: keep.")
        return 0

    approved = []
    skipped = []
    for e in kept:
        errs = graph_growth.validate_proposal(taxonomy, e.proposal)
        if errs:
            skipped.append((e.proposal.name, "; ".join(errs)))
        else:
            approved.append(e.proposal)

    for name, reason in skipped:
        print(f"SKIP {name}: {reason}")

    if args.dry_run:
        print(f"[dry-run] would append {len(approved)} record(s); "
              f"{len(skipped)} skipped. No write.")
        return 0

    result = graph_growth.append_approved_to_taxonomy(
        tax_path, approved, new_version=args.new_version)
    # Re-import the grown taxonomy into the sidecar graph tables.
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    store.upsert_layered_taxonomy(load_layered_taxonomy(tax_path))
    print(f"Appended {result.appended} genre(s); skipped {len(skipped)}. "
          f"Taxonomy now {args.new_version}.")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph_growth.py -k cli_ingest_growth -v`
Expected: PASS (both).

- [ ] **Step 5: Run full SP3a suite + ruff**

Run: `pytest tests/unit/test_graph_growth.py -v && ruff check src/ai_genre_enrichment/graph_growth.py scripts/ai_genre_enrich.py src/ai_genre_enrichment/storage.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_graph_growth.py
git commit -m "feat(growth): graph-ingest-growth CLI with dry-run + validation (SP3a task 9)"
```

---

## Task 10: Real-data candidate report (read-only smoke)

Confirms the candidate pass over the live sidecar produces a sane, recognizable candidate list before any AI spend. Read-only; no taxonomy writes.

**Files:**
- Create: `scripts/growth_candidate_report.py`

- [ ] **Step 1: Write the script**

```python
# scripts/growth_candidate_report.py
#!/usr/bin/env python3
"""Read-only report of graph-growth candidates over the live sidecar. No writes."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ai_genre_enrichment.graph_growth import collapse_variants, gather_growth_candidates
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    p.add_argument("--min-album-freq", type=int, default=3)
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args(argv)

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    cands = collapse_variants(
        gather_growth_candidates(store, taxonomy, min_album_freq=args.min_album_freq))
    print(f"{len(cands)} growth candidate(s) at min_album_freq={args.min_album_freq}")
    for c in cands[: args.top]:
        variants = f" (variants: {', '.join(c.variants)})" if c.variants else ""
        print(f"  {c.album_frequency:4d}  {c.term}{variants}  "
              f"~ {', '.join(c.cooccurring_tags[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it (after the SP2 scrapes have collected tags)**

Run (PowerShell): `python scripts\growth_candidate_report.py --top 50`
Expected: a ranked list whose top entries are recognizable genres (not noise like
`aaron`/`1996`). If the top is dominated by noise, raise `--min-album-freq` or
extend the taxonomy's `reject` records before proposing. STOP and review here
before spending on `graph-propose-growth`.

- [ ] **Step 3: Commit**

```bash
git add scripts/growth_candidate_report.py
git commit -m "feat(growth): read-only growth-candidate report (SP3a task 10)"
```

---

## Self-Review

**Spec coverage:**
- Candidate gathering (classify, `review`+`canonical_id is None`, ≥ threshold, evidence) → Task 2.
- Alias-collapse before proposing → Task 3.
- AI full-placement proposal (slug/kind/parent/specificity/edges/term_kind_confirm) grounded in taxonomy neighborhood → Task 4.
- Editable proposal file + decision field → Tasks 5, 8.
- Structural validation (unique slug, parents exist, specificity range, kind, genre-only, ≥1 parent) → Task 6.
- Ingest appends records + alias records, bumps version, re-imports; loader `_validate_taxonomy` is the backstop → Tasks 7, 9.
- `--dry-run` writes nothing → Task 9.
- Read-only real-data smoke before spend → Task 10.
- Web search off by default for proposals (`--web-mode off` default) → Task 8.

**Placeholder scan:** none — every step has runnable code/commands. Fixture caveats (substitute an unmapped term / an existing family name) are inline, with a one-liner to discover the right value.

**Type/name consistency:** `GrowthCandidate(term, album_frequency, cooccurring_tags, examples, variants)`, `GrowthProposal(name, kind, status, specificity_score, parent_edges, similar_to, alias_variants, term_kind_confirm, rationale)`, `ProposalEntry(term, album_frequency, cooccurring_tags, examples, decision, proposal)`, `gather_growth_candidates(store, taxonomy, *, min_album_freq, max_examples, max_cooccurring)`, `collapse_variants(list)`, `propose_placement(candidate, taxonomy, *, client, web_mode)`, `write_proposals(path, list[tuple])` / `read_proposals(path)`, `validate_proposal(taxonomy, proposal) -> list[str]`, `append_approved_to_taxonomy(path, list[GrowthProposal], *, new_version) -> AppendResult` — all consistent across tasks. CLI reuses `_record_id`, `normalize_taxonomy_name`, `load_layered_taxonomy`, `SidecarStore.upsert_layered_taxonomy`, `OpenAIEnrichmentClient`, `WebMode`, all verified present in current code.

**Safety:** taxonomy YAML is additive + git-versioned; ingest `--dry-run`; loader validation backstop; Task 10 read-only. No `metadata.db` writes in SP3a.
