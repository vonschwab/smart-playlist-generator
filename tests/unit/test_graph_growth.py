import shutil

import pytest

from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment import graph_growth
from src.ai_genre_enrichment import graph_growth as gg
from src.ai_genre_enrichment.layered_taxonomy import (
    DEFAULT_TAXONOMY_PATH, _record_id, load_default_layered_taxonomy,
    load_layered_taxonomy, normalize_taxonomy_name)


@pytest.fixture(autouse=True)
def _pin_openai_provider(monkeypatch):
    """These tests stub the OpenAI client class; pin the factory to it."""
    monkeypatch.setenv("PG_AI_PROVIDER", "openai")


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


def test_gather_candidates_keeps_unmapped_genres_above_threshold(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    # "xyzzy-growth-test" is unmapped; appears on 3 releases -> candidate.
    for i in range(3):
        _page_with_tags(store, f"a{i}::alb{i}", f"a{i}", f"alb{i}",
                        "lastfm_tags", ["xyzzy-growth-test", "ambient"])
    # "rock" is a known family -> dropped. "xyzzy-growth-test" on 3 albums -> kept.
    cands = graph_growth.gather_growth_candidates(store, taxonomy, min_album_freq=3)
    terms = {c.term for c in cands}
    assert "xyzzy-growth-test" in terms
    vw = next(c for c in cands if c.term == "xyzzy-growth-test")
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


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def call_structured(self, prompt, response_format, *, instructions):
        self.calls.append(prompt)
        return self._payload


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


def test_proposal_file_round_trip_preserves_facet_and_alias_fields(tmp_path):
    # `read_proposals` must carry `facet_type`/`canonical_target` through, or a
    # facet/alias proposal silently loses them on the write->edit->read cycle
    # that `graph-ingest-growth` relies on (the YAML is the human review surface).
    facet = graph_growth.GrowthProposal(
        name="brand new descriptor", kind="facet", status="active",
        specificity_score=0.3, term_kind_confirm="facet", facet_type="texture",
        rationale="x",
    )
    alias = graph_growth.GrowthProposal(
        name="orchestra", kind="alias", status="alias_only", specificity_score=0.0,
        term_kind_confirm="genre", canonical_target="orchestral", rationale="x",
    )
    path = tmp_path / "proposals.yaml"
    graph_growth.write_proposals(path, [
        (graph_growth.GrowthCandidate(term="brand new descriptor", album_frequency=0), facet),
        (graph_growth.GrowthCandidate(term="orchestra", album_frequency=0), alias),
    ])
    entries = graph_growth.read_proposals(path)
    by_term = {e.term: e.proposal for e in entries}
    assert by_term["brand new descriptor"].facet_type == "texture"
    assert by_term["orchestra"].canonical_target == "orchestral"


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


def test_validate_proposal_rejects_alias_parent_target():
    # "rnb" is a `kind: alias` record whose canonical_target is "r&b/soul" in
    # the default taxonomy — genre_by_name resolves it, but the loader's
    # parent-edge resolution dict is keyed by canonical name only (and the
    # alias's normalized form differs from the canonical name's), so an
    # alias-named parent target would silently fail to resolve on reload.
    taxonomy = load_default_layered_taxonomy()
    errs = graph_growth.validate_proposal(taxonomy, _valid_proposal(parent="rnb"))
    assert any("alias" in e.lower() for e in errs)


def test_validate_proposal_rejects_facet_parent_target():
    # "lo-fi" is a `kind: facet` record in the default taxonomy — it passes
    # the existence check in validate_proposal, but the loader silently drops
    # facet-target parent edges, leaving the new genre with zero parents.
    taxonomy = load_default_layered_taxonomy()
    errs = graph_growth.validate_proposal(taxonomy, _valid_proposal(parent="lo-fi"))
    assert any("facet" in e.lower() for e in errs)


def _valid_umbrella_proposal(name="brand new umbrella bucket"):
    return graph_growth.GrowthProposal(
        name=name, kind="umbrella", status="active", specificity_score=0.3,
        parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="genre", rationale="x",
    )


def _valid_facet_proposal(name="brand new facet descriptor", facet_type="texture"):
    return graph_growth.GrowthProposal(
        name=name, kind="facet", status="active", specificity_score=0.4,
        parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="facet", facet_type=facet_type, rationale="x",
    )


def test_validate_proposal_accepts_umbrella_without_parent_edges():
    # Umbrellas are context buckets (e.g. "world music" has zero parent edges) —
    # the leaf "needs >=1 parent edge" rule must not apply to them.
    taxonomy = load_default_layered_taxonomy()
    assert graph_growth.validate_proposal(taxonomy, _valid_umbrella_proposal()) == []


def test_validate_proposal_accepts_umbrella_with_parent_edges():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_umbrella_proposal()
    p.parent_edges = [{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.4, "confidence": 0.8}]
    assert graph_growth.validate_proposal(taxonomy, p) == []


def test_validate_proposal_accepts_facet_with_valid_facet_type():
    taxonomy = load_default_layered_taxonomy()
    assert graph_growth.validate_proposal(taxonomy, _valid_facet_proposal()) == []


def test_validate_proposal_rejects_facet_missing_facet_type():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_facet_proposal(facet_type=None)
    assert any("facet_type" in e.lower() for e in graph_growth.validate_proposal(taxonomy, p))


def test_validate_proposal_rejects_facet_with_invalid_facet_type():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_facet_proposal(facet_type="theme")  # not in the closed enum
    assert any("facet_type" in e.lower() for e in graph_growth.validate_proposal(taxonomy, p))


def test_validate_proposal_rejects_facet_with_parent_edges_or_similar_to():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_facet_proposal()
    p.parent_edges = [{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.4, "confidence": 0.8}]
    p.similar_to = ["ambient"]
    errs = graph_growth.validate_proposal(taxonomy, p)
    assert any("parent_edges" in e for e in errs)
    assert any("similar_to" in e for e in errs)


def _valid_alias_proposal(name="orchestra redirect term", target="orchestral"):
    return graph_growth.GrowthProposal(
        name=name, kind="alias", status="alias_only", specificity_score=0.0,
        parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="genre", canonical_target=target, rationale="x",
    )


def test_validate_proposal_accepts_alias_to_existing_facet():
    # "orchestral" is a `kind: facet` canonical record — alias-to-facet is a
    # loader-supported resolution path (unlike alias-to-facet *parent edges*).
    taxonomy = load_default_layered_taxonomy()
    assert graph_growth.validate_proposal(taxonomy, _valid_alias_proposal()) == []


def test_validate_proposal_accepts_alias_to_existing_genre():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_alias_proposal(name="electro music redirect", target="electronic")
    assert graph_growth.validate_proposal(taxonomy, p) == []


def test_validate_proposal_rejects_alias_target_that_is_itself_an_alias():
    # "rnb" is itself a `kind: alias` pointing at "r&b/soul" — chaining aliases
    # orphans the new one on load.
    taxonomy = load_default_layered_taxonomy()
    p = _valid_alias_proposal(name="rnb redirect term", target="rnb")
    errs = graph_growth.validate_proposal(taxonomy, p)
    assert any("alias" in e.lower() for e in errs)


def test_validate_proposal_rejects_alias_missing_canonical_target():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_alias_proposal(target=None)
    assert any("canonical_target" in e for e in graph_growth.validate_proposal(taxonomy, p))


def test_validate_proposal_rejects_alias_with_dangling_target():
    taxonomy = load_default_layered_taxonomy()
    p = _valid_alias_proposal(target="no such canonical record zzz")
    assert any("does not exist" in e for e in graph_growth.validate_proposal(taxonomy, p))


def test_append_approved_adds_alias_to_existing_facet(tmp_path):
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    proposal = graph_growth.GrowthProposal(
        name="orchestra", kind="alias", status="alias_only", specificity_score=0.0,
        parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="genre", canonical_target="orchestral", rationale="x",
    )
    result = graph_growth.append_approved_to_taxonomy(
        tax_path, [proposal], new_version="0.3.0-grown-test")
    assert result.appended == 1
    grown = load_layered_taxonomy(tax_path)
    target = grown.exact_alias_target_for_name("orchestra")
    assert target is not None
    assert normalize_taxonomy_name(target.name) == normalize_taxonomy_name("orchestral")


def test_append_approved_adds_genre_and_reloads(tmp_path):
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    taxonomy = load_layered_taxonomy(tax_path)
    proposal = graph_growth.GrowthProposal(
        name="xyzzy-growth-test", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=["xyzzy-gt"],
        term_kind_confirm="genre", rationale="microgenre",
    )
    result = graph_growth.append_approved_to_taxonomy(
        tax_path, [proposal], new_version="0.3.0-grown-test")
    assert result.appended == 1
    # Re-load: the new genre is present and resolves a parent family.
    grown = load_layered_taxonomy(tax_path)
    assert grown.version == "0.3.0-grown-test"
    gid = _record_id("xyzzy-growth-test")
    assert grown.genre_by_id(gid) is not None
    assert grown.genres  # still valid taxonomy (loader _validate_taxonomy passed)
    # alias variant registered
    assert grown.exact_alias_target_for_name("xyzzy-gt") is not None


def test_append_approved_adds_umbrella_record(tmp_path):
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    proposal = graph_growth.GrowthProposal(
        name="dream haze", kind="umbrella", status="active", specificity_score=0.32,
        parent_edges=[], similar_to=[], alias_variants=["dreamhaze"],
        term_kind_confirm="genre", rationale="broad context bucket",
    )
    result = graph_growth.append_approved_to_taxonomy(
        tax_path, [proposal], new_version="0.3.0-grown-test")
    assert result.appended == 1
    grown = load_layered_taxonomy(tax_path)
    gid = _record_id("dream haze")
    rec = grown.genre_by_id(gid)
    assert rec is not None
    assert rec.kind == "umbrella"
    assert rec.role == "context"
    assert grown.exact_alias_target_for_name("dreamhaze") is not None


def test_append_approved_adds_facet_record(tmp_path):
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    proposal = graph_growth.GrowthProposal(
        name="brand new descriptor", kind="facet", status="active",
        specificity_score=0.4, parent_edges=[], similar_to=[], alias_variants=[],
        term_kind_confirm="facet", facet_type="texture", rationale="modifier",
    )
    result = graph_growth.append_approved_to_taxonomy(
        tax_path, [proposal], new_version="0.3.0-grown-test")
    assert result.appended == 1
    grown = load_layered_taxonomy(tax_path)
    fid = _record_id("brand new descriptor")
    rec = grown.facet_by_id(fid)
    assert rec is not None
    assert rec.facet_type == "texture"


def test_cli_propose_growth_writes_file(tmp_path, monkeypatch):
    # sidecar with one unmapped term on >=3 albums
    side = tmp_path / "sidecar.db"
    store = SidecarStore(side)
    store.initialize()
    for i in range(3):
        _page_with_tags(store, f"a{i}::b{i}", f"a{i}", f"b{i}",
                        "lastfm_tags", ["xyzzy-growth-test", "ambient"])
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
    assert any(e.term == "xyzzy-growth-test" for e in entries)


def test_cli_ingest_growth_appends_kept_only(tmp_path):
    import shutil
    from src.ai_genre_enrichment.layered_taxonomy import (
        DEFAULT_TAXONOMY_PATH, load_layered_taxonomy)
    from scripts import ai_genre_enrich as cli

    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    side = tmp_path / "sidecar.db"
    SidecarStore(side).initialize()

    # one keep, one reject
    proposals_path = tmp_path / "proposals.yaml"
    keep = gg.GrowthProposal(
        name="xyzzy-growth-test", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="x")
    rej = gg.GrowthProposal(
        name="aaron", kind="subgenre", status="active", specificity_score=0.5,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.5, "confidence": 0.5}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="noise")
    gg.write_proposals(proposals_path, [
        (gg.GrowthCandidate(term="xyzzy-growth-test", album_frequency=9), keep),
        (gg.GrowthCandidate(term="aaron", album_frequency=4), rej),
    ])
    # user edits: keep xyzzy-growth-test, reject aaron
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
    assert grown.genre_by_id(_record_id("xyzzy-growth-test")) is not None
    assert grown.genre_by_id(_record_id("aaron")) is None   # rejected


def test_cli_ingest_growth_dry_run_writes_nothing(tmp_path):
    import shutil
    from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
    from scripts import ai_genre_enrich as cli
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    before = tax_path.read_text(encoding="utf-8")
    side = tmp_path / "sidecar.db"
    SidecarStore(side).initialize()
    proposals_path = tmp_path / "proposals.yaml"
    keep = gg.GrowthProposal(
        name="xyzzy-growth-test", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "electronic", "edge_type": "family_context",
                       "weight": 0.55, "confidence": 0.8}],
        similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="x")
    gg.write_proposals(proposals_path, [(gg.GrowthCandidate(term="xyzzy-growth-test", album_frequency=9), keep)])
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


def test_cli_ingest_growth_all_invalid_writes_nothing(tmp_path, capsys):
    import shutil
    from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
    from scripts import ai_genre_enrich as cli
    tax_path = tmp_path / "taxonomy.yaml"
    shutil.copy(DEFAULT_TAXONOMY_PATH, tax_path)
    before = tax_path.read_text(encoding="utf-8")
    side = tmp_path / "sidecar.db"
    SidecarStore(side).initialize()
    proposals_path = tmp_path / "proposals.yaml"
    keep = gg.GrowthProposal(
        name="vaporwave", kind="subgenre", status="active", specificity_score=0.8,
        parent_edges=[{"target": "this_genre_does_not_exist_xyz", "edge_type": "family_context",
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
        "--taxonomy-path", str(tax_path), "--new-version", "0.3.0-x",
    ])
    assert rc == 1
    assert tax_path.read_text(encoding="utf-8") == before   # unchanged
    out = capsys.readouterr().out
    assert "SKIP vaporwave" in out
    assert "nothing to append" in out
