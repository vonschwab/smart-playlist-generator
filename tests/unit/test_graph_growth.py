import json
import shutil

from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment import graph_growth
from src.ai_genre_enrichment.layered_taxonomy import (
    DEFAULT_TAXONOMY_PATH, load_default_layered_taxonomy, load_layered_taxonomy)


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
