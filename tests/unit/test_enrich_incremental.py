"""Incremental enrich: only (re)materialize new/changed releases.

Guards against the wholesale re-derivation trap (2026-06-13): a routine
analyze_library run used to re-materialize EVERY release through the fusion
policy, undoing targeted genre fixes and re-introducing collateral. The guard
fingerprints each release's evidence; unchanged releases are left exactly as
they are. New fusion rules still apply to every release that IS materialized.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ai_genre_enrichment.storage import SidecarStore


def _store(tmp_path) -> SidecarStore:
    s = SidecarStore(tmp_path / "side.db")
    s.initialize()
    return s


class TestMaterializationAction:
    @pytest.mark.parametrize(
        "force,has,stored,cur,expected",
        [
            (True, True, "x", "x", "materialize"),    # --force always re-derives
            (False, False, None, "x", "materialize"),  # new music: no assignments yet
            (False, True, None, "x", "adopt"),         # bootstrap: keep existing state
            (False, True, "x", "x", "skip"),           # unchanged evidence
            (False, True, "x", "y", "materialize"),    # evidence changed
        ],
    )
    def test_decisions(self, force, has, stored, cur, expected):
        from scripts.analyze_library import _enrich_materialization_action

        assert (
            _enrich_materialization_action(
                force=force, has_assignments=has, stored_fp=stored, current_fp=cur
            )
            == expected
        )


class TestStorageFingerprint:
    def test_has_genre_assignments(self, tmp_path):
        s = _store(tmp_path)
        assert s.has_genre_assignments("a::b") is False
        s.replace_layered_assignments_for_release(
            release_id="a::b",
            artist="a",
            album="b",
            genre_assignments=[
                {
                    "genre_id": "rock",
                    "assignment_layer": "observed_leaf",
                    "confidence": 0.9,
                    "source_reliability": 0.7,
                    "evidence_count": 1,
                    "rejected_by_user": False,
                    "provenance": {},
                }
            ],
            facet_assignments=[],
        )
        assert s.has_genre_assignments("a::b") is True

    def test_fingerprint_roundtrip_and_upsert(self, tmp_path):
        s = _store(tmp_path)
        assert s.materialization_fingerprint("a::b") is None
        s.set_materialization_fingerprint("a::b", "deadbeef")
        assert s.materialization_fingerprint("a::b") == "deadbeef"
        s.set_materialization_fingerprint("a::b", "cafe")
        assert s.materialization_fingerprint("a::b") == "cafe"


class TestEvidenceFingerprint:
    def _seed_page(self, s: SidecarStore, rk: str, tags: list[str]) -> int:
        artist, album = rk.split("::")
        pid = s.upsert_source_page(
            release_key=rk,
            normalized_artist=artist,
            normalized_album=album,
            album_id="alb",
            source_url=f"https://{artist}.bandcamp.com/album/{album}",
            source_type="bandcamp_release",
            identity_status="confirmed",
            identity_confidence=0.95,
            evidence_summary="t",
        )
        s.replace_source_tags(pid, tags)
        s.classify_source_tags(pid, adjudicate=False)
        return pid

    def _rel(self, rk: str, existing=None):
        artist, album = rk.split("::")
        return SimpleNamespace(
            release_key=rk,
            normalized_artist=artist,
            normalized_album=album,
            album_id="alb",
            existing_genres_by_source=existing or {},
        )

    def test_stable_for_unchanged_evidence(self, tmp_path):
        from scripts.analyze_library import _release_evidence_fingerprint

        s = _store(tmp_path)
        self._seed_page(s, "duster::stratosphere", ["slowcore", "dream pop"])
        rel = self._rel("duster::stratosphere", {"track:file": ["slowcore"]})
        fp1 = _release_evidence_fingerprint(s, rel)
        fp2 = _release_evidence_fingerprint(s, rel)
        assert fp1 and fp1 == fp2

    def test_changes_when_a_tag_is_added(self, tmp_path):
        from scripts.analyze_library import _release_evidence_fingerprint

        s = _store(tmp_path)
        pid = self._seed_page(s, "duster::stratosphere", ["slowcore"])
        rel = self._rel("duster::stratosphere")
        fp1 = _release_evidence_fingerprint(s, rel)
        s.replace_source_tags(pid, ["slowcore", "shoegaze"])
        s.classify_source_tags(pid, adjudicate=False)
        fp2 = _release_evidence_fingerprint(s, rel)
        assert fp1 != fp2

    def test_changes_when_existing_file_tags_change(self, tmp_path):
        from scripts.analyze_library import _release_evidence_fingerprint

        s = _store(tmp_path)
        self._seed_page(s, "x::y", ["rock"])
        fp1 = _release_evidence_fingerprint(s, self._rel("x::y", {}))
        fp2 = _release_evidence_fingerprint(s, self._rel("x::y", {"track:file": ["punk"]}))
        assert fp1 != fp2
