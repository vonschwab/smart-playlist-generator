import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_phase2_schemas_importable():
    from src.playlist_web.schemas import (
        CandidateOut,
        ReplaceSuggestionsRequest,
        ReplaceSuggestionsResponse,
        BlacklistRequest,
        EditGenresRequest,
        PlexExportRequest,
    )

    cand = CandidateOut(track_id="k1", title="T", artist="A", album="Al", genres=["x"], fit_score=0.7)
    assert cand.fit_score == 0.7

    bl = BlacklistRequest(scope="album", value="Leisure", artist="Marbled Eye")
    assert bl.artist == "Marbled Eye"

    cands = ReplaceSuggestionsResponse.from_worker_candidates(
        position=3,
        raw=[{"rating_key": "k9", "title": "Song", "artist": "Band", "album": "LP",
              "genres": ["slowcore"], "mean_t": 0.66}],
    )
    assert cands.candidates[0].track_id == "k9"
    assert cands.candidates[0].fit_score == 0.66
