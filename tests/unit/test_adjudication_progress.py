"""run_adjudication must tick a progress reporter once per album (the AI genre
adjudication stage otherwise blocks silently on sequential Claude calls -- the
2026-07-07 "analyze hangs" report). progress=None stays a no-op.
"""
from src.ai_genre_enrichment.adjudication_runner import run_adjudication


class _FakeProg:
    def __init__(self):
        self.updates = 0
        self.finished = 0
        self.details = []

    def update(self, n=1, detail=None):
        self.updates += 1
        self.details.append(detail)

    def finish(self, detail=None):
        self.finished += 1


class _FakeStore:
    def save(self, **kw):
        pass


class _FakeAdapter:
    def node(self, name):
        return None

    def canonicalize_tag(self, tag):
        return tag


class _FakeClient:
    """Streams a result per album to on_result (failed path -> no response-format coupling)."""

    def call_structured_session(self, items, *, response_format, validator,
                                instructions, on_result, reset_every):
        for album_id, _prompt in items:
            on_result(album_id, None, "stub-fail", {})


def _todo(n):
    return [
        {"album_id": f"a{i}", "release_key": None, "prompt": "p",
         "input_hash": "h", "file_tags": []}
        for i in range(n)
    ]


def test_run_adjudication_reports_progress_per_album():
    fp = _FakeProg()
    summary = run_adjudication(
        _FakeStore(), _todo(3), model="sonnet", instructions="x",
        prompt_version="v1", adapter=_FakeAdapter(), client=_FakeClient(),
        progress=fp,
    )
    assert fp.updates == 3          # one tick per album
    assert fp.finished == 1         # exactly one finish
    assert fp.details == ["a0", "a1", "a2"]
    assert summary.failed == 3


def test_run_adjudication_progress_none_is_noop():
    # No progress reporter -> unchanged behavior, no crash.
    summary = run_adjudication(
        _FakeStore(), _todo(2), model="sonnet", instructions="x",
        prompt_version="v1", adapter=_FakeAdapter(), client=_FakeClient(),
    )
    assert summary.failed == 2
