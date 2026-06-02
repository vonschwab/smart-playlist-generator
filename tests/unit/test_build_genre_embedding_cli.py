from __future__ import annotations

from scripts import build_genre_embedding


def _run_main(monkeypatch, *args):
    calls = []
    monkeypatch.setattr(
        build_genre_embedding,
        "build_genre_embedding_sidecar",
        lambda *call_args, **kwargs: calls.append((call_args, kwargs)),
    )
    monkeypatch.setattr(
        build_genre_embedding,
        "configure_logging",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        build_genre_embedding.sys,
        "argv",
        ["build_genre_embedding.py", *args],
    )

    assert build_genre_embedding.main() == 0
    assert len(calls) == 1
    return calls[0][1]


def test_cli_default_is_corpus_only(monkeypatch):
    kwargs = _run_main(monkeypatch)

    assert kwargs["skip_prior"] is True


def test_cli_use_prior_is_explicit_opt_in(monkeypatch):
    kwargs = _run_main(monkeypatch, "--use-prior", "--provider", "openai")

    assert kwargs["skip_prior"] is False
    assert kwargs["provider"] == "openai"


def test_cli_skip_prior_remains_supported(monkeypatch):
    kwargs = _run_main(monkeypatch, "--skip-prior")

    assert kwargs["skip_prior"] is True
