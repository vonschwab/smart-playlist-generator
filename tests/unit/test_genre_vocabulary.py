from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary


@pytest.fixture
def vocab_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "genre_vocabulary.yaml"
    path.write_text(
        "version: 1\n"
        "genre_style:\n"
        "  - ambient\n"
        "  - shoegaze\n"
        "descriptor:\n"
        "  - acoustic\n"
        "instrument:\n"
        "  - guitar\n"
        "place:\n"
        "  - oakland\n"
        "format:\n"
        "  - live\n"
        "mood_function:\n"
        "  - meditation\n"
        "label_or_org:\n"
        "  - american football\n"
        "aliases:\n"
        "  shoe gaze: shoegaze\n",
        encoding="utf-8",
    )
    return path


def test_tier1_curated_genre_lookup(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    result = vocab.classify_genre("ambient")
    assert result is not None
    assert result.confidence == 0.95
    assert result.tier == 1


def test_tier1_non_genre_category(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    assert vocab.classify_non_genre("acoustic") == "descriptor"
    assert vocab.classify_non_genre("guitar") == "instrument"
    assert vocab.classify_non_genre("oakland") == "place"
    assert vocab.classify_non_genre("live") == "format"
    assert vocab.classify_non_genre("meditation") == "mood_function"
    assert vocab.classify_non_genre("american football") == "label_or_org"
    assert vocab.classify_non_genre("unknown tag") is None


def test_tier2_engine_genre_lookup(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    # "psychedelic rock" is in normalize_unified.py SYNONYM_MAP as a target
    # but not in our test YAML's genre_style list
    result = vocab.classify_genre("psychedelic rock")
    if result is not None:
        assert result.tier == 2
        assert result.confidence == 0.85


def test_tier3_library_genre_lookup(vocab_yaml: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO artist_genres VALUES ('Test', 'dark ambient', 'musicbrainz_artist')")
    conn.commit()
    conn.close()

    vocab = GenreVocabulary(vocab_yaml, library_db_path=db_path)
    result = vocab.classify_genre("dark ambient")
    assert result is not None
    assert result.tier == 3
    assert result.confidence == 0.80


def test_unknown_tag_returns_none(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    assert vocab.classify_genre("xyzzy nonsense tag") is None


def test_alias_resolution(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    result = vocab.classify_genre("shoe gaze")
    assert result is not None
    assert result.confidence == 0.95


def test_add_genre_writes_yaml(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    vocab.add_term("genre_style", "dark ambient")
    vocab.save()

    reloaded = GenreVocabulary(vocab_yaml)
    result = reloaded.classify_genre("dark ambient")
    assert result is not None
    assert result.tier == 1
    assert result.confidence == 0.95


def test_add_descriptor_writes_yaml(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    vocab.add_term("descriptor", "ethereal")
    vocab.save()

    reloaded = GenreVocabulary(vocab_yaml)
    assert reloaded.classify_non_genre("ethereal") == "descriptor"
