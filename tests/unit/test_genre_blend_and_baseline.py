from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from genre.blend import alpha_schedule, blend_with_prior
from scripts import measure_genre_baseline


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha_at_zero": -0.01}, "alpha_at_zero"),
        ({"alpha_at_zero": 1.01}, "alpha_at_zero"),
        ({"alpha_at_zero": np.nan}, "alpha_at_zero"),
        ({"alpha_at_zero": np.inf}, "alpha_at_zero"),
        ({"alpha_at_zero": -np.inf}, "alpha_at_zero"),
        ({"half_life": 0}, "half_life"),
        ({"half_life": -1}, "half_life"),
        ({"half_life": np.nan}, "half_life"),
        ({"half_life": np.inf}, "half_life"),
        ({"half_life": -np.inf}, "half_life"),
    ],
)
@pytest.mark.parametrize("function", [blend_with_prior, alpha_schedule])
def test_blend_functions_reject_invalid_parameters(function, kwargs, match):
    support = np.array([0, 1], dtype=np.float32)
    if function is blend_with_prior:
        corpus = np.eye(2, dtype=np.float32)
        prior = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match=match):
            function(corpus, prior, support, **kwargs)
    else:
        with pytest.raises(ValueError, match=match):
            function(support, **kwargs)


@pytest.mark.parametrize(
    "support",
    [
        np.array(1),
        np.array([[1, 2]]),
        np.array([0, -1]),
        np.array([0, np.nan]),
        np.array([0, np.inf]),
        np.array([0, -np.inf]),
    ],
)
@pytest.mark.parametrize("function", [blend_with_prior, alpha_schedule])
def test_blend_functions_reject_invalid_support(function, support):
    if function is blend_with_prior:
        corpus = np.eye(2, dtype=np.float32)
        prior = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match="support"):
            function(corpus, prior, support)
    else:
        with pytest.raises(ValueError, match="support"):
            function(support)


def test_alpha_schedule_rejects_empty_support_with_actionable_error():
    with pytest.raises(ValueError, match="support must not be empty"):
        alpha_schedule(np.array([], dtype=np.float32))


def test_blend_with_prior_rejects_empty_vocabulary_with_actionable_error():
    empty_embedding = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="embeddings must contain vocabulary rows"):
        blend_with_prior(
            empty_embedding,
            empty_embedding,
            np.array([], dtype=np.float32),
        )


@pytest.mark.parametrize("embedding_name", ["corpus", "prior"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_blend_with_prior_rejects_non_finite_embeddings(embedding_name, bad_value):
    corpus = np.eye(2, dtype=np.float32)
    prior = np.eye(2, dtype=np.float32)
    embedding = corpus if embedding_name == "corpus" else prior
    embedding[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match=rf"{embedding_name}_emb must contain only finite values",
    ):
        blend_with_prior(corpus, prior, np.array([0, 1], dtype=np.float32))


class _TrackingReader(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class _StreamingArtifact:
    def __init__(self, value: bytes) -> None:
        self.reader = _TrackingReader(value)

    def open(self, mode: str):
        assert mode == "rb"
        return self.reader

    def read_bytes(self) -> bytes:
        raise AssertionError("artifact hashing must not load the entire file")


def test_sha256_file_streams_artifact_in_chunks():
    content = b"dense genre artifact" * 100_000
    artifact = _StreamingArtifact(content)

    digest = measure_genre_baseline._sha256_file(artifact)

    assert digest == hashlib.sha256(content).hexdigest()
    assert artifact.reader.read_sizes
    assert all(size > 0 for size in artifact.reader.read_sizes)
    assert max(artifact.reader.read_sizes) < len(content)


def _case(name: str, *, error: str | None = None, mean: float = 0.25):
    if error is not None:
        return {"name": name, "error": error}
    method = {
        "distribution": {
            "min": 0.0,
            "p50": 0.1,
            "p90": 0.2,
            "p95": 0.3,
            "p99": 0.4,
            "max": 0.5,
            "mean": mean,
        },
        "candidates_at_floor": {"0.1": 3},
    }
    return {
        "name": name,
        "artist": "Artist",
        "title": "Title",
        "description": "Reference case",
        "track_id": "track-id",
        "artifact_idx": 1,
        "seed_genre_count": 2,
        "seed_l2_norm": 1.0,
        "active_genres": ["genre-a", "genre-b"],
        "cosine": method,
        "ensemble": method,
    }


def _snapshot(*cases):
    return {
        "artifact_path": "ignored-machine-specific-path",
        "artifact_sha256": "abc123",
        "artifact_shape": {"n_tracks": 10, "genre_vocab_size": 3},
        "floor_thresholds": [0.1],
        "cases": list(cases),
    }


def test_compare_reports_artifact_metadata_case_set_error_and_metric_differences(capsys):
    baseline = _snapshot(
        _case("removed"),
        _case("errored", error="old error"),
        _case("recovered", error="old error"),
        _case("metrics"),
    )
    current = _snapshot(
        _case("new"),
        _case("errored", error="new error"),
        _case("recovered"),
        _case("metrics", mean=0.75),
    )
    current["artifact_sha256"] = "def456"
    current["artifact_shape"] = {"n_tracks": 11, "genre_vocab_size": 4}

    with pytest.raises(SystemExit):
        measure_genre_baseline._compare(baseline, current)

    output = capsys.readouterr().out
    assert "artifact_sha256" in output
    assert "artifact_shape.n_tracks" in output
    assert "artifact_shape.genre_vocab_size" in output
    assert "REMOVED CASE: removed" in output
    assert "NEW CASE: new" in output
    assert "errored.error" in output
    assert "recovered.error" in output
    assert "metrics.cosine.distribution.mean" in output
