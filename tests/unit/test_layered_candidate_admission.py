import numpy as np
import yaml

from src.config_loader import Config
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig
from src.playlist_generator import build_ds_overrides


def _cfg() -> CandidatePoolConfig:
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
    )


def _layered_matrices():
    # Rows: seed, specific leaf match, broad-only family match, bridge-supported match.
    # Leaf dimensions: [jangle pop, synth-pop]
    X_leaf = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    X_family = np.ones((4, 1), dtype=float)
    X_bridge = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    X_facet = np.array(
        [
            [1.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=float,
    )
    return X_leaf, X_family, X_bridge, X_facet


def _layered_vocabs():
    return {
        "genre_leaf_vocab": ["jangle pop", "synth-pop"],
        "genre_family_vocab": ["pop"],
        "genre_bridge_vocab": ["jangle pop", "synth-pop"],
        "facet_vocab": ["reverb-heavy"],
    }


def _run_pool(*, genre_graph_source: str, include_vocabs: bool = False, mode: str = "strict", **extra):
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed", "leaf_match", "broad_only", "bridge"])
    track_ids = np.array(["seed-id", "leaf-id", "broad-id", "bridge-id"])
    X_leaf, X_family, X_bridge, X_facet = _layered_matrices()
    kwargs = {}
    if include_vocabs:
        kwargs.update(_layered_vocabs())
    return build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=_cfg(),
        random_seed=0,
        mode=mode,
        genre_graph_source=genre_graph_source,
        X_genre_leaf_idf=X_leaf,
        X_genre_family=X_family,
        X_genre_bridge=X_bridge,
        X_facet=X_facet,
        **kwargs,
        **extra,
    )


def test_layered_shadow_does_not_change_candidate_admission():
    legacy = _run_pool(genre_graph_source="legacy")
    shadow = _run_pool(genre_graph_source="layered_shadow")

    assert shadow.pool_indices.tolist() == legacy.pool_indices.tolist()
    assert shadow.eligible_indices.tolist() == legacy.eligible_indices.tolist()
    assert shadow.stats["layered_genre_shadow"]["enabled"] is True
    assert shadow.stats["layered_genre_admission"]["source"] == "layered_shadow"
    assert shadow.stats["layered_genre_admission"]["applied"] is False


def test_layered_admission_filters_broad_only_strict_candidate():
    result = _run_pool(genre_graph_source="layered")

    kept_track_ids = set(result.stats["layered_genre_admission"]["admitted_track_ids"])
    assert kept_track_ids == {"leaf-id", "bridge-id"}
    assert set(result.pool_indices.tolist()) == {1, 3}
    assert set(result.eligible_indices.tolist()) == {1, 3}
    assert result.stats["layered_genre_admission"]["applied"] is True
    assert result.stats["layered_genre_admission"]["rejected_count"] == 1
    assert result.stats["layered_genre_admission"]["rejection_reason_counts"] == {
        "broad_only_without_leaf_support": 1
    }


def test_layered_shadow_samples_include_named_genre_graph_terms():
    result = _run_pool(genre_graph_source="layered_shadow", include_vocabs=True)

    samples = {
        row["track_id"]: row
        for row in result.stats["layered_genre_shadow"]["samples"]
    }

    broad_only = samples["broad-id"]
    assert broad_only["reason"] == "broad_only_without_leaf_support"
    assert broad_only["candidate_family_terms"] == ["pop"]
    assert broad_only["candidate_leaf_terms"] == []
    assert broad_only["shared_family_terms"] == ["pop"]

    bridge = samples["bridge-id"]
    assert bridge["reason"] == "bridge_supported"
    assert bridge["candidate_leaf_terms"] == ["synth-pop"]
    assert bridge["shared_facet_terms"] == ["reverb-heavy"]
    assert bridge["seed_bridge_terms"] == ["synth-pop"]
    assert bridge["candidate_bridge_terms"] == ["jangle pop"]


def test_layered_rejected_samples_include_named_genre_graph_terms():
    result = _run_pool(genre_graph_source="layered", include_vocabs=True)

    rejected = result.stats["layered_genre_admission"]["rejected_samples"]

    assert rejected == [
        {
            **rejected[0],
            "track_id": "broad-id",
            "reason": "broad_only_without_leaf_support",
            "candidate_family_terms": ["pop"],
            "candidate_leaf_terms": [],
            "shared_family_terms": ["pop"],
        }
    ]


def test_layered_mode_ignores_legacy_flat_genre_gate():
    # Flat legacy vectors say the bridge candidate has no genre overlap with the seed.
    # Layered graph evidence says it is an explained jangle-pop -> synth-pop bridge.
    # In layered mode, the graph is the genre substrate; the flat gate must not
    # remove graph-valid candidates before layered admission sees them.
    flat = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    result = _run_pool(
        genre_graph_source="layered",
        mode="narrow",
        min_genre_similarity=0.95,
        X_genre_raw=flat,
        X_genre_smoothed=flat,
        genre_vocab=["old-seed-genre", "old-bridge-genre"],
    )

    assert "bridge-id" in result.stats["layered_genre_admission"]["admitted_track_ids"]
    assert result.stats["layered_genre_admission"]["legacy_flat_genre_gate_applied"] is False


def test_layered_source_without_complete_matrices_reports_non_genre_fallback():
    emb = np.array([[1.0, 0.0]] * 2, dtype=float)
    artist_keys = np.array(["seed", "candidate"])

    result = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        cfg=_cfg(),
        random_seed=0,
        mode="strict",
        genre_graph_source="layered",
        X_genre_leaf_idf=np.zeros((2, 1), dtype=float),
    )

    assert result.pool_indices.tolist() == [1]
    assert result.stats["layered_genre_admission"] == {
        "source": "layered",
        "applied": False,
        "reason": "missing_layered_matrices",
        "legacy_flat_genre_gate_applied": False,
    }


def test_ds_overrides_preserve_genre_graph_source():
    overrides = build_ds_overrides(
        {
            "genre_graph": {"source": "layered"},
            "genre_source": "layered_shadow",
            "candidate_pool": {"max_pool_size": 123},
        }
    )

    assert overrides["genre_graph"] == {"source": "layered"}
    assert overrides["genre_source"] == "layered_shadow"
    assert overrides["candidate_pool"]["max_pool_size"] == 123


def test_config_tuning_dict_exposes_genre_graph_source(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "library": {"database_path": "data/metadata.db"},
                "playlists": {
                    "cohesion_mode": "dynamic",
                    "ds_pipeline": {
                        "genre_graph": {"source": "layered"},
                        "genre_source": "layered_shadow",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_path))
    tuning = cfg.get_ds_tuning_dict()

    assert cfg.get_ds_genre_graph_source() == "layered"
    assert tuning["genre_graph"]["source"] == "layered"
    assert tuning["genre_source"] == "layered_shadow"
