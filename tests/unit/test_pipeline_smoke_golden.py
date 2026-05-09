"""
Pipeline override-parsing golden tests — the safety net for the Tier-1.5 split.

The pier_bridge_overrides parsing block inside generate_playlist_ds is ~600 LOC
of nested if-isinstance translation from a user-facing dict into a typed
PierBridgeConfig. PR-6 of the split extracts that block; the goldens here
catch any silent behavior drift.

Strategy: monkeypatch ``build_pier_bridge_playlist`` to capture its ``cfg``
argument (the resolved PierBridgeConfig) and raise a sentinel so the rest of
the pipeline doesn't need to succeed. This makes the test independent of
candidate-pool feasibility and pier-bridge construction logic — it only
exercises the orchestrator's prep + override-parsing phases.

For each scenario in SCENARIOS, the resolved cfg is reduced to a JSON-comparable
dict and compared against tests/unit/goldens/pipeline/<scenario>.json. First
run on a new scenario writes the baseline and skips; subsequent runs fail on
drift.

To re-baseline after an INTENTIONAL behavior change, delete the relevant
golden file and re-run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pytest


GOLDEN_DIR = Path(__file__).parent / "goldens" / "pipeline"


class _CaptureSentinel(Exception):
    """Raised by the monkeypatched build_pier_bridge_playlist after capture."""


def _build_smoke_fixture(tmp_path):
    """Synthetic 30-track artifact, deterministic from seed=0.

    Small enough to load fast, large enough that bundle-restriction phases
    have something to slice. The override parser does not actually consume
    these matrices — they're here so generate_playlist_ds reaches the
    pier-bridge call site (which we then short-circuit).
    """
    rng = np.random.default_rng(0)
    N = 30
    track_ids = np.array([f"t{i}" for i in range(N)])
    artist_keys = np.array([f"a{i % 6}" for i in range(N)])
    track_artists = np.array([f"Artist {i % 6}" for i in range(N)])
    track_titles = np.array([f"Song {i}" for i in range(N)])
    X_sonic = rng.standard_normal((N, 16))
    G = 8
    X_genre_raw = (rng.random((N, G)) > 0.7).astype(float)
    X_genre_smoothed = X_genre_raw + 0.05 * rng.standard_normal((N, G))
    genre_vocab = np.array([f"g{i}" for i in range(G)])
    durations_ms = np.full(N, 200_000, dtype=np.int64)

    artifact_path = tmp_path / "smoke_artifact.npz"
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        durations_ms=durations_ms,
    )
    return artifact_path


SCENARIOS = {
    # Baseline: dynamic mode, no overrides — exercises the default code path.
    "dynamic_default": {
        "mode": "dynamic",
        "num_tracks": 8,
        "overrides": {},
    },
    # Narrow mode with pier_bridge overrides — exercises override parsing
    # for floors, weights, and the progress/genre sub-blocks.
    "narrow_with_pier_bridge_overrides": {
        "mode": "narrow",
        "num_tracks": 6,
        "overrides": {
            "pier_bridge": {
                "bridge_floor_narrow": 0.04,
                "weight_bridge_narrow": 0.65,
                "weight_transition_narrow": 0.35,
                "soft_genre_penalty_threshold": 0.25,
                "soft_genre_penalty_strength": 0.12,
                "genre_tiebreak_weight": 0.06,
                "progress": {
                    "enabled": True,
                    "monotonic_epsilon": 0.06,
                    "penalty_weight": 0.18,
                },
                "genre": {"tie_break_band": 0.025},
                "segment_pool_strategy": "segment_scored",
                "segment_pool_max": 350,
                "max_segment_pool_max": 1100,
            }
        },
    },
    # Discover mode with full DJ-bridging override — the gnarly nested parser.
    # This is the load-bearing snapshot for the PR-6 (pier_bridge_overrides)
    # extraction; it touches almost every branch of the dj_bridging parser.
    "discover_with_dj_bridging": {
        "mode": "discover",
        "num_tracks": 8,
        "overrides": {
            "pier_bridge": {
                "dj_bridging": {
                    "enabled": True,
                    "seed_ordering": "auto",
                    "route_shape": "ladder",
                    "anchors": {"must_include_all": True},
                    "waypoint_weight": 0.18,
                    "waypoint_floor": 0.22,
                    "waypoint_penalty": 0.11,
                    "waypoint_cap": 0.08,
                    "pooling": {
                        "strategy": "dj_union",
                        "k_local": 180,
                        "k_toward": 70,
                        "k_genre": 40,
                        "k_union_max": 800,
                    },
                    "ladder": {
                        "top_labels": 6,
                        "min_label_weight": 0.04,
                        "min_similarity": 0.18,
                        "max_steps": 5,
                    },
                    "connector_bias": {
                        "enabled": True,
                        "max_per_segment_linear": 1,
                        "max_per_segment_adventurous": 2,
                    },
                    "far_thresholds": {
                        "sonic": 0.40,
                        "genre": 0.55,
                        "connector_scarcity": 0.12,
                    },
                    "dj_genre_use_idf": True,
                    "dj_genre_idf_power": 1.0,
                    "dj_genre_use_coverage": True,
                    "dj_genre_coverage_top_k": 6,
                    "dj_genre_coverage_weight": 0.13,
                    "dj_waypoint_squash": "tanh",
                    "dj_waypoint_squash_alpha": 4.0,
                    "dj_waypoint_delta_mode": "centered",
                    "dj_coverage_mode": "weighted",
                    "dj_coverage_presence_source": "raw",
                }
            }
        },
    },
    # Progress arc + experiment overrides (dry_run unlocks experiments).
    "narrow_progress_arc_dry_run": {
        "mode": "narrow",
        "num_tracks": 6,
        "dry_run": True,
        "overrides": {
            "pier_bridge": {
                "progress_arc": {
                    "enabled": True,
                    "weight": 0.22,
                    "shape": "arc",
                    "tolerance": 0.05,
                    "loss": "huber",
                    "huber_delta": 0.08,
                    "max_step": 0.40,
                    "max_step_mode": "penalty",
                    "max_step_penalty": 0.28,
                    "autoscale": {
                        "enabled": True,
                        "min_distance": 0.06,
                        "distance_scale": 0.45,
                        "per_step_scale": True,
                    },
                },
                "experiments": {
                    "bridge_scoring": {
                        "enabled": True,
                        "min_weight": 0.55,
                        "balance_weight": 0.30,
                    }
                },
            }
        },
    },
}


def _serialize_pb_cfg(pb_cfg):
    """PierBridgeConfig (a frozen dataclass) -> JSON-friendly dict."""
    if pb_cfg is None:
        return None
    if is_dataclass(pb_cfg):
        return {k: _normalize_value(v) for k, v in asdict(pb_cfg).items()}
    return _normalize_value(pb_cfg)


def _normalize_value(v):
    """Convert non-JSON-able primitives into JSON-able ones."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_value(x) for k, x in v.items()}
    return v


@pytest.mark.parametrize("scenario_name", sorted(SCENARIOS.keys()))
def test_pier_bridge_config_matches_golden(scenario_name, tmp_path, monkeypatch):
    """
    Override-parsing golden: capture the resolved PierBridgeConfig that
    pipeline hands to build_pier_bridge_playlist, compare against the
    checked-in JSON snapshot.

    First run on a new scenario writes the baseline and skips; subsequent
    runs hard-fail on drift. This is the load-bearing safety net for the
    Tier-1.5 pipeline.py split — every extraction PR re-runs this and
    confirms the override parser still produces the same cfg.
    """
    scenario = SCENARIOS[scenario_name]

    captured: dict = {}

    def _capture_and_short_circuit(*args, **kwargs):
        captured["pb_cfg"] = kwargs.get("cfg")
        raise _CaptureSentinel("captured")

    # Patch the call site inside pipeline (the orchestrator) so we intercept
    # the resolved cfg before pier-bridge construction runs.
    import src.playlist.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "build_pier_bridge_playlist", _capture_and_short_circuit)

    artifact_path = _build_smoke_fixture(tmp_path)

    with pytest.raises(_CaptureSentinel):
        pipeline_mod.generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            num_tracks=scenario["num_tracks"],
            mode=scenario["mode"],
            random_seed=0,
            overrides=scenario["overrides"],
            dry_run=scenario.get("dry_run", False),
        )

    pb_cfg = captured.get("pb_cfg")
    assert pb_cfg is not None, "build_pier_bridge_playlist was never called"
    snapshot = _serialize_pb_cfg(pb_cfg)

    golden_path = GOLDEN_DIR / f"{scenario_name}.json"
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        pytest.skip(f"Created golden baseline at {golden_path}; rerun to verify.")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert snapshot == expected, (
        f"Golden mismatch for {scenario_name}.\n"
        f"  golden file: {golden_path}\n"
        f"  Diff first 2 differing keys:\n"
        + "\n".join(
            f"    {k}: expected={expected.get(k)!r} got={snapshot.get(k)!r}"
            for k in sorted(set(snapshot) | set(expected))
            if snapshot.get(k) != expected.get(k)
        )[:1500]
    )
