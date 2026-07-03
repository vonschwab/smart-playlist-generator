"""Golden-fixture capture + replay for the lossless-speedup work.

Captures the EXACT kwargs entering ds_pipeline_runner.generate_playlist_ds
(the outermost deterministic seam: candidate-pool RNG is seeded downstream,
the beam is RNG-free) so a replay reproduces the identical playlist. Used by
the bit-diff regression gate in tests/integration/test_lossless_speedup_golden.py.

See docs/superpowers/specs/2026-07-03-lossless-generation-speedup-design.md.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return {"__set__": sorted(str(x) for x in obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, "fields": dataclasses.asdict(obj)}
    raise TypeError(
        f"Golden capture: non-serializable arg of type {type(obj)!r}; extend _json_default"
    )


def _decode(obj: Dict[str, Any]) -> Any:
    if "__set__" in obj:
        return set(obj["__set__"])
    return obj  # __dataclass__ reconstruction handled in replay if ever needed


def dump_golden_inputs(
    kwargs: Dict[str, Any],
    track_ids,
    min_transition,
    mean_transition,
    path: str,
) -> None:
    payload = {
        "kwargs": kwargs,
        "track_ids": list(track_ids),
        "min_transition": None if min_transition is None else float(min_transition),
        "mean_transition": None if mean_transition is None else float(mean_transition),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default, indent=2)


def load_golden(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh, object_hook=_decode)


def _reconstruct_pier_bridge_config(payload: Dict[str, Any]):
    """Rebuild a PierBridgeConfig from a captured asdict payload.

    Coerces JSON lists back to tuples for tuple-typed fields and keeps only
    fields the current class still defines (a new class field falls to its
    default — correct for reproducing a run captured before it existed).
    """
    import dataclasses

    from src.playlist.pier_bridge_builder import PierBridgeConfig

    fields = {f.name: f for f in dataclasses.fields(PierBridgeConfig)}
    kwargs: Dict[str, Any] = {}
    for name, value in payload["fields"].items():
        field = fields.get(name)
        if field is None:
            continue  # tolerate a field the class no longer has
        if isinstance(field.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return PierBridgeConfig(**kwargs)


def replay_golden(golden: Dict[str, Any], config_path: str = "config.yaml"):
    from src.config_loader import Config
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    # Reproduce the process-wide config side-effects the app performs at load
    # time (notably artifacts.sonic_variant_override -> the artifact loader),
    # which a bare generate_playlist_ds call would otherwise skip. Without this
    # the muq-native artifact fails _require_keys(['X_sonic']).
    Config(config_path)

    kwargs = dict(golden["kwargs"])
    pbc = kwargs.get("pier_bridge_config")
    if isinstance(pbc, dict) and pbc.get("__dataclass__") == "PierBridgeConfig":
        kwargs["pier_bridge_config"] = _reconstruct_pier_bridge_config(pbc)
    return generate_playlist_ds(**kwargs)
