from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.playlist.layered_genre_scoring import score_layered_transition


def build_layered_transition_diagnostics(
    *,
    bundle: Any,
    track_indices: Sequence[int],
    edge_scores: Sequence[dict[str, Any]],
    mode: str,
    enabled: bool,
    sample_limit: int = 25,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "reason": "disabled",
        }

    matrices = {
        "leaf": getattr(bundle, "X_genre_leaf_idf", None),
        "family": getattr(bundle, "X_genre_family", None),
        "bridge": getattr(bundle, "X_genre_bridge", None),
        "facet": getattr(bundle, "X_facet", None),
    }
    if any(matrix is None for matrix in matrices.values()):
        return {
            "enabled": False,
            "reason": "missing_layered_matrices",
        }

    leaf = np.asarray(matrices["leaf"], dtype=float)
    family = np.asarray(matrices["family"], dtype=float)
    bridge = np.asarray(matrices["bridge"], dtype=float)
    facet = np.asarray(matrices["facet"], dtype=float)
    if any(matrix.ndim != 2 for matrix in (leaf, family, bridge, facet)):
        return {
            "enabled": False,
            "reason": "layered_matrix_shape_mismatch",
        }
    if bridge.shape[1] != leaf.shape[1]:
        return {
            "enabled": False,
            "reason": "bridge_leaf_vocab_mismatch",
        }

    indices = [int(i) for i in track_indices]
    if len(indices) < 2:
        return {
            "enabled": True,
            "edge_count": 0,
            "explained_count": 0,
            "unexplained_count": 0,
            "reason_counts": {},
            "samples": [],
        }

    max_idx = max(indices)
    if any(matrix.shape[0] <= max_idx for matrix in (leaf, family, bridge, facet)):
        return {
            "enabled": False,
            "reason": "layered_matrix_track_mismatch",
        }

    track_ids = getattr(bundle, "track_ids", None)
    leaf_vocab = _coerce_vocab(getattr(bundle, "genre_leaf_vocab", None), leaf.shape[1])
    family_vocab = _coerce_vocab(getattr(bundle, "genre_family_vocab", None), family.shape[1])
    bridge_vocab = _coerce_vocab(getattr(bundle, "genre_bridge_vocab", None), bridge.shape[1])
    facet_vocab = _coerce_vocab(getattr(bundle, "facet_vocab", None), facet.shape[1])
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    explained_count = 0
    bridge_supported_count = 0
    edge_count = len(indices) - 1

    for edge_pos in range(edge_count):
        prev_idx = indices[edge_pos]
        cur_idx = indices[edge_pos + 1]
        edge = edge_scores[edge_pos] if edge_pos < len(edge_scores) and isinstance(edge_scores[edge_pos], dict) else {}
        sonic_similarity = _coerce_score(edge.get("S"), fallback=edge.get("T"), default=0.0)
        transition_quality = _coerce_score(edge.get("T"), fallback=edge.get("S"), default=0.0)
        decision = score_layered_transition(
            from_leaf=leaf[prev_idx],
            to_leaf=leaf[cur_idx],
            from_family=family[prev_idx],
            to_family=family[cur_idx],
            from_bridge=bridge[prev_idx],
            to_bridge=bridge[cur_idx],
            from_facet=facet[prev_idx],
            to_facet=facet[cur_idx],
            sonic_similarity=sonic_similarity,
            transition_quality=transition_quality,
            mode=mode,
        )
        if decision.explained:
            explained_count += 1
        if decision.reason == "bridge_supported":
            bridge_supported_count += 1
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        if len(rows) < sample_limit:
            rows.append(
                {
                    "edge_index": int(edge_pos),
                    "from_index": int(prev_idx),
                    "to_index": int(cur_idx),
                    "from_track_id": _track_id(track_ids, prev_idx),
                    "to_track_id": _track_id(track_ids, cur_idx),
                    "reason": decision.reason,
                    "explained": bool(decision.explained),
                    "score": round(float(decision.score), 6),
                    "local_family_continuity": round(float(decision.components.local_family_continuity), 6),
                    "local_leaf_continuity": round(float(decision.components.local_leaf_continuity), 6),
                    "bridge_edge_bonus": round(float(decision.components.bridge_edge_bonus), 6),
                    "facet_continuity": round(float(decision.components.facet_continuity), 6),
                    "sonic_similarity": round(float(decision.components.sonic_similarity), 6),
                    "transition_quality": round(float(decision.components.transition_quality), 6),
                    "unexplained_family_jump_penalty": round(
                        float(decision.components.unexplained_family_jump_penalty), 6
                    ),
                    "bridge_evidence_penalty": round(float(decision.components.bridge_evidence_penalty), 6),
                    "from_leaf_terms": _active_terms(leaf[prev_idx], leaf_vocab),
                    "to_leaf_terms": _active_terms(leaf[cur_idx], leaf_vocab),
                    "shared_leaf_terms": _shared_terms(leaf[prev_idx], leaf[cur_idx], leaf_vocab),
                    "from_family_terms": _active_terms(family[prev_idx], family_vocab),
                    "to_family_terms": _active_terms(family[cur_idx], family_vocab),
                    "shared_family_terms": _shared_terms(family[prev_idx], family[cur_idx], family_vocab),
                    "from_bridge_terms": _active_terms(bridge[prev_idx], bridge_vocab),
                    "to_bridge_terms": _active_terms(bridge[cur_idx], bridge_vocab),
                    "shared_bridge_terms": _shared_terms(bridge[prev_idx], bridge[cur_idx], bridge_vocab),
                    "from_facet_terms": _active_terms(facet[prev_idx], facet_vocab),
                    "to_facet_terms": _active_terms(facet[cur_idx], facet_vocab),
                    "shared_facet_terms": _shared_terms(facet[prev_idx], facet[cur_idx], facet_vocab),
                }
            )

    return {
        "enabled": True,
        "mode": mode,
        "edge_count": int(edge_count),
        "explained_count": int(explained_count),
        "unexplained_count": int(edge_count - explained_count),
        "bridge_supported_count": int(bridge_supported_count),
        "reason_counts": reason_counts,
        "samples": rows,
    }


def _coerce_score(value: Any, *, fallback: Any, default: float) -> float:
    raw = value if isinstance(value, (int, float)) else fallback
    if not isinstance(raw, (int, float)):
        return float(default)
    if raw != raw:
        return float(default)
    return max(0.0, min(1.0, float(raw)))


def _coerce_vocab(vocab: Any, width: int) -> list[str] | None:
    if vocab is None:
        return None
    try:
        values = [str(value) for value in np.asarray(vocab, dtype=object).reshape(-1).tolist()]
    except Exception:
        return None
    if len(values) != int(width):
        return None
    return values


def _active_terms(vector: np.ndarray, vocab: list[str] | None) -> list[str]:
    if vocab is None:
        return []
    values = np.asarray(vector, dtype=float).reshape(-1)
    return [vocab[i] for i, value in enumerate(values) if i < len(vocab) and float(value) > 0.0]


def _shared_terms(left: np.ndarray, right: np.ndarray, vocab: list[str] | None) -> list[str]:
    if vocab is None:
        return []
    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    width = min(len(vocab), left_values.shape[0], right_values.shape[0])
    return [
        vocab[i]
        for i in range(width)
        if float(left_values[i]) > 0.0 and float(right_values[i]) > 0.0
    ]


def _track_id(track_ids: Any, idx: int) -> str:
    if track_ids is None:
        return str(idx)
    try:
        return str(track_ids[int(idx)])
    except Exception:
        return str(idx)
