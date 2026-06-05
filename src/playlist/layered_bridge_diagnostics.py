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


def _track_id(track_ids: Any, idx: int) -> str:
    if track_ids is None:
        return str(idx)
    try:
        return str(track_ids[int(idx)])
    except Exception:
        return str(idx)
