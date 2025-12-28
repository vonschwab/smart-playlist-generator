from __future__ import annotations

from typing import List, Optional

from src.features.artifacts import ArtifactBundle


def dim_labels(bundle: ArtifactBundle) -> List[str]:
    names = getattr(bundle, "sonic_feature_names", None)
    if names is not None:
        return [str(x) for x in names]
    dim = getattr(bundle, "X_sonic", None)
    length = int(dim.shape[1]) if dim is not None else 0
    return [f"dim_{i:02d}" for i in range(length)]


def dim_label(bundle: ArtifactBundle, index: int) -> str:
    labels = dim_labels(bundle)
    if 0 <= index < len(labels):
        return labels[index]
    return f"dim_{index:02d}"


def dim_units(bundle: ArtifactBundle) -> List[str]:
    units = getattr(bundle, "sonic_feature_units", None)
    if units is not None:
        return [str(x) for x in units]
    dim = getattr(bundle, "X_sonic", None)
    length = int(dim.shape[1]) if dim is not None else 0
    return ["" for _ in range(length)]
