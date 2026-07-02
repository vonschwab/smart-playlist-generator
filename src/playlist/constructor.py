from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class PlaylistResult:
    track_indices: np.ndarray  # ordered indices, length L
    stats: Dict[str, Any]
    params_requested: Dict[str, Any]
    params_effective: Dict[str, Any]
