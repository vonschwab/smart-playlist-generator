"""Pure selection helper for energy admission-rescue (pace as a co-equal axis)."""
from __future__ import annotations
from typing import Sequence
import numpy as np


def select_energy_rescue(
    arousal: np.ndarray,
    source_indices: Sequence[int],
    k_energy: int,
) -> list[int]:
    """Pick up to k_energy indices spanning the source's arousal range.

    `arousal` is a 1-D (z-scored) array indexed library-wide. `source_indices`
    are the rescue-eligible tracks (rhythm-rejected but genre+sonic-OK). Returns
    indices evenly spaced across their sorted arousal so low/mid/high are present.
    NaN-arousal indices are skipped. k_energy<=0 or empty source -> [].
    """
    if int(k_energy) <= 0:
        return []
    src = [int(i) for i in source_indices if np.isfinite(arousal[int(i)])]
    if not src:
        return []
    src.sort(key=lambda i: float(arousal[i]))
    if len(src) <= int(k_energy):
        return src
    pos = np.linspace(0, len(src) - 1, int(k_energy)).round().astype(int)
    return [src[j] for j in sorted(set(int(p) for p in pos))]
