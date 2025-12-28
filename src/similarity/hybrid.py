from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class HybridEmbeddingModel:
    embedding: np.ndarray  # (N, D)
    sonic_pca: Optional[np.ndarray]  # (N, d1)
    genre_pca: Optional[np.ndarray]  # (N, d2)
    params_effective: dict


def _fit_pca(matrix: np.ndarray, n_components: int, seed: int, *, pre_scaled: bool = False) -> np.ndarray:
    if not pre_scaled:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(matrix)


def build_hybrid_embedding(
    X_sonic: np.ndarray,
    X_genre: np.ndarray,
    *,
    n_components_sonic: int = 32,
    n_components_genre: int = 32,
    w_sonic: float = 1.0,
    w_genre: float = 1.0,
    random_seed: int = 0,
    pre_scaled_sonic: bool = False,
    pre_scaled_genre: bool = False,
    use_pca_sonic: bool = True,
    use_pca_genre: bool = True,
) -> HybridEmbeddingModel:
    """
    StandardScaler -> PCA separately -> concat [w_sonic*sonic_pca, w_genre*genre_pca].
    Cap n_components at min(input_dim, requested).
    Use deterministic PCA (random_state=random_seed).
    Return params_effective that includes the capped component counts.
    """
    sonic_components = min(n_components_sonic, X_sonic.shape[1], X_sonic.shape[0])
    genre_components = min(n_components_genre, X_genre.shape[1], X_genre.shape[0])

    if use_pca_sonic:
        E_sonic = _fit_pca(X_sonic, sonic_components, random_seed, pre_scaled=pre_scaled_sonic)
    else:
        if pre_scaled_sonic:
            E_sonic = X_sonic
        else:
            scaler = StandardScaler()
            E_sonic = scaler.fit_transform(X_sonic)

    if use_pca_genre:
        E_genre = _fit_pca(X_genre, genre_components, random_seed, pre_scaled=pre_scaled_genre)
    else:
        if pre_scaled_genre:
            E_genre = X_genre
        else:
            scaler = StandardScaler()
            E_genre = scaler.fit_transform(X_genre)

    embedding = np.concatenate([w_sonic * E_sonic, w_genre * E_genre], axis=1)
    params_effective = {
        "n_components_sonic": sonic_components,
        "n_components_genre": genre_components,
        "w_sonic": w_sonic,
        "w_genre": w_genre,
        "pre_scaled_sonic": bool(pre_scaled_sonic),
        "pre_scaled_genre": bool(pre_scaled_genre),
        "use_pca_sonic": bool(use_pca_sonic),
        "use_pca_genre": bool(use_pca_genre),
    }
    return HybridEmbeddingModel(
        embedding=embedding,
        sonic_pca=E_sonic,
        genre_pca=E_genre,
        params_effective=params_effective,
    )


def cosine_sim_matrix_to_vector(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return cosine similarity of each row in X to v. Shape (N,). Deterministic."""
    v_norm = np.linalg.norm(v) + 1e-12
    row_norms = np.linalg.norm(X, axis=1) + 1e-12
    dots = X @ v
    return dots / (row_norms * v_norm)


def transition_similarity_end_to_start(
    X_end: np.ndarray,
    X_start: np.ndarray,
    prev_idx: int,
    cand_indices: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between end(prev) and start(cand). Returns (len(cand_indices),)."""
    prev_vec = X_end[prev_idx]
    prev_norm = np.linalg.norm(prev_vec) + 1e-12
    cand_mat = X_start[cand_indices]
    cand_norms = np.linalg.norm(cand_mat, axis=1) + 1e-12
    dots = cand_mat @ prev_vec
    return dots / (cand_norms * prev_norm)
