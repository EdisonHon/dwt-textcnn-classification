"""Deterministic PCA embedding compression baseline."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(
    matrix: np.ndarray,
    *,
    n_components: int = 150,
    whiten: bool = False,
    svd_solver: str = "full",
) -> PCA:
    """Fit PCA after validating the matrix and component count."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("PCA input must be a non-empty two-dimensional matrix")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("PCA input must contain only finite floating-point values")
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("PCA n_components must be a positive integer")
    if n_components > min(values.shape):
        raise ValueError(
            "PCA n_components cannot exceed the smaller input dimension"
        )

    model = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
    )
    model.fit(values)
    return model


def transform_pca(model: PCA, matrix: np.ndarray) -> np.ndarray:
    """Transform a complete vocabulary matrix into contiguous float32 output."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] != model.n_features_in_:
        raise ValueError("PCA transform input has an incompatible shape")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("PCA transform input must contain only finite values")
    result = np.ascontiguousarray(model.transform(values), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("PCA output contains non-finite values")
    return result
