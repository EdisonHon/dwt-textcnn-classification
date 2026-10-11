from __future__ import annotations

import numpy as np
import pytest

from dwt_textcnn.compression.pca import fit_pca, transform_pca


def test_full_svd_pca_is_repeatable_and_returns_float32() -> None:
    matrix = np.random.default_rng(13).normal(size=(200, 300)).astype(np.float32)

    first = fit_pca(matrix, n_components=150)
    second = fit_pca(matrix, n_components=150)
    first_output = transform_pca(first, matrix)
    second_output = transform_pca(second, matrix)

    assert first_output.shape == (200, 150)
    assert first_output.dtype == np.float32
    np.testing.assert_array_equal(first.components_, second.components_)
    np.testing.assert_array_equal(first_output, second_output)


def test_pca_rejects_too_many_components() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_pca(np.zeros((10, 20), dtype=np.float32), n_components=11)
