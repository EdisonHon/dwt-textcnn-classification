from __future__ import annotations

import numpy as np
import pywt
import pytest

from dwt_textcnn.compression.dwt import compress_dwt


@pytest.mark.parametrize("coefficient,index", [("cA", 0), ("cD", 1)])
def test_dwt_matches_pywavelets_and_halves_300_dimensions(
    coefficient: str, index: int
) -> None:
    matrix = np.arange(900, dtype=np.float32).reshape(3, 300) / 100.0

    result = compress_dwt(matrix, coefficient=coefficient)
    expected = pywt.dwt(matrix, "coif1", mode="periodization", axis=1)[index]

    assert result.shape == (3, 150)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected.astype(np.float32))


@pytest.mark.parametrize(
    "matrix,coefficient",
    [
        (np.zeros(3, dtype=np.float32), "cA"),
        (np.asarray([[np.nan]], dtype=np.float32), "cA"),
        (np.zeros((1, 300), dtype=np.float32), "invalid"),
    ],
)
def test_dwt_rejects_invalid_input(matrix: np.ndarray, coefficient: str) -> None:
    with pytest.raises(ValueError):
        compress_dwt(matrix, coefficient=coefficient)
