"""One-level discrete wavelet compression for embedding matrices."""

from __future__ import annotations

import numpy as np
import pywt


def compress_dwt(
    matrix: np.ndarray,
    *,
    wavelet: str = "coif1",
    mode: str = "periodization",
    coefficient: str = "cA",
) -> np.ndarray:
    """Apply a one-level row-wise DWT and retain one coefficient matrix."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("DWT input must be a non-empty two-dimensional matrix")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("DWT input must contain only finite floating-point values")
    if coefficient not in {"cA", "cD"}:
        raise ValueError("DWT coefficient must be 'cA' or 'cD'")

    approximation, detail = pywt.dwt(values, wavelet, mode=mode, axis=1)
    selected = approximation if coefficient == "cA" else detail
    result = np.ascontiguousarray(selected, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("DWT output contains non-finite values")
    return result
