from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "dwt_textcnn.data",
        "dwt_textcnn.vocabulary",
        "dwt_textcnn.embeddings",
        "dwt_textcnn.compression.dwt",
        "dwt_textcnn.compression.pca",
        "dwt_textcnn.models.mccnn",
        "dwt_textcnn.training",
        "dwt_textcnn.evaluation",
        "dwt_textcnn.profiling",
        "dwt_textcnn.reproducibility",
    ],
)
def test_planned_module_is_importable(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
