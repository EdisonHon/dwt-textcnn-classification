from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from dwt_textcnn.reproducibility import capture_environment, seed_everything


def _random_values() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_everything_repeats_random_streams() -> None:
    seed_everything(42)
    first = _random_values()
    seed_everything(42)
    second = _random_values()

    assert first == second


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_seed_everything_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_everything(seed)  # type: ignore[arg-type]


def test_environment_capture_is_json_serializable() -> None:
    environment = capture_environment()

    assert "python" in environment
    assert "torch" in environment["packages"]
    json.dumps(environment)
