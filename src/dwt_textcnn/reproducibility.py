"""Randomness controls and environment capture for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


TRACKED_PACKAGES = (
    "gensim",
    "nltk",
    "numpy",
    "pandas",
    "psutil",
    "PyWavelets",
    "PyYAML",
    "scikit-learn",
    "torch",
)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed supported random generators and configure PyTorch determinism."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def capture_environment() -> dict[str, Any]:
    """Return JSON-serializable software and hardware provenance."""

    packages: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"

    cuda_available = torch.cuda.is_available()
    cuda_devices = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
    }
