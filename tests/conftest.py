import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny64():
    """Tiny random timm ViT in float64 with randomised affines and LayerScale."""
    from mirror.adapters import TimmTinyAdapter

    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)


@pytest.fixture
def images64():
    g = torch.Generator().manual_seed(0)
    return torch.randn(3, 3, 56, 56, generator=g, dtype=torch.float64)
