"""Smoke tests for WrappedModule.

This file is intentionally runnable without pytest:
    python tests/test_wrapped_module.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_wrapper.model import WrappedModule
from llama_wrapper.rotation import CayleyRotation


def test_plain_linear_path() -> None:
    torch.manual_seed(0)

    base = nn.Linear(4, 4)
    wrapped = WrappedModule(base, path="l")
    sample = torch.randn(3, 4)

    expected = base(sample)
    actual = wrapped(sample)

    assert torch.allclose(actual, expected)


def test_rot_quant_linear_inverse_path_with_one_train_step() -> None:
    torch.manual_seed(0)

    base = nn.Linear(4, 4)
    rotation = CayleyRotation(dim=4, init_scale=0.01)
    wrapped = WrappedModule(
        base,
        path="rqli",
        rotation=rotation,
        quantization_bits=4,
        quantization_mode="asymmetric",
        use_rotation_cache=True,
    )

    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-2)
    sample = torch.randn(5, 4)
    target = torch.zeros_like(sample)

    rotation.clear_cache()
    assert rotation._cached_rotation is None

    output = wrapped(sample)
    assert output.shape == target.shape
    assert rotation._cached_rotation is not None

    cached_once = rotation.rotation_matrix(use_cache=True)
    cached_twice = rotation.rotation_matrix(use_cache=True)
    assert cached_once is cached_twice

    loss = (output - target).pow(2).mean()
    loss.backward()

    assert rotation.raw.grad is not None
    assert base.weight.grad is not None

    optimizer.step()
    rotation.clear_cache()
    assert rotation._cached_rotation is None

    refreshed = rotation.rotation_matrix(use_cache=True)
    assert rotation._cached_rotation is refreshed
    assert refreshed is not cached_once

    print("Input:")
    print(sample.detach())
    print("\nOutput:")
    print(output.detach())
    print("\nLoss:")
    print(loss.detach())
    print("\nCache reused before train step:")
    print(cached_once is cached_twice)
    print("\nCache refreshed after train step:")
    print(refreshed is not cached_once)


def test_wrapped_module_path() -> None:
    torch.manual_seed(0)

    base = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    rotation = CayleyRotation(dim=4, init_scale=0.01)
    wrapped = WrappedModule(
        base,
        path="rqli",
        rotation=rotation,
        quantization_bits=4,
        quantization_mode="asymmetric",
        use_rotation_cache=True,
    )
    sample = torch.randn(2, 4)

    rotation.clear_cache()
    output = wrapped(sample)

    assert output.shape == sample.shape
    assert rotation._cached_rotation is not None


if __name__ == "__main__":
    test_plain_linear_path()
    test_rot_quant_linear_inverse_path_with_one_train_step()
    test_wrapped_module_path()
    print("\nWrapper tests passed.")
