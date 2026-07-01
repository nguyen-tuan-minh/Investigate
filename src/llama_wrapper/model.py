"""Behavior-preserving wrappers for LLaMA modules.

These wrappers intentionally match the original modules for now. Later, this is
where rotation, quantization, and shadow-path logic can be inserted.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class WrappedLinear(nn.Module):
    """Pass-through wrapper around ``nn.Linear``.

    The wrapper keeps the original linear module intact so its weights, bias,
    dtype, device, and state-dict entries remain owned by the original module.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"WrappedLinear expected nn.Linear, got {type(base)!r}")

        self.base = base

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.base(input)

    def extra_repr(self) -> str:
        return self.base.extra_repr()


class WrappedModule(nn.Module):
    """Generic pass-through wrapper for a LLaMA submodule.

    Use this when you want to wrap a larger module, such as attention or MLP,
    while still preserving the original forward behavior exactly.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base(*args, **kwargs)


__all__ = ["WrappedLinear", "WrappedModule"]
