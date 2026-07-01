"""Behavior-preserving wrappers for LLaMA modules.

These wrappers intentionally match the original modules for now. Later, this is
where rotation, quantization, and shadow-path logic can be inserted.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from .quantization import fake_quantize_ste


_PATH_OPS = {"r", "q", "l", "i"}
_UNIQUE_OPTIONAL_OPS = {"r", "q", "i"}


class WrappedLinear(nn.Module):
    """Wrapper around ``nn.Linear`` with compact transform paths.

    The wrapper keeps the original linear module intact so its weights, bias,
    dtype, device, and state-dict entries remain owned by the original module.

    Path symbols are interpreted left-to-right:
        r: apply trainable rotation
        q: apply fake quantization
        l: apply base linear layer
        i: apply inverse rotation

    Examples:
        l:    Linear(x)
        rql:  Linear(Q(R(x)))
        rqil: Linear(R^-1(Q(R(x))))
        rqli: R^-1(Linear(Q(R(x))))
    """

    def __init__(
        self,
        base: nn.Linear,
        path: str = "l",
        rotation: Optional[nn.Module] = None,
        quantization_bits: int = 8,
        quantization_mode: str = "asymmetric",
        quantization_eps: float = 1e-8,
        use_rotation_cache: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"WrappedLinear expected nn.Linear, got {type(base)!r}")

        self.base = base
        self.path = _validate_path(path)
        self.rotation = rotation
        self.quantization_bits = quantization_bits
        self.quantization_mode = quantization_mode
        self.quantization_eps = quantization_eps
        self.use_rotation_cache = use_rotation_cache

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        value = input
        has_applied_linear = False

        for op in self.path:
            if op == "r":
                value = self._apply_rotation(value, has_applied_linear)
            elif op == "q":
                value = fake_quantize_ste(
                    value,
                    num_bits=self.quantization_bits,
                    mode=self.quantization_mode,
                    eps=self.quantization_eps,
                )
            elif op == "l":
                value = self.base(value)
                has_applied_linear = True
            elif op == "i":
                value = self._apply_inverse_rotation(value, has_applied_linear)

        return value

    def _apply_rotation(self, value: torch.Tensor, after_linear: bool) -> torch.Tensor:
        if self.rotation is None:
            position = "after linear" if after_linear else "before linear"
            raise ValueError(f"path requires rotation {position}, but none was provided")
        return self.rotation(value, use_cache=self.use_rotation_cache)

    def _apply_inverse_rotation(
        self,
        value: torch.Tensor,
        after_linear: bool,
    ) -> torch.Tensor:
        if self.rotation is None:
            position = "after linear" if after_linear else "before linear"
            raise ValueError(f"path requires inverse rotation {position}, but none was provided")
        if not hasattr(self.rotation, "rotation_matrix"):
            raise TypeError("inverse rotation requires a rotation_matrix method")

        rotation_matrix = self.rotation.rotation_matrix(use_cache=self.use_rotation_cache)
        return value.matmul(rotation_matrix.transpose(-1, -2))

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


def _validate_path(path: str) -> str:
    if not path:
        raise ValueError("path cannot be empty")
    if any(op not in _PATH_OPS for op in path):
        raise ValueError(f"path may only contain {sorted(_PATH_OPS)}, got {path!r}")
    if path.count("l") != 1:
        raise ValueError("path must contain exactly one 'l'")

    for op in _UNIQUE_OPTIONAL_OPS:
        if path.count(op) > 1:
            raise ValueError(f"path cannot contain '{op}' more than once")

    return path
