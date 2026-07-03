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


class WrappedModule(nn.Module):
    """Generic wrapper for a LLaMA submodule with compact transform paths.

    Use this when you want to wrap a larger module, such as attention or MLP,
    while still preserving the original forward behavior exactly. The path is
    applied to the first positional argument, then remaining args/kwargs are
    passed to the wrapped module at the ``l`` step.

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
        base: nn.Module,
        path: str = "l",
        rotation: Optional[nn.Module] = None,
        quantization_bits: Optional[int] = 8,
        quantization_mode: str = "asymmetric",
        quantization_eps: float = 1e-8,
        use_rotation_cache: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.path = _validate_path(path)
        self.rotation = rotation
        self.quantization_bits = quantization_bits
        self.quantization_mode = quantization_mode
        self.quantization_eps = quantization_eps
        self.use_rotation_cache = use_rotation_cache

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            input, *remaining_args = args
            return self._forward_path(input, *remaining_args, **kwargs)

        if "hidden_states" in kwargs:
            input = kwargs.pop("hidden_states")
            return self._forward_path(input, **kwargs)

        if "inputs_embeds" in kwargs:
            input = kwargs.pop("inputs_embeds")
            return self._forward_path(input, **kwargs)

        if "input_ids" in kwargs:
            input = kwargs.pop("input_ids")
            return self._forward_path(input, **kwargs)

        raise TypeError("WrappedModule requires at least one positional input")

    def _forward_path(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        value = input
        has_applied_linear = False

        for op in self.path:
            if op == "r":
                value = self._map_transform(
                    value,
                    lambda tensor: self._apply_rotation(tensor, has_applied_linear),
                )
            elif op == "q":
                value = self._map_transform(
                    value,
                    lambda tensor: fake_quantize_ste(
                        tensor,
                        num_bits=self.quantization_bits,
                        mode=self.quantization_mode,
                        eps=self.quantization_eps,
                    ),
                )
            elif op == "l":
                value = self.base(value, *args, **kwargs)
                has_applied_linear = True
            elif op == "i":
                value = self._map_transform(
                    value,
                    lambda tensor: self._apply_inverse_rotation(
                        tensor,
                        has_applied_linear,
                    ),
                )

        return value

    def _map_transform(self, value: Any, transform: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return transform(value)
        if isinstance(value, tuple):
            if not value or not isinstance(value[0], torch.Tensor):
                raise TypeError("path transform expected tensor as first tuple item")
            return (transform(value[0]), *value[1:])
        if isinstance(value, list):
            if not value or not isinstance(value[0], torch.Tensor):
                raise TypeError("path transform expected tensor as first list item")
            return [transform(value[0]), *value[1:]]

        raise TypeError(f"path transform expected tensor/tuple/list, got {type(value)!r}")

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
        rotation_matrix = rotation_matrix.to(device=value.device, dtype=value.dtype)
        return value.matmul(rotation_matrix.transpose(-1, -2))


__all__ = ["WrappedModule"]


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
