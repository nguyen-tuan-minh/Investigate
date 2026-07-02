"""Trainable rotation modules."""

from __future__ import annotations

import torch
from torch import nn


class CayleyRotation(nn.Module):
    """Trainable orthogonal rotation using the Cayley transform.

    The module learns an unconstrained matrix, converts it to a skew-symmetric
    matrix, then maps it to an orthogonal matrix with the Cayley transform.
    With the default zero initialization, the rotation starts as identity.
    """

    def __init__(
        self,
        dim: int,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if init_scale < 0:
            raise ValueError("init_scale must be non-negative")

        self.dim = dim
        raw = torch.empty(dim, dim)
        if init_scale == 0.0:
            nn.init.zeros_(raw)
        else:
            nn.init.normal_(raw, mean=0.0, std=init_scale)

        self.raw = nn.Parameter(raw)
        self._cached_rotation = None

    def skew_symmetric_matrix(self) -> torch.Tensor:
        return self.raw - self.raw.transpose(-1, -2)

    def clear_cache(self) -> None:
        self._cached_rotation = None

    def rotation_matrix(self, use_cache: bool = False) -> torch.Tensor:
        if use_cache and self._cached_rotation is not None:
            return self._cached_rotation

        skew = self.skew_symmetric_matrix()
        eye = torch.eye(self.dim, device=skew.device, dtype=skew.dtype)

        left = eye + skew
        right = eye - skew

        rotation = torch.linalg.solve(left, right)
        if use_cache:
            self._cached_rotation = rotation

        return rotation

    def forward(self, input: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if input.shape[-1] != self.dim:
            raise ValueError(
                "Input last dimension must match rotation dimension: "
                f"{input.shape[-1]} != {self.dim}"
            )

        rotation = self.rotation_matrix(use_cache=use_cache)
        rotation = rotation.to(device=input.device, dtype=input.dtype)

        return input.matmul(rotation)

    def orthogonality_error(self, use_cache: bool = False) -> torch.Tensor:
        rotation = self.rotation_matrix(use_cache=use_cache)
        eye = torch.eye(self.dim, device=rotation.device, dtype=rotation.dtype)
        residual = rotation.transpose(-1, -2).matmul(rotation) - eye
        return residual.abs().max()


__all__ = ["CayleyRotation"]


if __name__ == "__main__":
    rotation = CayleyRotation(dim=4, init_scale=0.01)
    sample = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, -0.5, 1.0, -1.0],
        ],
        requires_grad=True,
    )

    rotation.clear_cache()
    output = rotation(sample, use_cache=True)
    loss = output.pow(2).mean()
    loss.backward()

    print("Input:")
    print(sample.detach())

    print("\nRotation matrix:")
    print(rotation.rotation_matrix(use_cache=True).detach())

    print("\nOutput:")
    print(output.detach())

    print("\nMax orthogonality error:")
    print(rotation.orthogonality_error(use_cache=True).detach())

    print("\nGradient exists for raw rotation parameter:")
    print(rotation.raw.grad is not None)
