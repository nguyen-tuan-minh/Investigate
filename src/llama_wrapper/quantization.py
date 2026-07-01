"""Gradient-friendly quantization helpers.

The functions here use fake quantization: values are rounded in the forward
pass, but gradients flow through the original floating-point tensor.
"""

from __future__ import annotations

import torch


def fake_quantize_ste(
    tensor: torch.Tensor,
    num_bits: int = 8,
    mode: str = "symmetric",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-tensor fake quantization with straight-through gradients.

    Forward:
        tensor -> scale -> round/clamp -> dequantized tensor

    Backward:
        gradients pass through as if this function were the identity.
    """

    if mode == "symmetric":
        if num_bits < 2:
            raise ValueError("num_bits must be at least 2")
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        scale = tensor.detach().abs().max() / qmax
        scale = torch.clamp(scale, min=eps)

        quantized = torch.round(tensor / scale).clamp(qmin, qmax)
        dequantized = quantized * scale

    elif mode == "asymmetric":
        qmin = 0
        qmax = 2**num_bits - 1

        tensor_detached = tensor.detach()
        tensor_min = tensor_detached.min()
        tensor_max = tensor_detached.max()

        scale = (tensor_max - tensor_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=eps)

        zero_point = torch.round(qmin - tensor_min / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        dequantized = (quantized - zero_point) * scale

    else:
        raise ValueError("mode must be either 'symmetric' or 'asymmetric'")

    return tensor + (dequantized - tensor).detach()


__all__ = ["fake_quantize_ste"]
