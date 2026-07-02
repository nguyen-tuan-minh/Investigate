"""Gradient-friendly quantization helpers.

The functions here use fake quantization: values are rounded in the forward
pass, but gradients flow through the original floating-point tensor.
"""

from __future__ import annotations

from typing import Optional

import torch

# Note: For asymmetric, this still use 0 point
def fake_quantize_ste(
    tensor: torch.Tensor,
    num_bits: Optional[int] = 8,
    mode: str = "symmetric",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-tensor fake quantization with straight-through gradients.

    Forward:
        tensor -> scale -> round/clamp -> dequantized tensor

    Backward:
        gradients pass through as if this function were the identity.
    """

    if num_bits is None:
        return tensor

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


if __name__ == "__main__":
    samples = {
        "zero_centered": torch.tensor(
            [-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0],
            requires_grad=True,
        ),
        "positive_activation_like": torch.tensor(
            [0.0, 0.02, 0.10, 0.30, 0.75, 1.50, 3.00],
            requires_grad=True,
        ),
        "shifted_activation_like": torch.tensor(
            [-0.20, -0.05, 0.00, 0.10, 0.40, 1.20, 4.00],
            requires_grad=True,
        ),
        "outlier_activation_like": torch.tensor(
            [-0.10, -0.02, 0.00, 0.03, 0.08, 0.15, 6.00],
            requires_grad=True,
        ),
    }

    for name, sample in samples.items():
        unchanged = fake_quantize_ste(sample, num_bits=None)
        symmetric = fake_quantize_ste(sample, num_bits=4, mode="symmetric")
        asymmetric = fake_quantize_ste(sample, num_bits=4, mode="asymmetric")

        loss = symmetric.sum() + asymmetric.sum()
        loss.backward()

        print(f"\n=== {name} ===")
        print("Input:")
        print(sample.detach())

        print("\nSymmetric fake quantized:")
        print(symmetric.detach())

        print("\nAsymmetric fake quantized:")
        print(asymmetric.detach())

        print("\nNo-op with num_bits=None:")
        print(unchanged.detach())

        print("\nGradient after STE backward:")
        print(sample.grad)
