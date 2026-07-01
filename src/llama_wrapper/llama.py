"""Helpers for patching Hugging Face LLaMA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from torch import nn

from .model import WrappedModule
from .rotation import CayleyRotation


@dataclass
class LlamaWrapResult:
    """Result returned after patching a LLaMA model in-place."""

    rotation: CayleyRotation
    wrapped_modules: List[str]


def wrap_llama_rotated_quantized(
    model: nn.Module,
    rotation: Optional[CayleyRotation] = None,
    quantization_bits: int = 8,
    quantization_mode: str = "asymmetric",
    quantization_eps: float = 1e-8,
    use_rotation_cache: bool = True,
) -> LlamaWrapResult:
    """Patch a Hugging Face LLaMA model in-place.

    The wrapped flow is:
        embedding:  input_ids -> embedding -> rotate
        attention:  quantize -> inverse rotate -> attention -> rotate
        MLP:        quantize -> inverse rotate -> MLP -> rotate
        lm_head:    inverse rotate -> lm_head

    This keeps hidden states in the rotated basis between major submodules.
    """

    backbone = _get_llama_backbone(model)
    hidden_size = _get_hidden_size(model, backbone)
    rotation = rotation or CayleyRotation(dim=hidden_size)

    wrapped_modules: List[str] = []

    if not isinstance(backbone.embed_tokens, WrappedModule):
        backbone.embed_tokens = WrappedModule(
            backbone.embed_tokens,
            path="lr",
            rotation=rotation,
            quantization_bits=quantization_bits,
            quantization_mode=quantization_mode,
            quantization_eps=quantization_eps,
            use_rotation_cache=use_rotation_cache,
        )
        wrapped_modules.append("embed_tokens")

    for layer_index, layer in enumerate(backbone.layers):
        if not isinstance(layer.self_attn, WrappedModule):
            layer.self_attn = WrappedModule(
                layer.self_attn,
                path="qilr",
                rotation=rotation,
                quantization_bits=quantization_bits,
                quantization_mode=quantization_mode,
                quantization_eps=quantization_eps,
                use_rotation_cache=use_rotation_cache,
            )
            wrapped_modules.append(f"layers.{layer_index}.self_attn")

        if not isinstance(layer.mlp, WrappedModule):
            layer.mlp = WrappedModule(
                layer.mlp,
                path="qilr",
                rotation=rotation,
                quantization_bits=quantization_bits,
                quantization_mode=quantization_mode,
                quantization_eps=quantization_eps,
                use_rotation_cache=use_rotation_cache,
            )
            wrapped_modules.append(f"layers.{layer_index}.mlp")

    if hasattr(model, "lm_head") and not isinstance(model.lm_head, WrappedModule):
        model.lm_head = WrappedModule(
            model.lm_head,
            path="il",
            rotation=rotation,
            quantization_bits=quantization_bits,
            quantization_mode=quantization_mode,
            quantization_eps=quantization_eps,
            use_rotation_cache=use_rotation_cache,
        )
        wrapped_modules.append("lm_head")

    return LlamaWrapResult(rotation=rotation, wrapped_modules=wrapped_modules)


def clear_llama_rotation_cache(model: nn.Module) -> None:
    """Clear cached rotation matrices for wrapped LLaMA modules."""

    for module in model.modules():
        rotation = getattr(module, "rotation", None)
        if rotation is not None and hasattr(rotation, "clear_cache"):
            rotation.clear_cache()


def _get_llama_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model

    raise TypeError("Expected a LLaMA model or LLaMAForCausalLM-like wrapper")


def _get_hidden_size(model: nn.Module, backbone: nn.Module) -> int:
    config = getattr(model, "config", None) or getattr(backbone, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    embedding_dim = getattr(backbone.embed_tokens, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)

    raise ValueError("Could not infer LLaMA hidden_size")


__all__ = [
    "LlamaWrapResult",
    "clear_llama_rotation_cache",
    "wrap_llama_rotated_quantized",
]
