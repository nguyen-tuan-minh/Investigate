"""Smoke test for LLaMA patch helper using a tiny fake model.

This file is intentionally runnable without pytest:
    python tests/test_llama_wrap.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_wrapper.llama import clear_llama_rotation_cache, wrap_llama_rotated_quantized
from llama_wrapper.model import WrappedModule


class FakeAttention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return hidden_states + 1.0, None


class FakeMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2.0


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = FakeAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = FakeMLP()


class FakeLlamaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 16, bias=False)


def test_wrap_fake_llama() -> None:
    torch.manual_seed(0)

    model = FakeLlamaModel()
    result = wrap_llama_rotated_quantized(
        model,
        quantization_bits=4,
        quantization_mode="asymmetric",
    )

    assert isinstance(model.embed_tokens, WrappedModule)
    assert isinstance(model.norm, WrappedModule)
    assert isinstance(model.lm_head, WrappedModule)
    assert model.embed_tokens.path == "lr"
    assert model.norm.path == "qil"
    assert model.lm_head.path == "l"
    assert len(result.wrapped_modules) == 11

    for layer in model.layers:
        assert isinstance(layer.input_layernorm, WrappedModule)
        assert isinstance(layer.self_attn, WrappedModule)
        assert isinstance(layer.post_attention_layernorm, WrappedModule)
        assert isinstance(layer.mlp, WrappedModule)
        assert layer.input_layernorm.path == "qil"
        assert layer.self_attn.path == "lr"
        assert layer.post_attention_layernorm.path == "qil"
        assert layer.mlp.path == "lr"

    input_ids = torch.tensor([[1, 2, 3]])
    hidden_states = model.embed_tokens(input_ids)
    assert hidden_states.shape == (1, 3, 4)

    normed = model.layers[0].input_layernorm(hidden_states)
    attn_output, _ = model.layers[0].self_attn(hidden_states=normed)
    post_normed = model.layers[0].post_attention_layernorm(hidden_states)
    mlp_output = model.layers[0].mlp(post_normed)
    final_hidden = model.norm(hidden_states)
    logits = model.lm_head(final_hidden)

    assert normed.shape == hidden_states.shape
    assert attn_output.shape == hidden_states.shape
    assert post_normed.shape == hidden_states.shape
    assert mlp_output.shape == hidden_states.shape
    assert final_hidden.shape == hidden_states.shape
    assert logits.shape == (1, 3, 16)
    assert result.rotation._cached_rotation is not None

    clear_llama_rotation_cache(model)
    assert result.rotation._cached_rotation is None


if __name__ == "__main__":
    test_wrap_fake_llama()
    print("LLaMA wrap smoke test passed.")
