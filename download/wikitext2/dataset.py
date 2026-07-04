"""PyTorch Dataset and DataLoader helpers for processed WikiText-2 blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .download import download_wikitext2
from .paths import raw_split_path, resolve_repo_root, token_blocks_path
from .process import load_token_blocks, save_token_blocks, tokenizer_identifier


class Wikitext2BlockDataset(Dataset):
    """Causal-LM dataset where one sample is one fixed-length token block."""

    def __init__(self, blocks: torch.Tensor) -> None:
        if blocks.ndim != 2:
            raise ValueError(
                f"blocks must have shape [num_samples, seq_len], got {blocks.shape}"
            )
        self.blocks = blocks.long()

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        block = self.blocks[index]
        return {
            "input_ids": block,
            "labels": block.clone(),
        }


def create_wikitext2_dataloader(
    split: str,
    repo_root: Optional[Path] = None,
    seq_len: int = 2048,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    map_location: Optional[str] = "cpu",
    tokenizer: Optional[object] = None,
    tokenizer_name: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
    auto_prepare: bool = True,
) -> DataLoader:
    repo_root = resolve_repo_root(repo_root)
    ensure_wikitext2_token_blocks(
        split,
        repo_root=repo_root,
        seq_len=seq_len,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        tokenizer_id=tokenizer_id,
        max_samples=max_samples,
        stride=stride,
        auto_prepare=auto_prepare,
    )

    tokenizer_id = _resolve_tokenizer_id(tokenizer, tokenizer_name, tokenizer_id)
    blocks = load_token_blocks(
        split,
        repo_root=repo_root,
        seq_len=seq_len,
        map_location=map_location,
        tokenizer_id=tokenizer_id,
    )
    dataset = Wikitext2BlockDataset(blocks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def ensure_wikitext2_token_blocks(
    split: str,
    repo_root: Optional[Path] = None,
    seq_len: int = 2048,
    tokenizer: Optional[object] = None,
    tokenizer_name: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
    auto_prepare: bool = True,
) -> Path:
    repo_root = resolve_repo_root(repo_root)
    tokenizer_id = _resolve_tokenizer_id(tokenizer, tokenizer_name, tokenizer_id)
    processed_path = token_blocks_path(
        split,
        seq_len=seq_len,
        repo_root=repo_root,
        tokenizer_id=tokenizer_id,
    )

    _validate_prepare_inputs(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        tokenizer_id=tokenizer_id,
        auto_prepare=auto_prepare,
        processed_path=processed_path,
    )

    if processed_path.exists():
        return processed_path

    if not auto_prepare:
        raise FileNotFoundError(
            f"Missing {processed_path}. Set auto_prepare=True to build it."
        )

    raw_path = raw_split_path(split, repo_root=repo_root)
    if not raw_path.exists():
        download_wikitext2(repo_root)

    tokenizer = tokenizer or _load_tokenizer(tokenizer_name)
    return save_token_blocks(
        split,
        tokenizer,
        repo_root=repo_root,
        seq_len=seq_len,
        max_samples=max_samples,
        stride=stride,
        tokenizer_id=tokenizer_id,
    )


def _validate_prepare_inputs(
    tokenizer: Optional[object],
    tokenizer_name: Optional[str],
    tokenizer_id: Optional[str],
    auto_prepare: bool,
    processed_path: Path,
) -> None:
    if tokenizer is not None and tokenizer_name is not None:
        raise ValueError("Pass either tokenizer or tokenizer_name, not both")

    if tokenizer_id is None:
        raise ValueError(
            "tokenizer, tokenizer_name, or tokenizer_id is required to select "
            "a token block path"
        )

    if auto_prepare and tokenizer is None and tokenizer_name is None:
        raise ValueError(
            f"Missing {processed_path}. tokenizer or tokenizer_name is required "
            "to auto-process token blocks"
        )


def _load_tokenizer(tokenizer_name: Optional[str]) -> object:
    if tokenizer_name is None:
        raise ValueError(
            "tokenizer or tokenizer_name is required when processed blocks are missing"
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _resolve_tokenizer_id(
    tokenizer: Optional[object],
    tokenizer_name: Optional[str],
    tokenizer_id: Optional[str],
) -> Optional[str]:
    if tokenizer_id is not None:
        return tokenizer_id
    if tokenizer_name is not None:
        return tokenizer_name
    if tokenizer is not None:
        return tokenizer_identifier(tokenizer)
    return None
