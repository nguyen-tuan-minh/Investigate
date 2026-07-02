"""PyTorch Dataset and DataLoader helpers for processed WikiText-2 blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .process import load_token_blocks


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
) -> DataLoader:
    blocks = load_token_blocks(
        split,
        repo_root=repo_root,
        seq_len=seq_len,
        map_location=map_location,
    )
    dataset = Wikitext2BlockDataset(blocks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
