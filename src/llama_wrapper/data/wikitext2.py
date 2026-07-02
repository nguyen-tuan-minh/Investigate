"""Helpers for local WikiText-2 data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


SPLITS = ("train", "validation", "test")


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return split


def raw_split_path(split: str, repo_root: Optional[Path] = None) -> Path:
    split = validate_split(split)
    root = repo_root or repo_root_from_file()
    return root / "data" / "raw" / "wikitext2" / f"{split}.jsonl"


def processed_split_path(split: str, repo_root: Optional[Path] = None) -> Path:
    split = validate_split(split)
    root = repo_root or repo_root_from_file()
    return root / "data" / "processed" / "wikitext2" / f"{split}_tokenized.pt"


def load_wikitext2_split(split: str, repo_root: Optional[Path] = None) -> List[str]:
    """Load a raw WikiText-2 split from JSONL as a list of text rows."""

    path = raw_split_path(split, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_wikitext2.py first."
        )

    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(row["text"])

    return texts


def tokenize_wikitext2_split(
    split: str,
    tokenizer: object,
    repo_root: Optional[Path] = None,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize a WikiText-2 split with a provided Hugging Face tokenizer."""

    texts = load_wikitext2_split(split, repo_root=repo_root)
    texts = [text for text in texts if text.strip()]

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    )

    return {key: value for key, value in tokenized.items()}


def save_tokenized_wikitext2_split(
    split: str,
    tokenizer: object,
    repo_root: Optional[Path] = None,
    max_length: Optional[int] = None,
) -> Path:
    """Tokenize and save a WikiText-2 split as a ``.pt`` file."""

    output_path = processed_split_path(split, repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenized = tokenize_wikitext2_split(
        split,
        tokenizer,
        repo_root=repo_root,
        max_length=max_length,
    )
    torch.save(tokenized, output_path)

    return output_path
