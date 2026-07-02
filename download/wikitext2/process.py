"""Process WikiText-2 raw JSONL files into fixed-length token blocks.

Run from the repository root:
    python -m download.wikitext2.process --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from .paths import SPLITS, raw_split_path, resolve_repo_root, token_blocks_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer name or path.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Process one split. Defaults to all splits.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Number of tokens per calibration sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of blocks per split.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Distance between block starts. Defaults to seq-len.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the current repository.",
    )
    return parser.parse_args()


def load_raw_split(split: str, repo_root: Optional[Path] = None) -> List[str]:
    path = raw_split_path(split, repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m download.wikitext2.download first."
        )

    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["text"]
            if text.strip():
                texts.append(text)

    return texts


def build_token_blocks(
    split: str,
    tokenizer: object,
    repo_root: Optional[Path] = None,
    seq_len: int = 2048,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_samples is not None and max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")

    stride = seq_len if stride is None else stride
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required")

    tokenized = tokenizer(
        load_raw_split(split, repo_root),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    token_stream: List[int] = []
    for token_ids in tokenized["input_ids"]:
        if not token_ids:
            continue
        if token_stream:
            token_stream.append(eos_token_id)
        token_stream.extend(token_ids)

    if len(token_stream) < seq_len or max_samples == 0:
        return torch.empty((0, seq_len), dtype=torch.long)

    blocks: List[torch.Tensor] = []
    last_start = len(token_stream) - seq_len
    for start in range(0, last_start + 1, stride):
        blocks.append(torch.tensor(token_stream[start : start + seq_len]))
        if max_samples is not None and len(blocks) >= max_samples:
            break

    return torch.stack(blocks).long()


def save_token_blocks(
    split: str,
    tokenizer: object,
    repo_root: Optional[Path] = None,
    seq_len: int = 2048,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
) -> Path:
    output_path = token_blocks_path(split, seq_len=seq_len, repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = build_token_blocks(
        split,
        tokenizer,
        repo_root=repo_root,
        seq_len=seq_len,
        max_samples=max_samples,
        stride=stride,
    )
    torch.save(blocks, output_path)

    return output_path


def load_token_blocks(
    split: str,
    repo_root: Optional[Path] = None,
    seq_len: int = 2048,
    map_location: Optional[str] = "cpu",
) -> torch.Tensor:
    path = token_blocks_path(split, seq_len=seq_len, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m download.wikitext2.process first."
        )
    return torch.load(path, map_location=map_location)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    splits = (args.split,) if args.split is not None else SPLITS

    for split in splits:
        path = save_token_blocks(
            split,
            tokenizer,
            repo_root=repo_root,
            seq_len=args.seq_len,
            max_samples=args.max_samples,
            stride=args.stride,
        )
        print(f"Wrote {split} token blocks to {path}")


if __name__ == "__main__":
    main()
