"""Shared WikiText-2 paths and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLITS = ("train", "validation", "test")


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_root(repo_root: Optional[Path] = None) -> Path:
    return repo_root or repo_root_from_file()


def validate_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return split


def raw_dir(repo_root: Optional[Path] = None) -> Path:
    return resolve_repo_root(repo_root) / "data" / "raw" / "wikitext2"


def processed_dir(repo_root: Optional[Path] = None) -> Path:
    return resolve_repo_root(repo_root) / "data" / "processed" / "wikitext2"


def cache_dir(repo_root: Optional[Path] = None) -> Path:
    return resolve_repo_root(repo_root) / "data" / "cache" / "huggingface"


def manifest_path(repo_root: Optional[Path] = None) -> Path:
    return resolve_repo_root(repo_root) / "data" / "manifests" / "wikitext2.json"


def raw_split_path(split: str, repo_root: Optional[Path] = None) -> Path:
    return raw_dir(repo_root) / f"{validate_split(split)}.jsonl"


def token_blocks_path(
    split: str,
    seq_len: int = 2048,
    repo_root: Optional[Path] = None,
) -> Path:
    return processed_dir(repo_root) / f"{validate_split(split)}_blocks_{seq_len}.pt"
