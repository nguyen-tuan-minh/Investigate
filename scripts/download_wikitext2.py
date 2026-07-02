"""Download WikiText-2 raw splits to local JSONL files.

Run from the repository root:
    python scripts/download_wikitext2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    return parser.parse_args()


def write_jsonl(split_dataset: object, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in split_dataset:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
            row_count += 1

    return row_count


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root

    raw_dir = repo_root / "data" / "raw" / "wikitext2"
    cache_dir = repo_root / "data" / "cache" / "huggingface"
    manifest_path = repo_root / "data" / "manifests" / "wikitext2.json"

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        cache_dir=str(cache_dir),
    )

    split_counts: Dict[str, int] = {}
    split_paths: Dict[str, str] = {}

    for split in SPLITS:
        output_path = raw_dir / f"{split}.jsonl"
        split_counts[split] = write_jsonl(dataset[split], output_path)
        split_paths[split] = str(output_path.relative_to(repo_root))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "splits": list(SPLITS),
        "row_counts": split_counts,
        "raw_paths": split_paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote WikiText-2 splits to {raw_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
