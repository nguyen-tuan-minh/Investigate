"""Download WikiText-2 raw splits to JSONL files.

Run from the repository root:
    python -m download.wikitext2.download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset

from .paths import (
    DATASET_CONFIG,
    DATASET_NAME,
    SPLITS,
    cache_dir,
    manifest_path,
    raw_split_path,
    resolve_repo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the current repository.",
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


def download_wikitext2(repo_root: Path) -> Dict[str, int]:
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        cache_dir=str(cache_dir(repo_root)),
    )

    split_counts: Dict[str, int] = {}
    split_paths: Dict[str, str] = {}

    for split in SPLITS:
        output_path = raw_split_path(split, repo_root)
        split_counts[split] = write_jsonl(dataset[split], output_path)
        split_paths[split] = str(output_path.relative_to(repo_root))

    output_manifest = manifest_path(repo_root)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "config": DATASET_CONFIG,
                "splits": list(SPLITS),
                "row_counts": split_counts,
                "raw_paths": split_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return split_counts


def main() -> None:
    repo_root = resolve_repo_root(parse_args().repo_root)
    split_counts = download_wikitext2(repo_root)

    print(f"Wrote WikiText-2 raw files to {raw_split_path('train', repo_root).parent}")
    print(f"Wrote manifest to {manifest_path(repo_root)}")
    print(f"Rows: {split_counts}")


if __name__ == "__main__":
    main()
