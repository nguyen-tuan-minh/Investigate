"""WikiText-2 download, processing, and dataloader helpers."""

from .paths import DATASET_CONFIG, DATASET_NAME, SPLITS

__all__ = [
    "DATASET_CONFIG",
    "DATASET_NAME",
    "SPLITS",
    "Wikitext2BlockDataset",
    "build_token_blocks",
    "create_wikitext2_dataloader",
    "load_raw_split",
    "load_token_blocks",
    "save_token_blocks",
]


def __getattr__(name: str) -> object:
    if name in {"Wikitext2BlockDataset", "create_wikitext2_dataloader"}:
        from .dataset import Wikitext2BlockDataset, create_wikitext2_dataloader

        exports = {
            "Wikitext2BlockDataset": Wikitext2BlockDataset,
            "create_wikitext2_dataloader": create_wikitext2_dataloader,
        }
        return exports[name]

    if name in {
        "build_token_blocks",
        "load_raw_split",
        "load_token_blocks",
        "save_token_blocks",
    }:
        from .process import (
            build_token_blocks,
            load_raw_split,
            load_token_blocks,
            save_token_blocks,
        )

        exports = {
            "build_token_blocks": build_token_blocks,
            "load_raw_split": load_raw_split,
            "load_token_blocks": load_token_blocks,
            "save_token_blocks": save_token_blocks,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
