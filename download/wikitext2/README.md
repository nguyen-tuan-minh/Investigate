# WikiText-2 Data Utilities

This folder is separate from `src/` and only handles WikiText-2 download,
processing, and PyTorch dataloader creation.

## Quick Start

Run this from the repository root:

```bash
bash download/wikitext2/download_and_process.sh
```

By default, this downloads all WikiText-2 splits and processes them with:

```text
tokenizer = TinyLlama/TinyLlama-1.1B-Chat-v1.0
seq_len   = 2048
```

Override defaults with environment variables:

```bash
TOKENIZER=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
SEQ_LEN=2048 \
MAX_SAMPLES=128 \
bash download/wikitext2/download_and_process.sh
```

## 1. Download Raw WikiText-2

Run this from the repository root:

```bash
python -m download.wikitext2.download
```

This creates:

```text
data/raw/wikitext2/train.jsonl
data/raw/wikitext2/validation.jsonl
data/raw/wikitext2/test.jsonl
data/manifests/wikitext2.json
```

Each JSONL row has one field:

```json
{"text": "..."}
```

## 2. Process Into Token Blocks

Run this after downloading:

```bash
python -m download.wikitext2.process \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --seq-len 2048
```

This tokenizes the raw text, inserts the tokenizer EOS token between documents,
concatenates everything into one token stream, and splits it into fixed-length
blocks.

It creates:

```text
data/processed/wikitext2/train_TinyLlama_TinyLlama-1.1B-Chat-v1.0_blocks_2048.pt
data/processed/wikitext2/validation_TinyLlama_TinyLlama-1.1B-Chat-v1.0_blocks_2048.pt
data/processed/wikitext2/test_TinyLlama_TinyLlama-1.1B-Chat-v1.0_blocks_2048.pt
```

One block is one calibration/training sample:

```text
sample shape = [seq_len]
```

## 3. Create a DataLoader

The dataloader helper auto-prepares missing data. If processed token blocks do
not exist, it will download raw WikiText-2 files if needed, then process the
requested split.

```python
from download.wikitext2 import create_wikitext2_dataloader

train_loader = create_wikitext2_dataloader(
    "train",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=2048,
    batch_size=8,
    shuffle=True,
)

batch = next(iter(train_loader))

print(batch["input_ids"].shape)
print(batch["labels"].shape)
```

Expected shape:

```text
torch.Size([8, 2048])
torch.Size([8, 2048])
```

`tokenizer_name` or `tokenizer` is required only when the processed `.pt` file
is missing. If the token blocks already exist, the helper loads them directly.
You may pass `tokenizer_id` when you only need to select an existing processed
file without loading a tokenizer:

```python
train_loader = create_wikitext2_dataloader(
    "train",
    tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=2048,
    batch_size=8,
    auto_prepare=False,
)
```

For causal language modeling, `labels` are a clone of `input_ids`. Hugging Face
causal LM models shift the labels internally.

## Split Usage

Use the splits like this:

```text
train       -> train or calibrate rotation
validation  -> check loss/perplexity while tuning settings
test        -> final evaluation only
```

## Useful Options

Process only one split:

```bash
python -m download.wikitext2.process \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --split train \
  --seq-len 2048
```

Limit the number of blocks for a quick test:

```bash
python -m download.wikitext2.process \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --seq-len 2048 \
  --max-samples 128
```

Use overlapping blocks:

```bash
python -m download.wikitext2.process \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --seq-len 2048 \
  --stride 1024
```

By default, `stride` is equal to `seq_len`, so blocks are non-overlapping.

## Generated Files

Raw downloads, processed token blocks, and Hugging Face cache files are ignored
by Git:

```text
data/raw/
data/processed/
data/cache/
```
