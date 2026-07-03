#!/usr/bin/env bash
set -euo pipefail

TOKENIZER="${TOKENIZER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
STRIDE="${STRIDE:-}"
SPLIT="${SPLIT:-}"

PROCESS_ARGS=(
  --tokenizer "$TOKENIZER"
  --seq-len "$SEQ_LEN"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  PROCESS_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$STRIDE" ]]; then
  PROCESS_ARGS+=(--stride "$STRIDE")
fi

if [[ -n "$SPLIT" ]]; then
  PROCESS_ARGS+=(--split "$SPLIT")
fi

python -m download.wikitext2.download
python -m download.wikitext2.process "${PROCESS_ARGS[@]}"
