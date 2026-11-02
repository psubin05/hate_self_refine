#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/reverse_mapping}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

for retrieval_method in random bm25 topk; do
  "${PYTHON_BIN}" src/iccd_minimal.py \
    --model-name-or-path "${MODEL}" \
    --output-dir "${OUTPUT_ROOT}/${retrieval_method}" \
    --negative-context-method reverse_mapping \
    --retrieval-method "${retrieval_method}" \
    --seed 42 \
    --reverse-mapping-seed 66 \
    --shots-per-class 3 \
    --batch-size "${BATCH_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --use-title \
    --use-chat-template
done
