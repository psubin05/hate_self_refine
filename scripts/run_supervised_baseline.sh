#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-klue/roberta-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/supervised/klue_roberta_base_seed42}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"

"${PYTHON_BIN}" src/train_supervised_hate_baseline.py \
  --model-name-or-path "${MODEL}" \
  --output-dir "${OUTPUT_ROOT}" \
  --seed 42 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}"
