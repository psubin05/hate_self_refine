#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/proposed}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

"${PYTHON_BIN}" src/iccd_minimal.py \
  --model-name-or-path "${MODEL}" \
  --output-dir "${OUTPUT_ROOT}" \
  --negative-context-method self_refine \
  --self-refine-target-policy ontology_minimal \
  --self-refine-critic-profile fact_only \
  --self-refine-revision-mode critic \
  --self-refine-iterations 1 \
  --seed 42 \
  --shots-per-class 3 \
  --batch-size "${BATCH_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --use-title \
  --use-chat-template
