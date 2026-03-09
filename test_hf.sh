#!/bin/bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2-7B-Instruct"

# uv run python scripts/local_hf_model.py \
#   --model-id "${MODEL_ID}" \
#   --download-if-missing \
#   smoke \
#   --prompt "Say hello from a local model in one sentence."

KERNEL_BENCH_DIR="../KernelBench/KernelBench"
uv run python scripts/local_hf_model.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --download-if-missing \
  autoroute \
  --problem "${KERNEL_BENCH_DIR}/level1/19_ReLU.py" \
  --verify \
  --no-router-cache