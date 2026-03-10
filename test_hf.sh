#!/bin/bash
set -euo pipefail

KERNEL_BENCH_DIR="../KernelBench/KernelBench"
MODEL_ID="Qwen/Qwen2-7B-Instruct"

MODE="smoke" ## options: smoke|autoroute|pipeline|direct 
## smoke is just quick one shot test for hf local 


## workers and jobs below are set to 1 since running on just 1 gpu with multiple threads/processes can slow stuff down

case "${MODE}" in
  smoke)
    uv run python scripts/local_hf_model.py \
      --model-id "${MODEL_ID}" \
      --download-if-missing \
      smoke \
      --prompt "Say hello from a local model in one sentence."
    ;;
  autoroute)
    uv run python scripts/local_hf_model.py \
      --model-id "${MODEL_ID}" \
      --download-if-missing \
      autoroute \
      --problem "${KERNEL_BENCH_DIR}/level1/19_ReLU.py" \
      --verify \
      --no-router-cache \
      --ka-workers 1 \
      --ka-rounds 1 \
      --workers 1 \
      --dispatch-jobs 1 \
      --max-iters 1
    ;;
  pipeline)
    uv run python scripts/local_hf_model.py \
      --model-id "${MODEL_ID}" \
      --download-if-missing \
      pipeline \
      --problem "${KERNEL_BENCH_DIR}/level1/19_ReLU.py" \
      --verify \
      --workers 1 \
      --dispatch-jobs 1 \
      --max-iters 1
    ;;
  direct)
    uv run python scripts/local_hf_model.py \
      --model-id "${MODEL_ID}" \
      --download-if-missing \
      direct \
      --problem-description "Implement ReLU over a contiguous 1D tensor of length 1024" \
      --num-workers 1 \
      --max-rounds 1
    ;;
  *)
    echo "Unknown MODE='${MODE}'. Use one of: smoke, autoroute, pipeline, direct."
    exit 2
    ;;
esac