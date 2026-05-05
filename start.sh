#!/bin/bash
# DeepSeek-OCR API Service Entrypoint
# Usage: ./start.sh
# Environment variables:
#   MODEL_PATH       - Path to model weights (default: /workspace/models/DeepSeek-OCR)
#   PORT             - API port (default: 8000)
#   GPU_MEM_UTIL     - GPU memory utilization 0.0-1.0 (default: 0.9)
#   MAX_MODEL_LEN    - Max model context length (default: 8192)
#   MAX_TOKENS       - Max output tokens (default: 8192)

set -e

export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Raise vLLM's per-iteration deadline (default 60s). Long pages on dense docs
# can exceed 60s under load; without this, one slow iteration kills the
# AsyncLLMEngine background loop and every subsequent request returns 500.
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-180}

MODEL_PATH=${MODEL_PATH:-/workspace/models/DeepSeek-OCR}
PORT=${PORT:-8000}

echo "============================================"
echo "  DeepSeek-OCR API Service"
echo "============================================"
echo "Model:    $MODEL_PATH"
echo "Port:     $PORT"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

cd "$(dirname "$0")"

python3 api_service.py
