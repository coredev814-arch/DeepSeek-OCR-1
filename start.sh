#!/bin/bash
# DeepSeek-OCR API Service Entrypoint
# Usage: ./start.sh
#
# GLM-OCR fallback is built-in — when DeepSeek-OCR fails on a page,
# the service automatically spawns a GLM-OCR worker subprocess.
# No separate server needed.
#
# Environment variables:
#   MODEL_PATH           - DeepSeek-OCR model path (default: /workspace/models/DeepSeek-OCR)
#   PORT                 - API port (default: 8000)
#   GPU_MEM_UTIL         - GPU memory utilization 0.0-1.0 (default: 0.80)
#   MAX_MODEL_LEN        - Max model context length (default: 8192)
#   MAX_TOKENS           - Max output tokens (default: 8192)
#   GLM_OCR_ENABLED      - Enable GLM-OCR fallback (default: true)
#   GLM_OCR_MODEL_PATH   - GLM-OCR model path (default: /workspace/models/GLM-OCR)
#   GLM_OCR_VENV_PYTHON  - GLM-OCR venv python (default: /workspace/glm-ocr-venv/bin/python)

set -e

export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-/workspace/models/DeepSeek-OCR}
PORT=${PORT:-8000}

echo "============================================"
echo "  DeepSeek-OCR API Service"
echo "============================================"
echo "Model:       $MODEL_PATH"
echo "Port:        $PORT"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "GLM-OCR:     ${GLM_OCR_ENABLED:-true}"
echo "GLM-OCR Model: ${GLM_OCR_MODEL_PATH:-/workspace/models/GLM-OCR}"
echo "============================================"

cd /workspace/DeepSeek-OCR-1

python3 api_service.py
