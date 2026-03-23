# DeepSeek-OCR API Service — RunPod GPU Pod
# Build:  docker build -t deepseek-ocr-api .
# Run:    docker run --gpus all -p 8000:8000 deepseek-ocr-api

FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/DeepSeek-OCR

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        vllm==0.8.5 \
        fastapi uvicorn python-multipart aiofiles && \
    pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Copy project files
COPY . .

# Download model at build time (optional — remove if mounting externally)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR', local_dir='/workspace/models/DeepSeek-OCR')"

RUN chmod +x start.sh

ENV MODEL_PATH=/workspace/models/DeepSeek-OCR
ENV PORT=8000
ENV GPU_MEM_UTIL=0.9

EXPOSE 8000

CMD ["./start.sh"]
