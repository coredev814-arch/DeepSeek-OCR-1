"""GLM-OCR worker process.

Loads the GLM-OCR model once and processes images received via stdin.
Communication protocol: JSON lines over stdin/stdout.

Request:  {"image_base64": "<base64>", "max_tokens": 4096}
Response: {"text": "...", "num_tokens": 123}
Error:    {"error": "..."}

This runs in an isolated venv with transformers 5.x to avoid
conflicts with the main DeepSeek-OCR service (transformers 4.x).
"""

import base64
import io
import json
import sys
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    model_path = os.environ.get("GLM_OCR_MODEL_PATH", "/workspace/models/GLM-OCR")

    # Load model once at startup
    sys.stderr.write(f"GLM-OCR worker: loading model from {model_path}\n")
    sys.stderr.flush()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    sys.stderr.write("GLM-OCR worker: model loaded, ready for requests\n")
    sys.stderr.flush()

    # Signal readiness to parent process
    print(json.dumps({"status": "ready"}), flush=True)

    # Process requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            image_b64 = request["image_base64"]
            max_tokens = request.get("max_tokens", 4096)

            # Decode image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Build chat messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            new_tokens = output[0][input_len:]
            text = processor.decode(new_tokens, skip_special_tokens=True)
            num_tokens = len(new_tokens)

            print(json.dumps({"text": text, "num_tokens": num_tokens}), flush=True)

        except Exception as e:
            sys.stderr.write(f"GLM-OCR worker error: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
