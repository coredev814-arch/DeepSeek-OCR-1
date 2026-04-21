"""Test all images in the images/ folder against the OCR service."""

import json
import os
import sys
import time
import requests

API_URL = "http://localhost:8000/ocr/image"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

def main():
    images = sorted(
        [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")],
        key=lambda x: int(x.replace("page_", "").replace(".png", ""))
    )
    print(f"Testing {len(images)} images...\n")
    print(f"{'Page':<12} {'Engine':<12} {'Flag':<8} {'Score':>6} {'Chars':>7} {'Tokens':>7} {'ExtOCR':>6} {'Time':>6}  Details")
    print("-" * 100)

    stats = {"green": 0, "yellow": 0, "red": 0, "deepseek": 0, "glm-ocr": 0, "skipped": 0}
    glm_pages = []
    total_time = 0

    for img_file in images:
        img_path = os.path.join(IMAGES_DIR, img_file)
        t0 = time.time()

        with open(img_path, "rb") as f:
            resp = requests.post(API_URL, files={"file": (img_file, f, "image/png")})

        elapsed = time.time() - t0
        total_time += elapsed

        if resp.status_code != 200:
            print(f"{img_file:<12} ERROR {resp.status_code}: {resp.text[:80]}")
            continue

        r = resp.json()
        engine = r.get("ocr_engine", "?")
        flag = r.get("flag", "?")
        score = r.get("score", {}).get("composite", 0)
        chars = len(r.get("text", ""))
        tokens = r.get("num_tokens", 0)
        ext_ocr = r.get("needs_external_ocr", False)
        details = [d.get("code", "") for d in r.get("flag_details", [])]

        stats[flag] = stats.get(flag, 0) + 1
        stats[engine] = stats.get(engine, 0) + 1
        if engine == "glm-ocr":
            glm_pages.append(img_file)

        detail_str = ", ".join(details) if details else ""
        ext_str = "YES" if ext_ocr else ""
        print(f"{img_file:<12} {engine:<12} {flag:<8} {score:>6.3f} {chars:>7} {tokens:>7} {ext_str:>6} {elapsed:>5.1f}s  {detail_str}")

    print("-" * 100)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/len(images):.1f}s/page avg)")
    print(f"Flags:  GREEN={stats['green']}  YELLOW={stats['yellow']}  RED={stats['red']}")
    print(f"Engine: DeepSeek={stats['deepseek']}  GLM-OCR={stats['glm-ocr']}  Skipped={stats['skipped']}")
    if glm_pages:
        print(f"GLM-OCR used on: {', '.join(glm_pages)}")

if __name__ == "__main__":
    main()
