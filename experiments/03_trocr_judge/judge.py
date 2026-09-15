#!/usr/bin/env python3
"""TrOCR read-back judge — the quality gate for the re-roll loop.

Reads a generated handwriting image and scores it against the intended text.
CER above threshold = illegible = caller should re-roll with a new seed.

Usage:
    python experiments/03_trocr_judge/judge.py --image out/onedm.png --expect "Hello world"
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

MODEL_ID = "microsoft/trocr-base-handwritten"  # swap to -large- for borderline calls
DEFAULT_THRESHOLD = 0.15  # 15% character error rate


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower()


def cer(target: str, recognized: str) -> float:
    """Character Error Rate via Levenshtein distance / len(target)."""
    t, r = normalize(target), normalize(recognized)
    if not t:
        return 0.0
    prev = list(range(len(r) + 1))
    for i, tc in enumerate(t, 1):
        cur = [i]
        for j, rc in enumerate(r, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (tc != rc)))
        prev = cur
    return prev[-1] / len(t)


def _device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def recognize(image: Path, device: str = "auto") -> str:
    from transformers import (
        RobertaTokenizerFast,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )
    from PIL import Image

    dev = _device(device)
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    except Exception:
        img_proc = ViTImageProcessor.from_pretrained(MODEL_ID)
        tok = RobertaTokenizerFast.from_pretrained(MODEL_ID)
        processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(dev)
    pixel = processor(images=Image.open(image).convert("RGB"),
                      return_tensors="pt").pixel_values.to(dev)
    ids = model.generate(pixel)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--expect", required=True, help="the text the image is supposed to say")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()

    recognized = recognize(args.image, device=args.device)
    score = cer(args.expect, recognized)
    verdict = "PASS" if score <= args.threshold else "FAIL (re-roll)"

    print(f"expected   : {args.expect}")
    print(f"recognized : {recognized}")
    print(f"CER        : {score:.3f}  (threshold {args.threshold})")
    print(f"verdict    : {verdict}")
    sys.exit(0 if score <= args.threshold else 1)


if __name__ == "__main__":
    main()
