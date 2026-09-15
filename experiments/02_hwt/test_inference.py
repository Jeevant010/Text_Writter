#!/usr/bin/env python3
"""HWT few-shot handwriting generation — runs on CPU or GPU.

Pretrained HWT (official iam_model.pth), no training: your photo (optional) is
inference-time style conditioning. If you don't pass --style, the bundled
handwriting samples are used so the pipeline still runs end to end.

Usage:
    python experiments/02_hwt/test_inference.py --text "The quick brown fox"
    python experiments/02_hwt/test_inference.py \
        --style samples/my_handwriting.jpg --text "Hello world" --out out/hwt.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import generate, prepare_style  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from textwritter.runtime import OUT_DIR, print_runtime_card  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog")
    ap.add_argument("--style", default=None,
                    help="photo of handwriting (optional — bundled samples if omitted)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "hwt_test.png")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    print_runtime_card("hwt")
    work = args.out.parent / "_hwt_work"
    style_dir, source, warnings = prepare_style(args.style, work)
    for w in warnings:
        print(f"[hwt] note: {w}")
    result = generate(args.text, style_dir, args.out, device=args.device)
    print(f"\nDone. style={source} device={args.device or 'auto'} "
          f"image={result['image']}")


if __name__ == "__main__":
    main()
