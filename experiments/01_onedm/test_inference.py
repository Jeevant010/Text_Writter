#!/usr/bin/env python3
"""One-DM one-shot handwriting generation — the GPU (best-fidelity) engine.

Requires an NVIDIA CUDA GPU (the official test.py uses NCCL). On CPU machines use
the HWT engine instead: `python experiments/02_hwt/test_inference.py` or
`python src/textwritter/quicktest.py --engine hwt`.

Usage:
    python experiments/01_onedm/test_inference.py --check      # no downloads
    python experiments/01_onedm/test_inference.py --text "Hello world"
    python experiments/01_onedm/test_inference.py \
        --style samples/my_handwriting.jpg --text "Hello world" --out out/onedm.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from textwritter.runtime import OUT_DIR, engine_hint, print_runtime_card  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog")
    ap.add_argument("--style", default=None,
                    help="photo of handwriting (optional; bundled samples if omitted)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "onedm_test.png")
    ap.add_argument("--device", default=None, help="cuda (default)")
    ap.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    ap.add_argument("--check", action="store_true",
                    help="report readiness without downloading or generating")
    args = ap.parse_args()

    if args.check:
        st = engine.info()
        print("One-DM readiness")
        print(f"  CUDA GPU    : {st['cuda']} ({st['gpu'] or 'none'})")
        print(f"  vendor repo : {st['vendor']}")
        print(f"  checkpoint  : {st['checkpoint']}  ({st['doc']})")
        print(f"  unifont     : {st['unifont']}")
        print(f"\n  {engine_hint('onedm')}")
        sys.exit(0 if st["cuda"] else 1)

    print_runtime_card("onedm")
    work = args.out.parent / "_onedm_work"
    style_dir, source, warnings = engine.prepare_style(args.style, work)
    for w in warnings:
        print(f"[onedm] note: {w}")
    result = engine.generate(args.text, style_dir, args.out, device=args.device,
                             steps=args.steps)
    print(f"\nDone. style={source} image={result['image']}")


if __name__ == "__main__":
    main()
