#!/usr/bin/env python3
"""Create handwriting now — one command, any machine.

    python src/textwritter/quicktest.py                          # auto engine
    python src/textwritter/quicktest.py --engine hwt --text "Hi"  # force CPU-safe
    python src/textwritter/quicktest.py --style my_handwriting.jpg

Engine auto-picks by device (One-DM on CUDA for best fidelity, HWT everywhere
else). No style photo? Bundled handwriting samples are used, so it always runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textwritter.pipeline import DEFAULT_TEXT, create  # noqa: E402
from textwritter.runtime import engine_hint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Create handwriting from text.")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="what to write")
    ap.add_argument("--style", default=None,
                    help="photo of handwriting (optional; bundled samples if omitted)")
    ap.add_argument("--engine", default="auto", choices=["auto", "onedm", "hwt"],
                    help="auto = One-DM on CUDA, HWT otherwise (TW_ENGINE overrides)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=50, help="One-DM sampling steps")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the TrOCR read-back verdict")
    args = ap.parse_args()

    print(f"[tip] {engine_hint(args.engine)}")
    try:
        result = create(text=args.text, style=args.style, engine=args.engine,
                        out=args.out, device=args.device, steps=args.steps,
                        judge=not args.no_judge)
    except Exception as e:  # noqa: BLE001 - friendly top-level error
        print(f"\n[!] {e}")
        sys.exit(2)
    print(f"\nOpen your image: {result['out']}")
    if result.get("compare"):
        print(f"Side-by-side:    {result['compare']}")
    if not result["verdict"].get("judged"):
        print("[tip] judging skipped — generation still succeeded.")


if __name__ == "__main__":
    main()
