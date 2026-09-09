#!/usr/bin/env python3
"""HWT (Handwriting Transformers) few-shot generation — test script.

Uses the OFFICIAL pretrained HWT checkpoint (iam_model.pth). No training.

Usage:
    python experiments/02_hwt/test_inference.py \
        --style samples/my_handwriting.jpg --text "The quick brown fox" --out out/hwt.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ankanbhunia/Handwriting-Transformers"
VENDOR_DIR = Path(__file__).resolve().parent / "_vendor" / "HWT"
# Official pretrained weights + data bundle, from the repo's INSTALL.md:
GDRIVE_ID = "16g9zgysQnWk7-353_tMig92KsZsrcM6k"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo() -> None:
    if (VENDOR_DIR / "files" / "iam_model.pth").exists():
        return
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not VENDOR_DIR.exists():
        sh(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    sh([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "gdown"])
    sh(["gdown", "--id", GDRIVE_ID], cwd=VENDOR_DIR)
    sh(["unzip", "-o", "files.zip"], cwd=VENDOR_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", type=Path, required=True,
                    help="paragraph photo; HWT crops a few word images from it")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", type=Path, default=Path("out/hwt_test.png"))
    args = ap.parse_args()

    ensure_repo()

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    print(f"Device: {device} (CPU is fine for HWT: ~1-3 s/word)")

    # HWT's official entry for custom handwriting is demo_custom_handwriting.ipynb.
    # For scripted use: write the query text into mytext.txt, place cropped style
    # word-images per the repo's custom-dataset format, then run their inference.
    # See: https://github.com/ankanbhunia/Handwriting-Transformers#readme
    (VENDOR_DIR / "mytext.txt").write_text(args.text + "\n")
    print(
        "Repo ready at", VENDOR_DIR, "\n"
        "Next (per repo README): place your style word-crops in the custom format,\n"
        "then run the repo's inference on mytext.txt. The official Colab\n"
        "(demo_custom_handwriting.ipynb) is the fastest way to try it interactively."
    )


if __name__ == "__main__":
    main()
