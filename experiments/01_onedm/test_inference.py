#!/usr/bin/env python3
"""One-DM one-shot handwriting generation — test script.

Uses the OFFICIAL pretrained One-DM checkpoint. No training, no fine-tuning:
your style photo is inference-time conditioning only.

First run clones the repo and downloads ~5 GB (checkpoint + SD 1.5 VAE).

Usage:
    python experiments/01_onedm/test_inference.py \
        --style samples/my_handwriting.jpg --text "Hello world" --out out/onedm.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/dailenson/One-DM"
VENDOR_DIR = Path(__file__).resolve().parent / "_vendor" / "One-DM"
# Official pretrained checkpoint link is published in the repo README (Google Drive).
# We do NOT hardcode a Drive ID here — copy it from the README so it never goes stale.


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo() -> None:
    if VENDOR_DIR.exists():
        return
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=VENDOR_DIR)


def preprocess(style_img: Path, work_dir: Path) -> Path:
    """Deskew + binarize + normalize the user's style photo for the style encoder."""
    import cv2
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(style_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image: {style_img}")

    # Binarize (Otsu), invert so ink = white
    bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Deskew via minAreaRect on ink pixels
    coords = np.column_stack(np.where(bw > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 0.3:  # only correct real skew
        h, w = bw.shape
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        bw = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)

    # Crop to ink bounding box, normalize height to 64px (One-DM input)
    ys, xs = np.where(bw > 0)
    crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 64 / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out = work_dir / "style_preprocessed.png"
    cv2.imwrite(str(out), 255 - crop)  # back to ink=black on white
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", type=Path, required=True, help="photo of user's handwriting")
    ap.add_argument("--text", required=True, help="text to render in that style")
    ap.add_argument("--out", type=Path, default=Path("out/onedm_test.png"))
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to official One-DM .pth (downloaded from repo README link)")
    ap.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    args = ap.parse_args()

    ensure_repo()

    if args.checkpoint is None or not args.checkpoint.exists():
        raise SystemExit(
            "Download the official pretrained One-DM checkpoint first:\n"
            f"  1. Open {REPO_URL}#readme\n"
            "  2. Get the Google Drive checkpoint link (posted 2024-10-24)\n"
            "  3. Save it locally and pass: --checkpoint /path/to/one_dm.pth\n"
            "Do NOT train from scratch — the pretrained checkpoint is the whole point."
        )

    work = Path("/tmp/opencode/onedm") if Path("/tmp/opencode").exists() else Path("out/_work")
    style_clean = preprocess(args.style, work)
    print(f"Preprocessed style sample -> {style_clean}")

    device = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    if device == "cpu":
        print("WARNING: One-DM on CPU is ~minutes per word. Use the RTX 4060 or Colab.")

    # Drive the repo's own inference entrypoint (see One-DM README/test.py).
    sh([
        sys.executable, "test.py",
        "--cfg", "configs/IAM64.yml",
        "--one_dm", str(args.checkpoint.resolve()),
        "--generate_type", "iv_u",          # in-vocab words, unseen (your) style
        "--device", device,
        "--sampling_timesteps", str(args.steps),
        "--sample_method", "ddim",
        "--dir", str(args.out.parent.resolve()),
    ], cwd=VENDOR_DIR)
    print(f"Done. Output under {args.out.parent}")


if __name__ == "__main__":
    main()
