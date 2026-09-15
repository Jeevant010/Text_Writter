#!/usr/bin/env python3
"""HWT engine — real handwriting creation that runs on CPU *and* GPU.

This is the "works anywhere" generation path: one small pretrained bundle
(~685 MB, auto-downloaded), a few style word images (a photo of your handwriting,
or the bundled IAM samples when you don't give one), and the text you want.

It drives the official HWT repo's own model (no retraining) through a small
driver script executed inside the vendored clone, because that repo uses
repo-relative imports (`from params import *`). Everything around it — style
preparation, word segmentation, output cropping, judging hand-off — lives here.

Engine contract (shared with 01_onedm/engine.py):
    generate(text, style_dir, out_png, device, ...) -> {"image": Path, "text": str}
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ENGINE_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from textwritter.runtime import SAMPLE_STYLES_DIR, detect_environment  # noqa: E402
from textwritter.style import prepare_style as _prepare_style  # noqa: E402

REPO_URL = "https://github.com/ankanbhunia/Handwriting-Transformers"
VENDOR_DIR = _ENGINE_DIR / "_vendor" / "HWT"
GDRIVE_ID = "16g9zgysQnWk7-353_tMig92KsZsrcM6k"  # official bundle (repo README)
DRIVER_NAME = "_tw_hwt_driver.py"


# ---------------------------------------------------------------------------
# Vendor repo + weights (get-or-guide, never a raw crash)
# ---------------------------------------------------------------------------

def _sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_gdown() -> None:
    """gdown is in requirements.txt; install it here if the env lacks it."""
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass
    attempts = [
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        ["uv", "pip", "install", "--python", sys.executable, "gdown"],
    ]
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True)
            import gdown  # noqa: F401
            return
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("gdown is required to fetch HWT weights — install it: "
                       "pip install gdown")


def ensure_repo(verbose: bool = True) -> Path:
    """Clone HWT and fetch/unzip the official bundle; returns the vendor dir."""
    if not (VENDOR_DIR / "models" / "model.py").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        _sh(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    weights = VENDOR_DIR / "files" / "iam_model.pth"
    if not weights.exists():
        import zipfile

        _ensure_gdown()
        import gdown

        zip_path = VENDOR_DIR / "files.zip"
        if not zip_path.exists():
            gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False)
        print(f"[hwt] unzipping {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(VENDOR_DIR)
        zip_path.unlink(missing_ok=True)
        if not weights.exists():
            raise RuntimeError(
                f"HWT weights missing after download: {weights}. Place the "
                "official files.zip contents into "
                f"{VENDOR_DIR / 'files'} (see repo INSTALL.md).")
    # The repo's TRGAN reads these at import; create if the bundle lacked them.
    for name, default in (("mytext.txt", "hello world\n"),
                          ("files/english_words.txt", "hello\nworld\n")):
        p = VENDOR_DIR / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(default)
    if verbose:
        print(f"[hwt] vendor ready: {VENDOR_DIR}")
    return VENDOR_DIR


def alphabet() -> str:
    """The trained character set (from the vendor params.py, parsed statically)."""
    from textwritter.runtime import read_string_constant

    vendor = ensure_repo(verbose=False)
    return read_string_constant(vendor / "params.py", "ALPHABET")


def filter_text(text: str) -> tuple[str, list[str]]:
    """Drop characters the model was never trained on (warn instead of crash)."""
    allowed = set(alphabet())
    if not allowed:
        return text, []
    kept, dropped = [], []
    for ch in text:
        (kept if ch in allowed or ch.isspace() else dropped).append(ch)
    warnings = []
    if dropped:
        warnings.append(f"dropped {len(dropped)} unsupported character(s): "
                        f"{''.join(sorted(set(dropped)))!r}")
    return " ".join("".join(kept).split()), warnings


def extract_iam_styles(verbose: bool = True) -> list[Path]:
    """Save a writer's IAM word images (from the official bundle) as bundled styles."""
    import pickle

    ensure_repo(verbose=False)
    data_path = VENDOR_DIR / "files" / "IAM-32.pickle"
    if not data_path.exists():
        return []
    with open(data_path, "rb") as f:
        writers = pickle.load(f)["train"]
    for wid in sorted(writers, key=str):
        items = writers[wid]
        if len(items) < 5:
            continue
        SAMPLE_STYLES_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for k, item in enumerate(items[:15]):
            img = item["img"].convert("L")
            label = re.sub(r"[^A-Za-z0-9]+", "_", str(item.get("label", k)))[:24]
            p = SAMPLE_STYLES_DIR / f"{k:02d}_{label or 'w'}.png"
            img.save(p)
            saved.append(p)
        if verbose:
            print(f"[hwt] extracted bundled style words from IAM writer {wid}")
        return saved
    return []


def prepare_style(style, work_dir: Path, verbose: bool = True):
    """Style word folder — shared logic in textwritter.style."""
    return _prepare_style(style, work_dir, extract=extract_iam_styles, verbose=verbose)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# The driver runs inside the vendored repo (repo-relative imports work there).
# It mirrors the official demo notebook, device-aware, and keeps only the
# generated half of the model's output image.
_DRIVER_SRC = r'''
import argparse, json, os, sys, types
import numpy as np
import cv2
import torch
from PIL import Image

# model.py does `import pandas` but never uses it for inference; stub if absent.
try:
    import pandas  # noqa: F401
except ImportError:
    import importlib.machinery
    _pandas = types.ModuleType("pandas")
    _pandas.__spec__ = importlib.machinery.ModuleSpec("pandas", None)
    sys.modules.setdefault("pandas", _pandas)

from params import *
from models.model import TRGAN
from data.dataset import crop_, get_transform

ap = argparse.ArgumentParser()
ap.add_argument("--style-dir", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--model", default="files/iam_model.pth")
ap.add_argument("--num-samples", type=int, default=15)
a = ap.parse_args()
device = torch.device(a.device)


def load_style(folder, n):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    idx = np.random.choice(len(paths), n, replace=len(paths) <= n)
    imgs = []
    widths = []
    tf = get_transform(grayscale=True)
    for i in idx:
        im = np.array(Image.open(paths[i]).convert("L"))
        im = crop_(im)
        im = cv2.resize(im, (int(32 * (im.shape[1] / im.shape[0])), 32))
        img = 255 - im
        h, w = img.shape
        out = np.zeros((h, 192), dtype="float32")
        out[:, :w] = img[:, :192]
        img = 255 - out
        imgs.append(tf(Image.fromarray(img)))
        widths.append(min(w, 192))
    stacked = torch.cat(imgs, 0).unsqueeze(0).to(device)      # [1, n, 32, 192]
    return stacked, torch.Tensor(widths).unsqueeze(0).to(device)


model = TRGAN(batch_size=1)
model.netG.load_state_dict(torch.load(a.model, map_location="cpu"))
model.netG.to(device)
model.eval()

ST, SLEN = load_style(a.style_dir, a.num_samples)
words = a.text.split(" ")
enc, lens = model.netconverter.encode([w.encode() for w in words])
enc = enc.to(device).repeat(1, 1, 1)

with torch.no_grad():
    fakes = model.netG.Eval(ST, enc)

gap = np.ones([IMG_HEIGHT, 16])
word_t, word_l, line_wids = [], [], []
for idx, fake_ in enumerate(fakes):
    w = int(lens[idx]) * resolution
    word_t.append((fake_[0, 0, :, :w].cpu().numpy() + 1) / 2)
    word_t.append(gap)
    if len(word_t) == 16 or idx == len(fakes) - 1:
        line_ = np.concatenate(word_t, -1)
        word_l.append(line_)
        line_wids.append(line_.shape[1])
        word_t = []

gap_h = np.ones([16, max(line_wids)])
page_ = []
for line_ in word_l:
    pad = np.ones([IMG_HEIGHT, max(line_wids) - line_.shape[1]])
    page_.append(np.concatenate([line_, pad], 1))
    page_.append(gap_h)
page = np.concatenate(page_, 0) * 255

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cv2.imwrite(a.out, page.astype("uint8"))
print("TW_RESULT " + json.dumps({"image": a.out, "text": a.text, "device": a.device}))
'''


def generate(text: str, style_dir: Path, out_png: Path,
             device: str | None = None, num_samples: int = 15) -> dict:
    """Render `text` in the style of `style_dir`'s word images. Returns dict."""
    ensure_repo()
    device = device or detect_environment().device
    clean, warnings = filter_text(text)
    for w in warnings:
        print(f"[hwt] {w}")
    if not clean.strip():
        raise ValueError("nothing left to render after filtering unsupported chars")

    driver = VENDOR_DIR / DRIVER_NAME
    driver.write_text(_DRIVER_SRC)
    cmd = [sys.executable, DRIVER_NAME,
           "--style-dir", str(Path(style_dir).resolve()),
           "--text", clean,
           "--out", str(Path(out_png).resolve()),
           "--device", device,
           "--num-samples", str(num_samples)]
    proc = subprocess.run(cmd, cwd=VENDOR_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"HWT generation failed (exit {proc.returncode}):\n{tail}")
    if not Path(out_png).exists():
        raise RuntimeError("HWT generation reported success but produced no image")
    print(f"[hwt] wrote {out_png}")
    return {"image": Path(out_png), "text": clean, "warnings": warnings}
