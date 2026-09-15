#!/usr/bin/env python3
"""One-DM engine — the GPU (max-fidelity) creation path.

One-DM's official `test.py` is an *evaluation runner*: CUDA-only (NCCL), and it
generates every word in a corpus for every writer folder found on disk. This
adapter turns it into a single-photo, single-line tool:

  1. your style words (photo crops, or the bundled samples) are laid into a
     private data root as one writer ("me") — 64 px-high crops + Laplacian twins
     (the style inputs One-DM expects);
  2. the `iv_u` corpus file is replaced with just your target words;
  3. a private config points DATA_LOADER at that root;
  4. `torch.distributed.run --standalone` runs their test.py single-process;
  5. the per-word outputs are assembled into one line PNG.

The only external file it cannot synthesize is `data/unifont.pickle` (the printed
glyph conditions), which comes from the official One-DM English dataset bundle —
`--check` tells you what's present before any large download.

Engine contract (shared with 02_hwt/engine.py):
    generate(text, style_dir, out_png, device, steps) -> {"image": Path, "text": str}
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ENGINE_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from textwritter.style import prepare_style as _prepare_style  # noqa: E402
from textwritter.runtime import detect_environment, validate_engine  # noqa: E402

REPO_URL = "https://github.com/dailenson/One-DM"
VENDOR_DIR = _ENGINE_DIR / "_vendor" / "One-DM"
WRITER_ID = "me"
DATA_ROOT = VENDOR_DIR / "data_custom"
CONFIG_NAME = "IAM64_custom.yml"


# ---------------------------------------------------------------------------
# Vendor repo + required external assets
# ---------------------------------------------------------------------------

def _sh(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo(verbose: bool = True) -> Path:
    if not (VENDOR_DIR / "test.py").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        _sh(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    (VENDOR_DIR / "data").mkdir(exist_ok=True)
    if verbose:
        print(f"[onedm] vendor ready: {VENDOR_DIR}")
    return VENDOR_DIR


def _letters() -> str:
    """The trained character set (parsed from the vendored loader, statically)."""
    from textwritter.runtime import read_string_constant

    vendor = ensure_repo(verbose=False)
    return read_string_constant(vendor / "data_loader" / "loader.py", "letters")


def filter_words(text: str) -> tuple[list[str], list[str]]:
    allowed = set(_letters())
    warnings: list[str] = []
    if not allowed:
        return text.split(), warnings
    words, dropped = [], set()
    for w in text.split():
        kept = "".join(ch for ch in w if ch in allowed)
        dropped |= set(w) - allowed
        if kept:
            words.append(kept)
    if dropped:
        warnings.append(f"dropped unsupported character(s): {''.join(sorted(dropped))!r}")
    if not words:
        raise ValueError("no renderable words left after filtering")
    return words, warnings


def _content_asset(vendor: Path) -> Path:
    """data/unifont.pickle — fetched from the official English dataset bundle.

    The bundle arrives as English_data.zip inside the Drive folder, so extract it
    before looking for the pickle.
    """
    target = vendor / "data" / "unifont.pickle"
    if target.exists():
        return target
    from textwritter.env import CHECKPOINTS, get_checkpoint

    print("[onedm] unifont.pickle missing — fetching the official English data bundle")
    bundle = get_checkpoint("onedm_data")
    if bundle:
        hits = list(Path(bundle).rglob("unifont.pickle"))
        if not hits:
            import zipfile

            for z in Path(bundle).rglob("*.zip"):
                print(f"[onedm] extracting {z.name}")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent)
            hits = list(Path(bundle).rglob("unifont.pickle"))
        if hits:
            shutil.copy2(hits[0], target)
            print(f"[onedm] {hits[0]} -> {target}")
            return target
    info = CHECKPOINTS["onedm_data"]
    raise RuntimeError(
        "One-DM needs data/unifont.pickle from the official English dataset bundle, "
        "which could not be fetched automatically.\n"
        f"  Fix: download {info['note']} and drop unifont.pickle into "
        f"{vendor / 'data'}/, or set ONEDM_DATA_FOLDER_ID and re-run.\n"
        "  Or just use the HWT engine now: --engine hwt (runs on any machine).")


def checkpoint_path() -> Path:
    from textwritter.env import get_checkpoint

    ckpt = get_checkpoint("onedm_pretrained")
    if ckpt is None:
        raise RuntimeError(
            "One-DM checkpoint unavailable. Set ONEDM_CKPT_FOLDER_ID, or download "
            "One-DM-ckpt.pt from the repo README Model Zoo and drop it into "
            "checkpoints/. Or use --engine hwt now.")
    return ckpt


def info() -> dict:
    """Report readiness without downloading anything (used by --check)."""
    from textwritter.env import CHECKPOINTS, CKPT_ROOT

    env = detect_environment()
    return {
        "cuda": env.device == "cuda",
        "gpu": env.gpu_name,
        "vendor": (VENDOR_DIR / "test.py").exists(),
        "checkpoint": (CKPT_ROOT / CHECKPOINTS["onedm_pretrained"]["filename"]).exists(),
        "unifont": (VENDOR_DIR / "data" / "unifont.pickle").exists(),
        "doc": CHECKPOINTS["onedm_pretrained"]["note"],
    }


# ---------------------------------------------------------------------------
# Style → One-DM data layout
# ---------------------------------------------------------------------------

def _laplacian(gray):
    import cv2
    import numpy as np

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    peak = lap.max() or 1.0
    return (lap / peak * 255).astype("uint8")


def _lay_style(style_dir: Path) -> tuple[Path, int]:
    """Write 64 px-high word crops + Laplacian twins as the single writer folder."""
    import cv2

    style_out = DATA_ROOT / "IAM64-new" / "test" / WRITER_ID
    lap_out = DATA_ROOT / "IAM64_laplace" / "test" / WRITER_ID
    for d in (style_out, lap_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    kept = []
    for p in sorted(Path(style_dir).glob("*.png")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        if h < 2 or w < 2:
            continue
        img = cv2.resize(gray, (max(1, int(64 * w / h)), 64), interpolation=cv2.INTER_AREA)
        kept.append(img)
    if not kept:
        raise RuntimeError(f"no usable style word images in {style_dir}")

    # One-DM's sampler loop requires at least one style image wider than 128 px.
    if max(k.shape[1] for k in kept) <= 128:
        widest = max(kept, key=lambda k: k.shape[1])
        kept.append(cv2.resize(widest, (160, 64), interpolation=cv2.INTER_CUBIC))
        print("[onedm] widened a style word to satisfy One-DM's >128 px requirement")

    for i, img in enumerate(kept):
        cv2.imwrite(str(style_out / f"w{i:03d}.png"), img)
        cv2.imwrite(str(lap_out / f"w{i:03d}.png"), _laplacian(img))
    return style_out, len(kept)


def _write_config(vendor: Path, data_root: Path) -> Path:
    src = vendor / "configs" / "IAM64.yml"
    lines = src.read_text().splitlines()
    out = []
    for line in lines:
        if re.match(r"\s*IAMGE_PATH\s*:", line):
            out.append(f"  IAMGE_PATH: {data_root / 'IAM64-new'}")
        elif re.match(r"\s*STYLE_PATH\s*:", line):
            out.append(f"  STYLE_PATH: {data_root / 'IAM64-new'}")
        elif re.match(r"\s*LAPLACE_PATH\s*:", line):
            out.append(f"  LAPLACE_PATH: {data_root / 'IAM64_laplace'}")
        else:
            out.append(line)
    dst = vendor / "configs" / CONFIG_NAME
    dst.write_text("\n".join(out) + "\n")
    return dst


def prepare_style(style, work_dir: Path, verbose: bool = True):
    """Same contract as the HWT engine (raw word crops; generate() resizes).

    With no photo, the bundled samples are used. If they aren't on disk yet, they
    are extracted from the official HWT bundle (one-time download) rather than
    falling back to a printed sample.
    """
    def _extract() -> list:
        try:
            eng = importlib.util.spec_from_file_location(
                "tw_hwt_engine", _ENGINE_DIR.parent / "02_hwt" / "engine.py")
            mod = importlib.util.module_from_spec(eng)
            eng.loader.exec_module(mod)
            return mod.extract_iam_styles(verbose=verbose)
        except Exception as e:  # noqa: BLE001 - fall back to the printed sample
            if verbose:
                print(f"[onedm] could not fetch bundled styles ({e})")
            return []

    return _prepare_style(style, work_dir, extract=_extract, verbose=verbose)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _crop_ink(img):
    import cv2
    import numpy as np

    bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(bw > 0)
    if not len(ys):
        return img
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def _compose(words: list[str], word_images: dict[str, Path], out_png: Path) -> None:
    """Assemble per-word PNGs into one baseline-aligned line image."""
    import cv2
    import numpy as np

    tiles = []
    for w in words:
        p = word_images.get(w)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tiles.append(_crop_ink(img))
    if not tiles:
        raise RuntimeError("One-DM produced no word images to assemble")

    height = max(t.shape[0] for t in tiles)
    gap = 14
    total_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((height + 20, total_w + 20,), 255, dtype="uint8")
    x = 10
    for t in tiles:
        y = height - t.shape[0] + 10
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def generate(text: str, style_dir: Path, out_png: Path, device: str | None = None,
             steps: int = 50, seed: int = 0) -> dict:
    """Render `text` with One-DM. GPU-only. Returns the shared engine dict."""
    env = detect_environment()
    problem = validate_engine("onedm")
    if problem:
        raise RuntimeError(problem)
    dev = device or env.device

    vendor = ensure_repo()
    ckpt = checkpoint_path()
    _content_asset(vendor)

    words, warnings = filter_words(text)
    for w in warnings:
        print(f"[onedm] {w}")

    _, n_style = _lay_style(style_dir)
    print(f"[onedm] style: {n_style} word crops laid out as writer '{WRITER_ID}'")

    # Replace the iv_u corpus with just our words (their runner walks the whole file).
    corpus = vendor / "data" / "in_vocab.subset.tro.37"
    unique_words = list(dict.fromkeys(words))
    corpus.write_text("\n".join(unique_words) + "\n")
    cfg = _write_config(vendor, DATA_ROOT)

    raw_out = Path(out_png).resolve().parent / "_onedm_words"
    if raw_out.exists():
        shutil.rmtree(raw_out)
    raw_out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nnodes=1", "--nproc_per_node=1", "test.py",
           "--cfg", f"configs/{CONFIG_NAME}",
           "--one_dm", str(Path(ckpt).resolve()),
           "--generate_type", "iv_u",
           "--device", dev,
           "--sampling_timesteps", str(steps),
           "--sample_method", "ddim",
           "--dir", str(raw_out)]
    try:
        _sh(cmd, cwd=vendor)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"One-DM generation failed (exit {e.returncode}). Common causes: no "
            "CUDA/NCCL, or missing data/unifont.pickle. Try --engine hwt.") from e

    # Their runner writes to os.path.join(out, "<word>.png"), but under a directory
    # it derives from the writer id (it takes only the first character). Do not
    # assume that layout — just find the produced word images.
    word_images = {}
    for w in unique_words:
        hits = list(raw_out.rglob(f"{w}.png"))
        if hits:
            word_images[w] = hits[0]
    if not word_images:
        raise RuntimeError(
            f"One-DM wrote no outputs under {raw_out} (looked for "
            f"{len(unique_words)} word image(s)).")
    _compose(words, word_images, Path(out_png))
    print(f"[onedm] wrote {out_png}")
    return {"image": Path(out_png), "text": " ".join(words),
            "warnings": warnings, "word_images": word_images}
