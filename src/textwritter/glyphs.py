#!/usr/bin/env python3
"""Handwritten glyph bank — the symbols a handwriting font cannot draw.

Kalam & friends cover Latin only, so `∈ → Σ ∇ …` fall back to DejaVu and look
typed. This module closes that gap with YOUR pen:

  1. `make_sheet`  — print a sheet of labelled boxes (3 boxes per symbol, so the
                     same symbol is not identical every time it appears).
  2. `slice_sheet` — photograph/scan the filled sheet; corner registration marks
                     let it perspective-correct, then it cuts each box out.
  3. `bank`        — assignment_engine asks this for a glyph before falling back
                     to DejaVu.

    python src/textwritter/glyphs.py sheet
    python src/textwritter/glyphs.py slice --image samples/glyph_sheet_filled.jpg
    python src/textwritter/glyphs.py check
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from textwritter.runtime import ROOT  # noqa: E402

GLYPH_DIR = Path(os.environ.get("TW_GLYPH_DIR", str(ROOT / "assets" / "glyphs")))

# What to write, in sheet order. (char, spoken name) — the name is printed so
# there is no guessing which symbol a box wants.
NEEDED: list[tuple[str, str]] = [
    ("∈", "belongs to"),
    ("∉", "not in"),
    ("→", "arrow right"),
    ("←", "arrow left"),
    ("⇒", "implies"),
    ("⇔", "iff"),
    ("Σ", "sum (sigma)"),
    ("∏", "product (pi)"),
    ("∇", "nabla / del"),
    ("∂", "partial d"),
    ("√", "square root"),
    ("∞", "infinity"),
    ("∀", "for all"),
    ("∃", "there exists"),
    ("∅", "empty set"),
    ("∪", "union"),
    ("∩", "intersection"),
    ("⊂", "subset"),
    ("⊃", "superset"),
    ("≤", "less or equal"),
    ("≥", "greater or equal"),
    ("≠", "not equal"),
    ("≈", "approx"),
    ("×", "times"),
    ("·", "dot"),
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("δ", "delta small"),
    ("ε", "epsilon"),
    ("θ", "theta"),
    ("λ", "lambda"),
    ("μ", "mu"),
    ("σ", "sigma small"),
    ("π", "pi small"),
    ("Δ", "Delta big"),
    ("Θ", "Theta big"),
    ("Λ", "Lambda big"),
    ("Ω", "Omega big"),
    ("Φ", "Phi big"),
    ("Γ", "Gamma big"),
    ("η", "eta"),
    ("ℓ", "script l (loss)"),
    ("⊙", "odot / hadamard"),
    ("✓", "check"),
    ("⊤", "transpose / top"),
    ("∫", "integral"),
    ("±", "plus minus"),
]

if len({c for c, _ in NEEDED}) != len(NEEDED):  # a repeat would collide in cell_rects
    raise AssertionError("NEEDED has duplicate characters")

SAMPLES_PER_GLYPH = 3

# ── Sheet geometry (A4 at 200 dpi) ───────────────────────────────────────────
SHEET_W, SHEET_H = 1654, 2339
MARK_SIZE = 54
MARK_INSET = 70
BOX_W, BOX_H = 104, 104
LABEL_W = 92
GROUP_GAP = 22
ROW_GAP = 12
GRID_TOP = 258
GRID_LEFT = 84
GROUPS_PER_ROW = 3

BOX_OUTLINE = (205, 205, 210)   # light: thresholding drops it, pen ink survives
LABEL_COLOR = (90, 95, 110)


def _group_w() -> int:
    return LABEL_W + SAMPLES_PER_GLYPH * BOX_W


def cell_rects() -> dict[tuple[str, int], tuple[int, int, int, int]]:
    """(char, sample_index) -> box rect on the canonical sheet.

    The sheet writer and the slicer both call this, so they can never disagree.
    """
    rects: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    gw = _group_w()
    for i, (ch, _name) in enumerate(NEEDED):
        row, col = divmod(i, GROUPS_PER_ROW)
        gx = GRID_LEFT + col * (gw + GROUP_GAP)
        gy = GRID_TOP + row * (BOX_H + ROW_GAP)
        for s in range(SAMPLES_PER_GLYPH):
            x = gx + LABEL_W + s * BOX_W
            rects[(ch, s)] = (x, gy, x + BOX_W, gy + BOX_H)
    return rects


def _marks() -> list[tuple[int, int]]:
    """Registration mark top-left corners: TL, TR, BL, BR."""
    return [
        (MARK_INSET, MARK_INSET),
        (SHEET_W - MARK_INSET - MARK_SIZE, MARK_INSET),
        (MARK_INSET, SHEET_H - MARK_INSET - MARK_SIZE),
        (SHEET_W - MARK_INSET - MARK_SIZE, SHEET_H - MARK_INSET - MARK_SIZE),
    ]


def _symbol_font(size: int) -> ImageFont.FreeTypeFont:
    from textwritter.assignment_engine import fallback_font_path

    p = fallback_font_path()
    if p is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(p), size)


def make_sheet(out_path: Path | None = None) -> Path:
    """Draw the printable collection sheet."""
    out_path = Path(out_path or ROOT / "out" / "glyph_sheet.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (SHEET_W, SHEET_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for mx, my in _marks():
        draw.rectangle([mx, my, mx + MARK_SIZE, my + MARK_SIZE], fill=(0, 0, 0))

    title = _symbol_font(38)
    small = _symbol_font(19)
    sym = _symbol_font(38)
    draw.text((GRID_LEFT, 152), "Handwriting symbol sheet — Text_Writter",
              fill=(20, 20, 30), font=title)
    draw.text((GRID_LEFT, 200),
              "Write each symbol 3 times with a dark pen, inside the boxes.",
              fill=LABEL_COLOR, font=small)
    draw.text((GRID_LEFT, 226),
              "Do not touch the box edges. Keep the 4 black corner squares visible "
              "in the photo.", fill=LABEL_COLOR, font=small)

    rects = cell_rects()
    gw = _group_w()
    for i, (ch, name) in enumerate(NEEDED):
        row, col = divmod(i, GROUPS_PER_ROW)
        gx = GRID_LEFT + col * (gw + GROUP_GAP)
        gy = GRID_TOP + row * (BOX_H + ROW_GAP)
        draw.text((gx + 6, gy + 10), ch, fill=(20, 20, 30), font=sym)
        draw.text((gx + 6, gy + 62), name[:13], fill=LABEL_COLOR, font=small)
        for s in range(SAMPLES_PER_GLYPH):
            x0, y0, x1, y1 = rects[(ch, s)]
            draw.rectangle([x0, y0, x1, y1], outline=BOX_OUTLINE, width=2)

    if out_path.suffix.lower() == ".pdf":
        img.save(out_path, "PDF", resolution=200)
    else:
        img.save(out_path)
    print(f"[glyphs] sheet -> {out_path}")
    print(f"[glyphs] {len(NEEDED)} symbols x {SAMPLES_PER_GLYPH} boxes "
          f"= {len(NEEDED) * SAMPLES_PER_GLYPH} things to write")
    return out_path


# ── Slicing a filled sheet ───────────────────────────────────────────────────

def _find_marks(gray: np.ndarray) -> np.ndarray | None:
    """Locate the 4 filled corner squares; returns TL,TR,BL,BR centres."""
    import cv2

    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    h, w = gray.shape
    area_lo = (min(h, w) * 0.012) ** 2
    area_hi = (min(h, w) * 0.075) ** 2
    cands = []
    for i in range(1, n):
        x, y, bw_, bh_, area = stats[i]
        if not (area_lo < area < area_hi):
            continue
        if bh_ == 0 or not (0.65 < bw_ / bh_ < 1.55):
            continue
        if area / float(bw_ * bh_) < 0.7:  # squares are solid
            continue
        cands.append(cents[i])
    if len(cands) < 4:
        return None
    pts = np.array(cands, dtype="float32")
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
    picked = []
    for c in corners:
        d = np.linalg.norm(pts - c, axis=1)
        picked.append(pts[int(np.argmin(d))])
    out = np.array(picked, dtype="float32")
    if len({tuple(np.round(p)) for p in out}) < 4:
        return None
    return out


def slice_sheet(image: Path, out_dir: Path | None = None,
                min_ink_px: int = 40) -> dict:
    """Cut a filled sheet into per-glyph PNGs. Returns a small report."""
    import cv2

    out_dir = Path(out_dir or GLYPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read {image}")

    marks = _find_marks(gray)
    if marks is None:
        raise RuntimeError(
            "could not find the 4 black corner squares. Re-shoot the sheet: "
            "whole page in frame, flat, even light, no fingers over the corners.")

    # Warp so the photo matches the canonical sheet the boxes were drawn on.
    mk = np.array(_marks(), dtype="float32") + MARK_SIZE / 2.0
    matrix = cv2.getPerspectiveTransform(marks, mk)
    flat = cv2.warpPerspective(gray, matrix, (SHEET_W, SHEET_H),
                               flags=cv2.INTER_CUBIC, borderValue=255)

    rects = cell_rects()
    written, empty = [], []
    inset = 9
    for (ch, s), (x0, y0, x1, y1) in rects.items():
        cell = flat[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
        if cell.size == 0:
            continue
        # Pen ink is much darker than the printed guide box.
        ink = cell < 150
        if ink.sum() < min_ink_px:
            empty.append(ch)
            continue
        ys, xs = np.where(ink)
        pad = 3
        cy0 = max(0, ys.min() - pad)
        cy1 = min(cell.shape[0], ys.max() + pad + 1)
        cx0 = max(0, xs.min() - pad)
        cx1 = min(cell.shape[1], xs.max() + pad + 1)
        crop = cell[cy0:cy1, cx0:cx1]
        # Normalise to clean ink-on-white.
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        name = f"U+{ord(ch):04X}_{s}.png"
        cv2.imwrite(str(out_dir / name), crop)
        written.append(name)

    _bank_cached.cache_clear()
    aspect.cache_clear()
    print(f"[glyphs] wrote {len(written)} glyph image(s) to {out_dir}")
    present = {chr(int(p.stem.split('_')[0][2:], 16))
               for p in out_dir.glob("U+*.png")}
    have = sorted(present)
    missing = [ch for ch, _n in NEEDED if ch not in present]
    if missing:
        print(f"[glyphs] still empty: {' '.join(missing)}")
    if out_dir.resolve() != GLYPH_DIR.resolve():
        print(f"[glyphs] note: the renderer reads {GLYPH_DIR} — set "
              f"TW_GLYPH_DIR={out_dir} or slice without --out-dir to use these")
    return {"written": written, "blank_boxes": len(empty), "have": have,
            "missing": missing}


# ── Lookup used by the renderer ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _bank_cached() -> dict[str, tuple[Path, ...]]:
    out: dict[str, list[Path]] = {}
    if GLYPH_DIR.exists():
        for p in sorted(GLYPH_DIR.glob("U+*.png")):
            code = p.stem.split("_")[0][2:]
            try:
                ch = chr(int(code, 16))
            except ValueError:
                continue
            out.setdefault(ch, []).append(p)
    return {k: tuple(v) for k, v in out.items()}


def bank() -> dict[str, tuple[Path, ...]]:
    """char -> handwritten sample images (may be several per char)."""
    return _bank_cached()


def pick(ch: str) -> Path | None:
    """One random handwritten sample for `ch`, so repeats are not identical."""
    opts = bank().get(ch)
    return random.choice(opts) if opts else None


@lru_cache(maxsize=256)
def aspect(ch: str) -> float | None:
    """Mean width/height of this char's samples, for width measurement."""
    opts = bank().get(ch)
    if not opts:
        return None
    ratios = []
    for p in opts:
        try:
            with Image.open(p) as im:
                if im.height:
                    ratios.append(im.width / im.height)
        except Exception:  # noqa: BLE001
            continue
    return sum(ratios) / len(ratios) if ratios else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Handwritten symbol bank")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sheet = sub.add_parser("sheet", help="make the printable sheet")
    p_sheet.add_argument("--out", type=Path, default=None)

    p_slice = sub.add_parser("slice", help="cut a filled sheet into glyphs")
    p_slice.add_argument("--image", type=Path, required=True)
    p_slice.add_argument("--out-dir", type=Path, default=None)

    sub.add_parser("check", help="what the bank has so far")

    args = ap.parse_args()
    if args.cmd == "sheet":
        make_sheet(args.out)
    elif args.cmd == "slice":
        slice_sheet(args.image, args.out_dir)
    else:
        have = bank()
        print(f"[glyphs] dir: {GLYPH_DIR}")
        print(f"[glyphs] have {len(have)}/{len(NEEDED)} symbols")
        for ch, name in NEEDED:
            n = len(have.get(ch, ()))
            print(f"  {ch}  {name:<16} {'x' + str(n) if n else '— missing'}")


if __name__ == "__main__":
    main()
