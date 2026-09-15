#!/usr/bin/env python3
"""Write a multi-page ruled assignment from YOUR handwriting style + a text file.

This is the path for “make something that looks like my 35-page notebook”:
  * paper  = the same red-margin / blue-line student copy
  * English = HWT (fast enough for many pages; One-DM is too slow here)
  * math-heavy lines = handwriting *font* in blue ink so the page is not blank

It will not clone your exact letters or layout formulas like the Adobe Scan PDF.

    python src/textwritter/write_assignment.py \\
        --style samples/my_handwriting.png \\
        --text-file notes.txt \\
        --engine hwt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from textwritter.notebook_engine import (
    FONTS_DIR, INKS, PAGE_LEFT, PAGE_LINE_H, PAGE_RIGHT, PAGE_TOP, STYLES,
    lines_per_page, make_blank_ruled_page,
)
from textwritter.pipeline import _ENGINE_FILES, _load
from textwritter.runtime import OUT_DIR, print_runtime_card

# IAM-ish Latin the generators keep. Everything else is drawn with a font.
_LATIN = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,;:'\"?!()-/"
)


def wrap_lines(text: str, max_chars: int = 48) -> list[str]:
    """Notebook lines. A blank source line becomes one empty ruled line."""
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        para = raw.strip()
        if not para:
            out.append("")
            continue
        words, cur, n = para.split(), [], 0
        for w in words:
            extra = len(w) + (1 if cur else 0)
            if n + extra > max_chars and cur:
                out.append(" ".join(cur))
                cur, n = [w], len(w)
            else:
                cur.append(w)
                n += extra
        if cur:
            out.append(" ".join(cur))
    return out


def _latin_ratio(s: str) -> float:
    letters = [c for c in s if not c.isspace()]
    if not letters:
        return 1.0
    return sum(c in _LATIN for c in letters) / len(letters)


def _ink_overlay(gray: Image.Image, ink: tuple[int, int, int]) -> Image.Image:
    g = np.array(gray.convert("L"), dtype=np.float32)
    if g.mean() < 127:
        g = 255.0 - g
    strength = np.clip((210.0 - g) / 180.0, 0.0, 1.0)
    strength = np.where(g < 205, strength, 0.0)
    rgba = np.zeros((*g.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = ink
    rgba[..., 3] = (strength * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def paste_generated_line(
    page: Image.Image,
    line_img: Image.Image,
    baseline_y: int,
    ink: tuple[int, int, int],
) -> None:
    page_rgba = page.convert("RGBA")
    overlay = _ink_overlay(line_img, ink)
    max_w = PAGE_RIGHT - PAGE_LEFT - 16
    target_h = 40
    w, h = overlay.size
    if h < 1 or w < 1:
        return
    scale = target_h / h
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    if nw > max_w:
        scale = max_w / w
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    overlay = overlay.resize((nw, nh), Image.Resampling.LANCZOS)
    y = baseline_y - nh + 4
    page_rgba.alpha_composite(overlay, (PAGE_LEFT + 18, max(0, y)))
    page.paste(page_rgba.convert("RGB"))


def draw_font_line(
    page: Image.Image,
    text: str,
    baseline_y: int,
    ink: tuple[int, int, int],
    style_key: str = "kalam",
) -> None:
    info = STYLES.get(style_key, STYLES["kalam"])
    font_file = FONTS_DIR / info["file"]
    font = (
        ImageFont.truetype(str(font_file), info["default_size"])
        if font_file.exists()
        else ImageFont.load_default()
    )
    draw = ImageDraw.Draw(page)
    y = baseline_y - int(info["default_size"] * info["y_offset_ratio"])
    draw.text((PAGE_LEFT + 18, y), text[:80], fill=ink, font=font)


def write_assignment(
    text: str,
    style: str | Path | None,
    engine: str = "hwt",
    out_dir: Path | None = None,
    ink_name: str = "blue",
    max_chars: int = 48,
) -> dict:
    """Render `text` onto as many ruled pages as needed. Returns paths + notes."""
    chosen = print_runtime_card(engine)
    if chosen == "onedm":
        print("[assignment] long notebooks should use HWT (minutes, not hours). "
              "One-DM is still available if you passed --engine onedm.")
    eng = _load(_ENGINE_FILES[chosen], f"tw_engine_{chosen}")
    work = (out_dir or OUT_DIR / "assignment") / "_style"
    style_dir, source, warnings = eng.prepare_style(style, work)
    for w in warnings:
        print(f"[assignment] {w}")
    print(f"[assignment] style source: {source}")

    ink = INKS.get(ink_name, INKS["blue"])
    lines = wrap_lines(text, max_chars=max_chars)
    per = lines_per_page()
    dest = Path(out_dir or OUT_DIR / "assignment")
    dest.mkdir(parents=True, exist_ok=True)

    pages: list[Path] = []
    page_i = 0
    page = None
    used = 0
    font_lines = 0
    neural_lines = 0

    def flush():
        nonlocal page, page_i
        if page is None:
            return
        page_i += 1
        path = dest / f"page_{page_i:02d}.png"
        page.save(path)
        pages.append(path)
        print(f"[assignment] wrote {path}")
        page = None

    for idx, line in enumerate(lines):
        if used == 0:
            page = make_blank_ruled_page(page_no=str(page_i + 1))
        baseline = PAGE_TOP + used * PAGE_LINE_H
        if line:
            if _latin_ratio(line) >= 0.78:
                try:
                    raw = dest / f"_line_{idx:04d}.png"
                    kw = {"steps": 30} if chosen == "onedm" else {}
                    res = eng.generate(line, style_dir, raw, **kw)
                    paste_generated_line(page, Image.open(res["image"]), baseline, ink)
                    neural_lines += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[assignment] neural failed on line {idx}: {e}; font fallback")
                    draw_font_line(page, line, baseline, ink)
                    font_lines += 1
            else:
                draw_font_line(page, line, baseline, ink)
                font_lines += 1
        used += 1
        if used >= per:
            flush()
            used = 0
    flush()

    pdf_path = dest / "assignment.pdf"
    if pages:
        imgs = [Image.open(p).convert("RGB") for p in pages]
        imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=120)
        print(f"[assignment] pdf {pdf_path}")

    return {
        "pages": pages,
        "pdf": pdf_path if pages else None,
        "neural_lines": neural_lines,
        "font_lines": font_lines,
        "style_source": source,
        "engine": chosen,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-page ruled assignment in your style.")
    ap.add_argument("--text", default=None, help="assignment body (or use --text-file)")
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--style", default=None,
                    help="photo/PNG of YOUR writing (export a PDF page first)")
    ap.add_argument("--engine", default="hwt", choices=["auto", "onedm", "hwt"],
                    help="hwt is the right default for many pages")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ink", default="blue", choices=list(INKS))
    args = ap.parse_args()

    if args.text_file:
        body = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        body = args.text
    else:
        body = (
            "Assignment notes\n"
            "Tensor order is the number of independent axes needed to identify "
            "a single element in the tensor.\n"
            "A matrix has tensor order 2. The number of entries is the product "
            "of the shape dimensions.\n"
        )
        print("[assignment] no --text/--text-file; using a short demo paragraph")

    result = write_assignment(
        text=body,
        style=args.style,
        engine=args.engine,
        out_dir=args.out_dir,
        ink_name=args.ink,
    )
    print(f"\nengine={result['engine']}  neural_lines={result['neural_lines']}  "
          f"font_fallback_lines={result['font_lines']}")
    if result["pdf"]:
        print(f"Open: {result['pdf']}")


if __name__ == "__main__":
    main()
