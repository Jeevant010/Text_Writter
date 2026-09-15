#!/usr/bin/env python3
"""Assignment Notebook Engine — Renders structured handwritten assignment pages.

Produces convincing handwritten assignment pages on ruled notebook paper with:
  - Math notation: superscripts (^{...}), subscripts (_{...}), symbols (∈ → × ∏)
  - Structured layout: Problem headers, sub-parts (a,b,c), sub-sub (i,ii,iii)
  - Indentation, underlines, page headers (Date, Page No, Expt No)
  - Multi-page PDF output
  - Natural handwriting variations (baseline wobble, ink pressure, spacing jitter)

Usage:
    python assignment_engine.py content.txt output.pdf
    python assignment_engine.py content.txt output.pdf --style kalam --ink blue
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

# ── Font styles ──────────────────────────────────────────────────────────────

STYLES = {
    "kalam": {
        "name": "Kalam (Student Notebook)",
        "file": "Kalam.ttf",
        "default_size": 32,
        "line_height": 48,
        "y_offset_ratio": 0.80,
    },
    "caveat": {
        "name": "Caveat (Flowing Cursive)",
        "file": "Caveat.ttf",
        "default_size": 38,
        "line_height": 48,
        "y_offset_ratio": 0.76,
    },
    "indie": {
        "name": "Indie Flower (Casual Hand)",
        "file": "IndieFlower.ttf",
        "default_size": 30,
        "line_height": 48,
        "y_offset_ratio": 0.78,
    },
    "patrick": {
        "name": "Patrick Hand (Neat Notes)",
        "file": "PatrickHand.ttf",
        "default_size": 32,
        "line_height": 48,
        "y_offset_ratio": 0.80,
    },
}

INKS = {
    "blue": (24, 62, 132),
    "black": (26, 28, 32),
    "darkblue": (15, 38, 85),
    "red": (185, 30, 40),
}

# assets/fonts/ is gitignored, so a fresh clone (Colab especially) has no faces.
# All four are OFL on Google Fonts.
_FONT_URLS = {
    "Kalam.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf",
    "Caveat.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "IndieFlower.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "PatrickHand.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
}


def ensure_fonts(verbose: bool = True) -> list[str]:
    """Download any missing handwriting faces. Returns the names now present."""
    import urllib.request

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _FONT_URLS.items():
        target = FONTS_DIR / name
        if target.exists():
            continue
        try:
            if verbose:
                print(f"[fonts] downloading {name}")
            urllib.request.urlretrieve(url, target)
        except Exception as e:  # noqa: BLE001 - offline is survivable if one face exists
            print(f"[fonts] could not fetch {name}: {e}")
            target.unlink(missing_ok=True)
    return sorted(p.name for p in FONTS_DIR.glob("*.ttf"))

# ── Unicode math helpers ─────────────────────────────────────────────────────

# Handwriting fonts cover Latin only, so math glyphs (∈ → ∏ ≤ …) must come from a
# fallback face or they render as .notdef boxes. DejaVuSans ships inside
# matplotlib, so the same lookup works on a PC and on Colab.
_FALLBACK_FONTS = [
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf"),
    Path("C:/Windows/Fonts/seguisym.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]


def _matplotlib_dejavu() -> Path | None:
    try:
        import matplotlib
    except Exception:  # noqa: BLE001 - optional
        return None
    p = (Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
         / "DejaVuSans.ttf")
    return p if p.exists() else None


@lru_cache(maxsize=8)
def fallback_font_path() -> Path | None:
    """A face with math coverage, or None (then math glyphs may box)."""
    for cand in _FALLBACK_FONTS:
        if cand.exists():
            return cand
    return _matplotlib_dejavu()


@lru_cache(maxsize=32)
def _coverage(path: str) -> frozenset[int] | None:
    """Codepoints a font can draw. None = unknown, so assume it covers."""
    try:
        from fontTools.ttLib import TTFont

        return frozenset(TTFont(path, fontNumber=0).getBestCmap().keys())
    except Exception:  # noqa: BLE001 - fontTools optional/odd font
        return None


def _covers(path: Path | None, ch: str) -> bool:
    if path is None:
        return False
    cov = _coverage(str(path))
    return True if cov is None else ord(ch) in cov


@lru_cache(maxsize=64)
def _load_font(path_str: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path_str, size)


def glyph_runs(text: str, primary: Path) -> list[tuple[str, Path]]:
    """Split text into runs, each drawn by a font that owns those glyphs."""
    fb = fallback_font_path()
    runs: list[tuple[str, Path]] = []
    for ch in text:
        use = primary
        if ch.strip() and not _covers(primary, ch) and _covers(fb, ch):
            use = fb  # type: ignore[assignment]
        if runs and runs[-1][1] == use:
            runs[-1] = (runs[-1][0] + ch, use)
        else:
            runs.append((ch, use))
    return runs


# Superscripts/subscripts are drawn as smaller raised/lowered text rather than
# mapped to Unicode ⁵/ₖ characters — handwriting faces don't have those.
_SEG_RE = re.compile(r"\^\{([^}]*)\}|\^(\S)|_\{([^}]*)\}|_(\S)")
SUP_SCALE = 0.62
SUP_RISE = 0.42
SUB_DROP = 0.16


def split_words(text: str) -> list[str]:
    """Split on spaces, but never inside a `{...}` group (`R^{d1 × d2}` is one token)."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == " " and depth == 0:
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _flatten_markup(text: str) -> str:
    """Nested markup can't nest visually — `d_1` inside a superscript reads `d1`."""
    return re.sub(r"[\^_]\{([^}]*)\}|[\^_](\S)",
                  lambda m: m.group(1) if m.group(1) is not None else m.group(2),
                  text)


def parse_segments(text: str) -> list[tuple[str, str]]:
    """('R', 'normal'), ('4×6', 'sup') … from `R^{4\\times6}`."""
    text = replace_symbols(text)
    segs: list[tuple[str, str]] = []
    pos = 0
    for m in _SEG_RE.finditer(text):
        if m.start() > pos:
            segs.append((text[pos:m.start()], "normal"))
        if m.group(1) is not None:
            segs.append((_flatten_markup(m.group(1)), "sup"))
        elif m.group(2) is not None:
            segs.append((m.group(2), "sup"))
        elif m.group(3) is not None:
            segs.append((_flatten_markup(m.group(3)), "sub"))
        else:
            segs.append((m.group(4), "sub"))
        pos = m.end()
    if pos < len(text):
        segs.append((text[pos:], "normal"))
    return [s for s in segs if s[0]]


def measure_segments(segments: list[tuple[str, str]], font_path: Path,
                     size: int) -> int:
    total = 0
    for chunk, kind in segments:
        csize = max(8, int(size * SUP_SCALE)) if kind != "normal" else size
        for run, path in glyph_runs(chunk, font_path):
            f = _load_font(str(path), csize)
            total += int(f.getlength(run))
    return total


# Upright fallback glyphs next to slanted handwriting read as "typed", so the
# math runs get sheared to roughly the handwriting's lean.
FALLBACK_SHEAR = 0.16


def _draw_sheared(img: Image.Image, run: str, font: ImageFont.FreeTypeFont,
                  x: int, top: float, ink: tuple) -> None:
    """Draw `run` with a right lean so it sits beside handwriting glyphs."""
    w = int(font.getlength(run)) + 6
    h = int(font.size * 2.2)
    pad = int(abs(FALLBACK_SHEAR) * h) + 4
    tile = Image.new("RGBA", (w + pad, h), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((pad // 2, 2), run, fill=(*ink, 255), font=font)
    tile = tile.transform(tile.size, Image.AFFINE,
                          (1, FALLBACK_SHEAR, -FALLBACK_SHEAR * h / 2, 0, 1, 0),
                          resample=Image.BICUBIC)
    img.paste(tile, (int(x - pad // 2), int(top - 2)), tile)


def draw_segments(img: Image.Image, draw: ImageDraw.ImageDraw, x: int,
                  baseline_y: float, segments: list[tuple[str, str]],
                  font_path: Path, size: int, ink: tuple, y_ratio: float,
                  wobble: bool = True) -> int:
    """Draw rich text; returns the x cursor after the last glyph."""
    for chunk, kind in segments:
        csize = max(8, int(size * SUP_SCALE)) if kind != "normal" else size
        if kind == "sup":
            dy = -size * SUP_RISE
        elif kind == "sub":
            dy = size * SUB_DROP
        else:
            dy = 0.0
        for run, path in glyph_runs(chunk, font_path):
            f = _load_font(str(path), csize)
            jitter = random.uniform(-1.2, 1.2) if wobble else 0.0
            top = baseline_y - (csize * y_ratio) + dy + jitter
            run_ink = _jitter_ink(ink, wobble)
            if path == font_path:
                draw.text((x, top), run, fill=run_ink, font=f)
            else:
                _draw_sheared(img, run, f, x, top, run_ink)
            x += int(f.getlength(run))
    return x


# Common shorthand → Unicode replacements
SYMBOL_MAP = {
    r"\in": "∈",
    r"\notin": "∉",
    r"\times": "×",
    r"\cdot": "·",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\implies": "⇒",
    r"\iff": "⇔",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\sum": "Σ",
    r"\prod": "∏",
    r"\pi": "π",
    r"\theta": "θ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\sigma": "σ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\epsilon": "ε",
    r"\partial": "∂",
    r"\infty": "∞",
    r"\neq": "≠",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\approx": "≈",
    r"\subset": "⊂",
    r"\supset": "⊃",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\emptyset": "∅",
    r"\nabla": "∇",
    r"\sqrt": "√",
    "R": "R",  # Keep R as-is (used in R^n notation)
}


def replace_symbols(text: str) -> str:
    """`\\in` -> `∈`, `\\times` -> `×`, … (longest name first)."""
    for sym, uni in sorted(SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        if sym.startswith("\\"):
            text = text.replace(sym, uni)
    return text


# ── Page layout constants ────────────────────────────────────────────────────

PAGE_W, PAGE_H = 1200, 1700
MARGIN_LEFT = 145
MARGIN_RIGHT = 1100
MARGIN_TOP = 140
LINE_SPACING = 48  # pixels between ruled lines


def make_notebook_page(
    page_num: int = 1,
    show_header: bool = True,
) -> Image.Image:
    """Create a realistic ruled notebook page background."""
    paper_color = (252, 250, 245)
    img = Image.new("RGB", (PAGE_W, PAGE_H), paper_color)
    
    # Paper grain texture
    grain = np.random.normal(0, 1.5, (PAGE_H, PAGE_W, 3)).astype(np.int16)
    base = np.array(img).astype(np.int16)
    img = Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    # Red margin line (double line like real notebooks)
    draw.line([(MARGIN_LEFT, 0), (MARGIN_LEFT, PAGE_H)], fill=(220, 100, 100), width=2)
    draw.line([(MARGIN_LEFT - 5, 0), (MARGIN_LEFT - 5, PAGE_H)], fill=(235, 150, 150), width=1)
    
    # Blue horizontal ruled lines
    y = MARGIN_TOP
    while y < PAGE_H - 60:
        # Slight natural wobble in the ruling lines
        draw.line([(0, y), (PAGE_W, y)], fill=(180, 205, 230), width=1)
        y += LINE_SPACING
    
    # Header area
    if show_header:
        hdr_font_path = FONTS_DIR / "PatrickHand.ttf"
        if hdr_font_path.exists():
            hdr_font = ImageFont.truetype(str(hdr_font_path), 20)
        else:
            hdr_font = ImageFont.load_default()
        
        hdr_color = (140, 140, 160)
        # Top header boxes (like real assignment notebooks)
        draw.text((40, 22), "Expt. No. ___________", fill=hdr_color, font=hdr_font)
        draw.text((PAGE_W - 300, 22), "Date ________________", fill=hdr_color, font=hdr_font)
        draw.text((PAGE_W - 200, 55), f"Page No. __{page_num}__", fill=hdr_color, font=hdr_font)
        # Header line
        draw.line([(0, 90), (PAGE_W, 90)], fill=(180, 205, 230), width=1)
    
    # Bottom area — "Teacher's Signature"
    if show_header:
        sig_y = PAGE_H - 55
        draw.line([(PAGE_W // 2, sig_y + 20), (PAGE_W - 40, sig_y + 20)], fill=(140, 140, 160), width=1)
        draw.text((PAGE_W // 2 + 20, sig_y + 22), "Teacher's Signature", fill=(140, 140, 160), font=hdr_font)
    
    return img


def _jitter_ink(base_ink: tuple, wobble: bool = True) -> tuple:
    """Natural ink pressure variation."""
    if not wobble:
        return base_ink
    r, g, b = base_ink
    v = random.randint(-10, 10)
    return (max(0, min(255, r + v)), max(0, min(255, g + v)), max(0, min(255, b + v)))


def _detect_indent(line: str) -> tuple[int, str]:
    """Detect indentation level from leading spaces/tabs."""
    stripped = line.lstrip()
    indent_chars = len(line) - len(stripped)
    # Each 2-3 spaces or 1 tab = 1 indent level
    indent_level = indent_chars // 3 if indent_chars > 0 else 0
    return indent_level, stripped


def render_assignment(
    content: str,
    out_path: str | Path,
    style_key: str = "kalam",
    ink_name: str = "blue",
    wobble: bool = True,
    title: str | None = None,
) -> dict:
    """Render a full assignment to multi-page PNG images or PDF.
    
    Content format:
        Plain text with simple markup:
        - ^{...} for superscripts, _{...} for subscripts
        - \\in, \\times, \\to, etc. for math symbols
        - Leading spaces/tabs for indentation
        - Blank lines for extra spacing
        - Lines starting with '---' for underlines/separators
        - Lines starting with '#' for headers/titles (underlined)
    
    Returns dict with list of output page paths.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load font
    style = STYLES.get(style_key, STYLES["kalam"])
    font_path = FONTS_DIR / style["file"]
    font_size = style["default_size"]
    y_ratio = style["y_offset_ratio"]
    
    if not font_path.exists():
        ensure_fonts()
    if not font_path.exists():
        raise FileNotFoundError(
            f"handwriting font missing and could not be downloaded: {font_path}. "
            f"Put {style['file']} into {FONTS_DIR}/ by hand.")
    header_size = int(font_size * 1.15)
    
    base_ink = INKS.get(ink_name, INKS["blue"])
    line_h = style["line_height"]
    
    # Split content into lines and process math
    raw_lines = content.split("\n")
    
    # Calculate usable lines per page
    usable_top = MARGIN_TOP + 10
    usable_bottom = PAGE_H - 100
    lines_per_page = (usable_bottom - usable_top) // line_h
    
    def _wrap(kind: str, raw_text: str, indent_level: int, size: int) -> None:
        """Break one source line into page-width lines, measured after math markup."""
        indent_px = MARGIN_LEFT + 25 + indent_level * 40
        max_width = MARGIN_RIGHT - indent_px
        words = split_words(raw_text)
        current: list[str] = []
        for word in words:
            if not word:
                continue
            trial = " ".join(current + [word])
            if current and measure_segments(parse_segments(trial), font_path,
                                            size) > max_width:
                processed_lines.append((kind, indent_level, " ".join(current)))
                # Continuation lines of a wrapped header are body text.
                kind = "text" if kind == "header" else kind
                current = [word]
            else:
                current.append(word)
        if current:
            processed_lines.append((kind, indent_level, " ".join(current)))

    # Pre-process: wrap long lines to fit page width
    processed_lines = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            processed_lines.append(("blank", 0, ""))
            continue

        indent_level, stripped = _detect_indent(raw_line)

        if stripped.startswith("#"):
            _wrap("header", stripped.lstrip("# ").strip(), indent_level, header_size)
            continue

        if stripped.startswith("---"):
            processed_lines.append(("sep", 0, ""))
            continue

        _wrap("text", stripped, indent_level, font_size)
    
    # Render pages
    pages = []
    line_idx = 0
    page_num = 1
    
    while line_idx < len(processed_lines):
        img = make_notebook_page(page_num=page_num)
        draw = ImageDraw.Draw(img)
        
        current_y_slot = 0  # Which ruled line we're on
        
        while line_idx < len(processed_lines) and current_y_slot < lines_per_page:
            kind, indent, text = processed_lines[line_idx]
            
            baseline_y = usable_top + current_y_slot * line_h
            
            if kind == "blank":
                current_y_slot += 1
                line_idx += 1
                continue
            
            if kind == "sep":
                sep_y = baseline_y + line_h // 2
                draw.line(
                    [(MARGIN_LEFT + 30, sep_y), (MARGIN_RIGHT - 30, sep_y)],
                    fill=_jitter_ink(base_ink, wobble), width=1,
                )
                current_y_slot += 1
                line_idx += 1
                continue
            
            indent_px = MARGIN_LEFT + 25 + indent * 40
            
            if kind == "header":
                ink = _jitter_ink(base_ink, wobble)
                wobble_y = random.uniform(-1.5, 1.5) if wobble else 0
                end_x = draw_segments(
                    img, draw, indent_px, baseline_y + wobble_y,
                    parse_segments(text), font_path, header_size, base_ink,
                    y_ratio, wobble,
                )
                draw.line([(indent_px, baseline_y + 2), (end_x, baseline_y + 2)],
                          fill=ink, width=1)
                current_y_slot += 1
                line_idx += 1
                continue
            
            # Normal text — word by word so spacing and baseline can wander
            space_w = int(_load_font(str(font_path), font_size).getlength(" "))
            x = indent_px
            for word in split_words(text):
                if not word:
                    continue

                is_underlined = (word.startswith("__") and word.endswith("__")
                                 and len(word) > 4)
                display_word = word[2:-2] if is_underlined else word

                wobble_y = random.uniform(-1.8, 1.8) if wobble else 0
                start_x = x
                x = draw_segments(
                    img, draw, x, baseline_y + wobble_y,
                    parse_segments(display_word), font_path, font_size, base_ink,
                    y_ratio, wobble,
                )
                if is_underlined:
                    draw.line([(start_x, baseline_y + 2), (x, baseline_y + 2)],
                              fill=_jitter_ink(base_ink, wobble), width=1)

                jitter_x = random.uniform(-1.0, 1.5) if wobble else 0
                x += int(space_w + jitter_x)
            
            current_y_slot += 1
            line_idx += 1
        
        # Save page
        if str(out_path).endswith(".pdf"):
            pages.append(img)
        else:
            if len(processed_lines) <= lines_per_page * 1.1:
                # Single page — save directly
                img.save(out_path)
                pages.append(out_path)
            else:
                page_path = out_path.parent / f"{out_path.stem}_page{page_num:02d}{out_path.suffix}"
                img.save(page_path)
                pages.append(page_path)
        
        page_num += 1
    
    # If PDF output, combine all pages
    if str(out_path).endswith(".pdf") and pages:
        if len(pages) == 1:
            pages[0].save(out_path, "PDF", resolution=150)
        else:
            pages[0].save(
                out_path, "PDF", resolution=150,
                save_all=True, append_images=pages[1:],
            )
        page_paths = [out_path]
    else:
        page_paths = pages
    
    return {
        "pages": page_paths,
        "total_pages": page_num - 1,
        "style": style["name"],
        "ink": ink_name,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Render handwritten assignment pages")
    ap.add_argument("input", help="Input text file with assignment content")
    ap.add_argument("output", help="Output file path (.png or .pdf)")
    ap.add_argument("--style", default="kalam", choices=list(STYLES.keys()))
    ap.add_argument("--ink", default="blue", choices=list(INKS.keys()))
    ap.add_argument("--no-wobble", action="store_true", help="Disable handwriting wobble")
    args = ap.parse_args()
    
    content = Path(args.input).read_text(encoding="utf-8")
    result = render_assignment(
        content, args.output,
        style_key=args.style,
        ink_name=args.ink,
        wobble=not args.no_wobble,
    )
    print(f"Done: {result['total_pages']} page(s) → {args.output}")
    print(f"Style: {result['style']}, Ink: {result['ink']}")


if __name__ == "__main__":
    main()
