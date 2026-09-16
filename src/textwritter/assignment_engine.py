#!/usr/bin/env python3
"""Assignment Notebook Engine — Renders structured handwritten assignment pages.

Produces convincing handwritten assignment pages on ruled notebook paper with:
  - Math notation: superscripts (^{...}), subscripts (_{...}), symbols (∈ → × ∏)
  - Structured layout: Problem headers, sub-parts (a,b,c), sub-sub (i,ii,iii)
  - Indentation, underlines, page headers (Date, Page No, Expt No)
  - Multi-page PDF output
  - Natural handwriting variations (baseline wobble, ink pressure, spacing jitter)

Usage:
    PYTHONPATH=src python assignment_engine.py content.txt output.pdf --style caveat --ink blue

Draw whole words (Caveat ligatures). Sit baselines on the ruled grid.
See AGENTS.md and docs/ASSIGNMENT_PAGES.md.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
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


@lru_cache(maxsize=4096)
def _pil_has_glyph(path_str: str, ch: str) -> bool:
    """Ask FreeType directly: does this face draw `ch`, or its .notdef box?

    Used when fontTools is unavailable (common on a bare Colab runtime).
    U+FFFF is a non-character, so its mask is whatever the face shows for
    "missing" — anything that matches it is missing too.
    """
    def bitmap(font, text: str) -> bytes:
        img = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(img).text((8, 8), text, fill=255, font=font)
        return img.tobytes()

    try:
        f = _load_font(path_str, 40)
        ref = bitmap(f, "\uffff")
        got = bitmap(f, ch)
    except Exception:  # noqa: BLE001 - odd font, assume it works
        return True
    blank = bytes(96 * 96)
    return got != ref and got != blank


def _covers(path: Path | None, ch: str) -> bool:
    if path is None:
        return False
    cov = _coverage(str(path))
    if cov is not None:
        return ord(ch) in cov
    return _pil_has_glyph(str(path), ch)


@lru_cache(maxsize=64)
def _load_font(path_str: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path_str, size)


# Caveat's cursive capital L is a bare loop that reads as an opening bracket, so
# `∂L/∂w` looks like `∂(/∂w`. Draw those few letters from the fallback face.
CONFUSABLE = {"Caveat.ttf": set("L")}


def glyph_runs(text: str, primary: Path) -> list[tuple[str, Path]]:
    """Split text into runs, each drawn by a font that owns those glyphs."""
    fb = fallback_font_path()
    swap = CONFUSABLE.get(primary.name, frozenset())
    runs: list[tuple[str, Path]] = []
    for ch in text:
        use = primary
        if ch in swap and _covers(fb, ch):
            use = fb  # type: ignore[assignment]
        elif ch.strip() and not _covers(primary, ch) and _covers(fb, ch):
            use = fb  # type: ignore[assignment]
        if runs and runs[-1][1] == use:
            runs[-1] = (runs[-1][0] + ch, use)
        else:
            runs.append((ch, use))
    return runs


# Superscripts/subscripts are drawn as smaller raised/lowered text rather than
# mapped to Unicode ⁵/ₖ characters — handwriting faces don't have those.
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


def _group_at(text: str, i: int) -> tuple[str, int]:
    """Read the argument of a `^`/`_` starting at `i`; returns (body, next index).

    `{...}` is read with brace counting so `w^{(2)}` survives inside another
    group; anything else takes just the one character.
    """
    if i < len(text) and text[i] == "{":
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j], j + 1
            j += 1
        return text[i + 1:], len(text)  # unbalanced; take the rest
    if i < len(text):
        return text[i], i + 1
    return "", i


def _flatten_markup(text: str) -> str:
    """One level of raising is all we can draw, so `w^{(2)}` inside a
    subscript flattens to `w(2)` rather than nesting a third size."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "^_":
            body, i = _group_at(text, i + 1)
            out.append(_flatten_markup(body))
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def parse_segments(text: str) -> list[tuple[str, str]]:
    """('R', 'normal'), ('4×6', 'sup') … from `R^{4\\times6}`."""
    text = replace_symbols(text)
    segs: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "^_" and i + 1 < len(text):
            body, nxt = _group_at(text, i + 1)
            if body:
                if buf:
                    segs.append(("".join(buf), "normal"))
                    buf = []
                segs.append((_flatten_markup(body),
                             "sup" if ch == "^" else "sub"))
                i = nxt
                continue
        buf.append(ch)
        i += 1
    if buf:
        segs.append(("".join(buf), "normal"))
    return [s for s in segs if s[0]]


def measure_segments(segments: list[tuple[str, str]], font_path: Path,
                     size: int) -> int:
    from textwritter.glyphs import aspect

    total = 0
    for chunk, kind in segments:
        csize = max(8, int(size * SUP_SCALE)) if kind != "normal" else size
        for run, path in glyph_runs(chunk, font_path):
            f = _load_font(str(path), csize)
            if path == font_path:
                total += int(f.getlength(run))
                continue
            for ch in run:
                ratio = aspect(ch)
                if ratio is None and ch in _PEN_SHAPES:
                    total += int(csize * 1.13)
                elif ratio is None:
                    total += int(f.getlength(ch))
                else:
                    h_ratio = _GLYPH_METRICS.get(ch, _GLYPH_DEFAULT)[0]
                    total += int(csize * h_ratio * ratio) + max(2, csize // 12)
    return total


# Upright fallback glyphs next to slanted handwriting read as "typed", so the
# math runs get sheared to roughly the handwriting's lean.
FALLBACK_SHEAR = 0.16

# Placement for handwritten symbol images: (height / font size, how far the
# bottom sits below the baseline / font size). Arrows float at mid x-height,
# big operators overhang, most symbols sit on the line.
_GLYPH_METRICS: dict[str, tuple[float, float]] = {
    "→": (0.44, 0.13), "←": (0.44, 0.13), "⇒": (0.44, 0.13), "⇔": (0.44, 0.13),
    "Σ": (0.90, -0.06), "∏": (0.90, -0.06), "√": (0.90, -0.04),
    "∇": (0.70, 0.0), "∂": (0.78, -0.10),
    "≤": (0.52, 0.04), "≥": (0.52, 0.04), "≠": (0.62, 0.0), "≈": (0.30, 0.16),
    "×": (0.34, 0.14), "·": (0.12, 0.14), "∞": (0.34, 0.14),
    "β": (0.80, -0.18), "γ": (0.60, -0.18), "μ": (0.58, -0.16),
    "σ": (0.48, 0.0), "λ": (0.72, 0.0), "θ": (0.78, 0.0), "δ": (0.74, 0.0),
    "α": (0.46, 0.0), "ε": (0.46, 0.0), "π": (0.46, 0.0), "Δ": (0.70, 0.0),
}
_GLYPH_DEFAULT = (0.58, 0.0)


def _draw_handwritten_glyph(mask: Image.Image, ch: str, x: int,
                            baseline_y: float, size: int,
                            pressure: int) -> int | None:
    """Stamp one of the user's own symbol scans. Returns advance, or None."""
    from textwritter.glyphs import pick

    path = pick(ch)
    if path is None:
        return None
    try:
        src = Image.open(path).convert("L")
    except Exception:  # noqa: BLE001 - a bad crop should not kill the page
        return None

    h_ratio, drop = _GLYPH_METRICS.get(ch, _GLYPH_DEFAULT)
    target_h = max(4, int(size * h_ratio))
    scale = target_h / max(1, src.height)
    target_w = max(3, int(src.width * scale))
    src = src.resize((target_w, target_h), Image.LANCZOS)

    arr = np.asarray(src, dtype=np.float32)
    coverage = np.clip((215.0 - arr) / 120.0, 0.0, 1.0) * 255.0
    tile = Image.fromarray(coverage.astype(np.uint8), "L")

    _stamp(mask, tile, x, baseline_y + size * drop - target_h, pressure)
    return target_w + max(2, size // 12)


def _draw_sheared(mask: Image.Image, run: str, font: ImageFont.FreeTypeFont,
                  x: int, top: float, pressure: int) -> None:
    """Draw `run` with a right lean so it sits beside handwriting glyphs."""
    w = int(font.getlength(run)) + 6
    h = int(font.size * 2.2)
    pad = int(abs(FALLBACK_SHEAR) * h) + 4
    tile = Image.new("L", (w + pad, h), 0)
    ImageDraw.Draw(tile).text((pad // 2, 2), run, fill=255, font=font)
    tile = tile.transform(tile.size, Image.AFFINE,
                          (1, FALLBACK_SHEAR, -FALLBACK_SHEAR * h / 2, 0, 1, 0),
                          resample=Image.BICUBIC)
    _stamp(mask, tile, x - pad // 2, top - 2, pressure)


# Variation lives at the *word* level. Per-letter rotate/scale snaps Caveat's
# joins (the font's whole reason to exist) and reads as typed glyphs on a line.
WORD_ANGLE = 0.55     # degrees, peak — a written word leans a little
WORD_SCALE = 0.018    # fraction, peak
WORD_SHIFT = 0.35     # px, peak


def _stamp(mask: Image.Image, tile: Image.Image, x: float, y: float,
           pressure: int) -> None:
    """Add a glyph tile to the page-wide ink coverage mask (max-blend)."""
    box = (int(x), int(y))
    region = mask.crop((box[0], box[1], box[0] + tile.width, box[1] + tile.height))
    if pressure < 255:
        tile = tile.point(lambda v: v * pressure // 255)
    mask.paste(ImageChops.lighter(region, tile), box)


def _draw_connected(mask: Image.Image, run: str, font: ImageFont.FreeTypeFont,
                    x: float, top: float, pressure: int) -> None:
    """Stamp one connected string so ligatures and joins stay intact."""
    pad = 10
    w = int(font.getlength(run)) + pad * 2
    h = int(font.size * 2.2) + pad * 2
    tile = Image.new("L", (max(1, w), max(1, h)), 0)
    ImageDraw.Draw(tile).text((pad, pad), run, fill=255, font=font)

    scale = 1.0 + random.uniform(-WORD_SCALE, WORD_SCALE)
    if abs(scale - 1.0) > 0.004:
        tile = tile.resize((max(1, int(tile.width * scale)),
                            max(1, int(tile.height * scale))), Image.LANCZOS)
    tile = tile.rotate(random.uniform(-WORD_ANGLE, WORD_ANGLE),
                       resample=Image.BICUBIC, expand=False)
    dx = random.uniform(-WORD_SHIFT, WORD_SHIFT)
    dy = random.uniform(-WORD_SHIFT, WORD_SHIFT)
    _stamp(mask, tile, x - pad + dx, top - pad + dy, pressure)


def _draw_run(mask: Image.Image, mdraw: ImageDraw.ImageDraw, run: str,
              font: ImageFont.FreeTypeFont, x: float, top: float,
              pressure: int, humanize: bool) -> float:
    """Draw a run of same-font text into the ink mask; returns the new x."""
    adv = font.getlength(run)
    if not run.strip():
        return x + adv
    if not humanize:
        mdraw.text((x, top), run, fill=pressure, font=font)
        return x + adv
    _draw_connected(mask, run, font, x, top, _jitter_pressure(pressure))
    return x + adv


def draw_segments(mask: Image.Image, mdraw: ImageDraw.ImageDraw, x: int,
                  baseline_y: float, segments: list[tuple[str, str]],
                  font_path: Path, size: int, pressure: int, y_ratio: float,
                  wobble: bool = True) -> int:
    """Draw rich text into the ink mask; returns the x cursor after it."""
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
            ascent, _ = f.getmetrics()
            # Sit on the ruled line: PIL draws from the glyph box top, not baseline.
            top = baseline_y - ascent + dy
            run_p = _jitter_pressure(pressure) if wobble else pressure
            if path == font_path:
                x = int(_draw_run(mask, mdraw, run, f, x, top, run_p, wobble))
                continue
            # Fallback run: the user's own scans, else a drawn pen shape, else
            # the sheared font glyph.
            base = baseline_y + dy
            for ch in run:
                advance = _draw_handwritten_glyph(mask, ch, x, base, csize, run_p)
                if advance is None and ch in _PEN_SHAPES:
                    advance = _PEN_SHAPES[ch](mdraw, x, base, csize)
                if advance is None:
                    _draw_sheared(mask, ch, f, x, top, run_p)
                    advance = int(f.getlength(ch))
                x += advance
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
    r"\int": "∫",
    r"\pm": "±",
    r"\ldots": "…",
    r"\cdots": "…",
    r"\eta": "η",
    r"\ell": "ℓ",
    r"\odot": "⊙",
    r"\top": "⊤",
    r"\propto": "∝",
    # Capital Greek — common in DL notation (Σ loss, Δ weights, Θ params).
    r"\Sigma": "Σ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Omega": "Ω",
    r"\Phi": "Φ",
    r"\Gamma": "Γ",
    r"\Pi": "∏",
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
    
    # Blue horizontal ruled lines. Keep them straight and on a fixed grid so
    # the writing can sit on them; a scan tilt later is enough "paper" noise.
    y = MARGIN_TOP
    while y < PAGE_H - 60:
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


def apply_scan_look(img: Image.Image, seed: int | None = None) -> Image.Image:
    """Make a clean render look like a phone/Adobe Scan capture of paper.

    A perfectly axis-aligned, evenly-lit page is the giveaway even when the
    writing is good, so: tiny rotation, uneven lighting, softness, grain.
    """
    rng = random.Random(seed)
    w, h = img.size

    angle = rng.uniform(-0.22, 0.22)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                     fillcolor=(252, 250, 245))

    arr = np.asarray(img).astype(np.float32)
    # Lighting gradient: brighter on one side, slight corner falloff.
    gx = np.linspace(rng.uniform(0.90, 0.99), rng.uniform(0.97, 1.03), w,
                     dtype=np.float32)
    gy = np.linspace(rng.uniform(0.95, 1.01), rng.uniform(0.90, 0.99), h,
                     dtype=np.float32)
    shade = np.outer(gy, gx)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    shade *= 1.0 - 0.055 * r
    arr *= shade[..., None]

    arr += np.random.normal(0, 2.1, arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(radius=0.18))


BASE_PRESSURE = 240


def _hand_line(mdraw: ImageDraw.ImageDraw, x0: float, x1: float, y: float,
               wobble: bool, width: int = 2) -> None:
    """A pen line drawn in short segments so it is never perfectly straight."""
    x0, x1 = float(x0), float(x1)
    if x1 <= x0:
        return
    steps = max(2, int((x1 - x0) // 26))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        jx = random.uniform(-1.5, 1.5) if wobble else 0.0
        jy = random.uniform(-1.4, 1.4) if wobble else 0.0
        pts.append((x0 + (x1 - x0) * t + jx, y + jy))
    mdraw.line(pts, fill=_jitter_pressure(), width=width, joint="curve")


def _underline(mdraw: ImageDraw.ImageDraw, x0: float, x1: float, y: float,
               wobble: bool) -> None:
    _hand_line(mdraw, x0, x1, y, wobble)


def _strike(mdraw: ImageDraw.ImageDraw, x0: float, x1: float, y: float,
            wobble: bool) -> None:
    _hand_line(mdraw, x0 - 2, x1 + 2, y, wobble)


# Pressure drifts as you write rather than jumping per letter, so it is a random
# walk shared across the page instead of independent noise per glyph.
_PRESSURE = {"v": float(BASE_PRESSURE)}


def _wobbly(pts: list[tuple[float, float]], amount: float = 1.1
            ) -> list[tuple[float, float]]:
    return [(x + random.uniform(-amount, amount), y + random.uniform(-amount, amount))
            for x, y in pts]


def _pen_arrow(mdraw: ImageDraw.ImageDraw, x: float, baseline: float, size: int,
               back: bool = False) -> int:
    """Hand-drawn arrow. A font arrow is dead straight, which reads as typed."""
    w = size * 0.95
    y = baseline - size * 0.30
    stroke = max(2, size // 13)
    p = _jitter_pressure()
    x0, x1 = x, x + w
    mid = (y + random.uniform(-1.4, 1.4))
    mdraw.line(_wobbly([(x0, y), (x0 + w * 0.5, mid), (x1, y)]),
               fill=p, width=stroke, joint="curve")
    tip = (x0, y) if back else (x1, y)
    d = 1 if back else -1
    for dy in (-1, 1):
        mdraw.line(_wobbly([tip, (tip[0] + d * size * 0.26,
                                  tip[1] + dy * size * 0.20)]),
                   fill=p, width=stroke, joint="curve")
    return int(w + size * 0.18)


def _pen_in(mdraw: ImageDraw.ImageDraw, x: float, baseline: float,
            size: int) -> int:
    """`∈` as a bowl plus a bar, drawn rather than typeset."""
    h = size * 0.58
    top = baseline - h
    w = h * 0.80
    stroke = max(2, size // 13)
    p = _jitter_pressure()
    mdraw.arc([x, top, x + w * 1.6, baseline], start=35, end=325, fill=p,
              width=stroke)
    bar_y = top + h * 0.52
    mdraw.line(_wobbly([(x + w * 0.16, bar_y), (x + w * 0.95, bar_y)], 0.8),
               fill=p, width=stroke, joint="curve")
    return int(w + size * 0.20)


# Symbols worth drawing by hand because they appear constantly and a font
# renders them too perfectly.
_PEN_SHAPES = {
    "→": lambda d, x, b, s: _pen_arrow(d, x, b, s, back=False),
    "←": lambda d, x, b, s: _pen_arrow(d, x, b, s, back=True),
    "∈": _pen_in,
}


def _jitter_pressure(pressure: int = BASE_PRESSURE) -> int:
    """How hard the pen pressed, drifting smoothly from the last glyph."""
    v = _PRESSURE["v"] + random.uniform(-7.0, 7.0)
    # Pull gently back toward the requested level so it cannot wander off.
    v += (pressure - v) * 0.12
    _PRESSURE["v"] = max(198.0, min(255.0, v))
    return int(_PRESSURE["v"])


def composite_ink(paper: Image.Image, mask: Image.Image, ink_color: tuple,
                  bleed: float = 0.35) -> Image.Image:
    """Lay the ink coverage mask into the paper like wet ink, not vector text.

    Crisp font antialiasing is a dead giveaway. Blurring the coverage and then
    re-curving it gives soft edges with a dark core, plus stroke-width variation,
    and a low-frequency field makes the ink flow lighter and darker down the page.
    """
    m = mask.filter(ImageFilter.GaussianBlur(bleed))
    a = np.asarray(m, dtype=np.float32) / 255.0
    a = np.clip(a * 1.30, 0.0, 1.0) ** 1.45

    h, w = a.shape
    # Smooth random field ~= ink flow / pen angle drifting as you write.
    small = np.random.default_rng().normal(0.0, 1.0, (max(2, h // 48),
                                                     max(2, w // 48)))
    flow = np.asarray(Image.fromarray(small.astype(np.float32), "F")
                      .resize((w, h), Image.BICUBIC))
    a *= np.clip(1.0 + 0.16 * flow, 0.7, 1.25)
    a = np.clip(a, 0.0, 1.0)

    arr = np.asarray(paper, dtype=np.float32)
    ink = np.array(ink_color, dtype=np.float32)
    # Where coverage is thin the paper shows through and the hue lifts slightly.
    ink_field = ink[None, None, :] + (1.0 - a[..., None]) * 26.0
    out = arr * (1.0 - a[..., None]) + ink_field * a[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
    scan: bool = True,
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
    # Writing must land on the same grid the blue rules are drawn on.
    line_h = LINE_SPACING
    
    # Split content into lines and process math
    raw_lines = content.split("\n")
    
    # Calculate usable lines per page
    usable_top = MARGIN_TOP
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
        # All pen strokes accumulate in one coverage mask, composited per page.
        mask = Image.new("L", img.size, 0)
        mdraw = ImageDraw.Draw(mask)
        
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
                mdraw.line(
                    [(MARGIN_LEFT + 30, sep_y), (MARGIN_RIGHT - 30, sep_y)],
                    fill=_jitter_pressure(), width=2,
                )
                current_y_slot += 1
                line_idx += 1
                continue
            
            # Sit on the ruled line. A real notebook line is the baseline;
            # floating above it is the "typed onto a template" look.
            indent_px = MARGIN_LEFT + 25 + indent * 40
            if wobble:
                indent_px += int(random.uniform(-2, 3))
            slope = random.uniform(-0.0022, 0.0018) if wobble else 0.0
            
            if kind == "header":
                wobble_y = random.uniform(-0.4, 0.4) if wobble else 0
                end_x = draw_segments(
                    mask, mdraw, indent_px, baseline_y + wobble_y,
                    parse_segments(text), font_path, header_size,
                    BASE_PRESSURE, y_ratio, wobble,
                )
                _underline(mdraw, indent_px, end_x, baseline_y + 3, wobble)
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
                # ~~word~~ = crossed out, the way a real page has corrections
                is_struck = (display_word.startswith("~~")
                             and display_word.endswith("~~")
                             and len(display_word) > 4)
                if is_struck:
                    display_word = display_word[2:-2]

                wobble_y = random.uniform(-0.45, 0.45) if wobble else 0
                wobble_y += slope * (x - indent_px)
                word_size = font_size
                start_x = x
                x = draw_segments(
                    mask, mdraw, x, baseline_y + wobble_y,
                    parse_segments(display_word), font_path, word_size,
                    BASE_PRESSURE, y_ratio, wobble,
                )
                if is_underlined:
                    _underline(mdraw, start_x, x, baseline_y + 3, wobble)
                if is_struck:
                    _strike(mdraw, start_x, x, baseline_y - font_size * 0.28,
                            wobble)

                jitter_x = random.uniform(-2.0, 3.2) if wobble else 0
                x += int(space_w + jitter_x)
            
            current_y_slot += 1
            line_idx += 1
        
        img = composite_ink(img, mask, base_ink)
        if scan:
            img = apply_scan_look(img)

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
    ap.add_argument("--no-scan", action="store_true",
                    help="Skip the scanned-paper look (flat digital render)")
    args = ap.parse_args()
    
    content = Path(args.input).read_text(encoding="utf-8")
    result = render_assignment(
        content, args.output,
        style_key=args.style,
        ink_name=args.ink,
        wobble=not args.no_wobble,
        scan=not args.no_scan,
    )
    print(f"Done: {result['total_pages']} page(s) → {args.output}")
    print(f"Style: {result['style']}, Ink: {result['ink']}")


if __name__ == "__main__":
    main()
