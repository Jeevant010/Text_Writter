#!/usr/bin/env python3
"""Clean HWT word images and composite them onto notebook paper.

Takes the raw HWT grayscale output (words with yellowish/gray backgrounds),
removes the background via thresholding, and composites the clean ink strokes
onto ruled notebook paper with proper layout.

This is the hybrid approach:
  - HWT generates naturally varied handwriting (real stroke variation)
  - This script removes the ugly background patches
  - Composites clean strokes onto lined notebook paper

Usage:
    python clean_compositor.py out/quicktest_hwt.png out/clean_result.png
    python clean_compositor.py out/quicktest_hwt.png out/clean_result.pdf --ink blue
"""
from __future__ import annotations

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# Import the notebook page generator
FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

# ── Page constants ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 1200, 1700
MARGIN_LEFT = 155
MARGIN_RIGHT = 1120
MARGIN_TOP = 140
LINE_H = 48

INKS = {
    "blue": (24, 62, 132),
    "black": (26, 28, 32),
    "darkblue": (15, 38, 85),
}


def make_notebook_page(page_num: int = 1) -> Image.Image:
    """Create ruled notebook page background."""
    paper = (252, 250, 245)
    img = Image.new("RGB", (PAGE_W, PAGE_H), paper)

    # Paper grain
    grain = np.random.normal(0, 1.5, (PAGE_H, PAGE_W, 3)).astype(np.int16)
    base = np.array(img).astype(np.int16)
    img = Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Red margin
    draw.line([(MARGIN_LEFT, 0), (MARGIN_LEFT, PAGE_H)], fill=(220, 100, 100), width=2)
    draw.line([(MARGIN_LEFT - 5, 0), (MARGIN_LEFT - 5, PAGE_H)], fill=(235, 150, 150), width=1)

    # Blue ruled lines
    y = MARGIN_TOP
    while y < PAGE_H - 60:
        draw.line([(0, y), (PAGE_W, y)], fill=(180, 205, 230), width=1)
        y += LINE_H

    # Header
    hdr_font_path = FONTS_DIR / "PatrickHand.ttf"
    if hdr_font_path.exists():
        hdr = ImageFont.truetype(str(hdr_font_path), 20)
    else:
        hdr = ImageFont.load_default()
    c = (140, 140, 160)
    draw.text((40, 22), "Expt. No. ___________", fill=c, font=hdr)
    draw.text((PAGE_W - 300, 22), "Date ________________", fill=c, font=hdr)
    draw.text((PAGE_W - 200, 55), f"Page No. __{page_num}__", fill=c, font=hdr)
    draw.line([(0, 90), (PAGE_W, 90)], fill=(180, 205, 230), width=1)

    # Teacher's signature
    sig_y = PAGE_H - 55
    draw.line([(PAGE_W // 2, sig_y + 20), (PAGE_W - 40, sig_y + 20)], fill=c, width=1)
    draw.text((PAGE_W // 2 + 20, sig_y + 22), "Teacher's Signature", fill=c, font=hdr)

    return img


def clean_hwt_image(img_path: str | Path) -> Image.Image:
    """Load an HWT output image and return it with transparent background.
    
    HWT outputs grayscale word images where:
      - Ink strokes are dark (0-100 range)
      - Background is light gray/white (200-255 range)
    
    We threshold to isolate ink, then create an RGBA image where
    background is fully transparent and ink is opaque.
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Adaptive threshold: ink is dark, background is light
    # Use Otsu-like approach: find the threshold that separates ink from paper
    threshold = 200  # Pixels darker than this are ink

    # Create alpha channel: ink = opaque, background = transparent
    # Smooth transition near the threshold for anti-aliasing
    alpha = np.clip((threshold - arr) / 40.0, 0, 1)  # Gradual fade
    alpha = (alpha * 255).astype(np.uint8)

    # Create ink intensity (darker = more opaque)
    ink = np.clip(arr, 0, 255).astype(np.uint8)

    # Build RGBA: use the grayscale as the visual, alpha as mask
    rgba = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rgba_arr = np.array(rgba)
    rgba_arr[:, :, 0] = ink  # R = grayscale value
    rgba_arr[:, :, 1] = ink  # G
    rgba_arr[:, :, 2] = ink  # B
    rgba_arr[:, :, 3] = alpha  # A = our computed mask

    return Image.fromarray(rgba_arr)


def colorize_ink(rgba: Image.Image, ink_color: tuple) -> Image.Image:
    """Recolor grayscale ink strokes to a specific ink color (blue, black, etc.).
    
    Takes an RGBA image where RGB is grayscale ink, and maps:
      - Dark pixels (ink) → target ink color
      - Light pixels (paper remnants) → keep transparent
    """
    arr = np.array(rgba).astype(np.float32)
    
    # The grayscale value tells us ink intensity:
    # 0 = maximum ink, 255 = no ink (paper)
    gray = arr[:, :, 0]  # All RGB channels are same (grayscale)
    
    # Ink strength: 0=no ink, 1=full ink (inverted from pixel value)
    ink_strength = np.clip((200 - gray) / 180.0, 0, 1)
    
    # Apply ink color weighted by strength
    r, g, b = ink_color
    out = np.zeros_like(arr)
    out[:, :, 0] = r * ink_strength + 255 * (1 - ink_strength)  # Blend toward paper
    out[:, :, 1] = g * ink_strength + 255 * (1 - ink_strength)
    out[:, :, 2] = b * ink_strength + 255 * (1 - ink_strength)
    out[:, :, 3] = arr[:, :, 3]  # Keep original alpha

    # Where ink is strong, make color fully the ink color
    mask = ink_strength > 0.15
    out[:, :, 0][mask] = np.clip(r + (1 - ink_strength[mask]) * 30, 0, 255)
    out[:, :, 1][mask] = np.clip(g + (1 - ink_strength[mask]) * 30, 0, 255)
    out[:, :, 2][mask] = np.clip(b + (1 - ink_strength[mask]) * 30, 0, 255)

    return Image.fromarray(out.astype(np.uint8))


def crop_ink_bbox(rgba: Image.Image, padding: int = 3) -> Image.Image:
    """Crop to the bounding box of non-transparent content."""
    alpha = np.array(rgba)[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not np.any(rows) or not np.any(cols):
        return rgba
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    rmax = min(rgba.height - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(rgba.width - 1, cmax + padding)
    return rgba.crop((cmin, rmin, cmax + 1, rmax + 1))


def _split_1d(arr_1d: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Find contiguous regions of non-zero values separated by gaps >= min_gap."""
    regions = []
    n = len(arr_1d)
    i = 0
    
    while i < n:
        # Skip gap (zeros)
        while i < n and arr_1d[i] == 0:
            i += 1
        if i >= n:
            break
        # Start of a region
        start = i
        while i < n:
            if arr_1d[i] > 0:
                i += 1
            else:
                # Found a zero — check gap length
                gap_start = i
                while i < n and arr_1d[i] == 0:
                    i += 1
                gap_len = i - gap_start
                if gap_len >= min_gap:
                    # Real gap — end this region
                    regions.append((start, gap_start))
                    break
                # Small gap — continue the region
        else:
            # Reached end while in a region
            regions.append((start, n))
    
    return regions


def split_into_lines_and_words(rgba: Image.Image, word_gap: int = 6, line_gap: int = 3) -> list[list[Image.Image]]:
    """Split an RGBA image into lines, then each line into words.
    
    Returns a list of lines, each line is a list of word images.
    """
    alpha = np.array(rgba)[:, :, 3]
    
    # Step 1: Split into horizontal lines (row-wise)
    row_alpha = np.sum(alpha > 10, axis=1)
    line_regions = _split_1d(row_alpha, line_gap)
    
    all_lines = []
    for r_start, r_end in line_regions:
        line_strip = rgba.crop((0, r_start, rgba.width, r_end))
        line_alpha = np.array(line_strip)[:, :, 3]
        
        # Step 2: Split this line into words (column-wise)
        col_alpha = np.sum(line_alpha > 10, axis=0)
        word_regions = _split_1d(col_alpha, word_gap)
        
        words = []
        for c_start, c_end in word_regions:
            word_img = line_strip.crop((c_start, 0, c_end, line_strip.height))
            word_img = crop_ink_bbox(word_img)
            if word_img.width > 3 and word_img.height > 3:
                words.append(word_img)
        
        if words:
            all_lines.append(words)
    
    return all_lines


def composite_on_notebook(
    hwt_image_path: str | Path,
    out_path: str | Path,
    ink_name: str = "blue",
    target_line_height: int = 36,
) -> dict:
    """Full pipeline: clean HWT → colorize → composite onto notebook paper.
    
    Args:
        hwt_image_path: Path to raw HWT output (grayscale)
        out_path: Output path (.png or .pdf)
        ink_name: Ink color name (blue, black, darkblue)
        target_line_height: Target height for each word on the page
    """
    ink_color = INKS.get(ink_name, INKS["blue"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Clean the HWT image (remove background)
    clean_rgba = clean_hwt_image(hwt_image_path)
    
    # Step 2: Colorize to desired ink color
    colored = colorize_ink(clean_rgba, ink_color)
    
    # Step 3: Split into lines, then words
    lines = split_into_lines_and_words(colored, word_gap=8, line_gap=3)
    
    # Flatten for counting
    total_words = sum(len(line) for line in lines)
    if total_words == 0:
        raise RuntimeError("No word images found after cleaning")
    
    print(f"[clean] extracted {total_words} words across {len(lines)} line(s)")
    
    # Step 4: Normalize word heights — find median height across all words
    all_heights = []
    for line in lines:
        for w in line:
            all_heights.append(w.height)
    median_h = sorted(all_heights)[len(all_heights) // 2] if all_heights else 30
    
    # Step 5: Create notebook page and place words line by line
    page = make_notebook_page(page_num=1)
    
    line_idx = 0
    max_lines = (PAGE_H - MARGIN_TOP - 100) // LINE_H
    
    for src_line in lines:
        x = MARGIN_LEFT + 30
        
        for word_img in src_line:
            # Scale word to target height, normalizing against median
            scale = target_line_height / max(median_h, 1)
            new_w = max(1, int(word_img.width * scale))
            new_h = max(1, int(word_img.height * scale))
            word_resized = word_img.resize((new_w, new_h), Image.LANCZOS)
            
            # Line wrap if needed
            if x + new_w > MARGIN_RIGHT:
                line_idx += 1
                x = MARGIN_LEFT + 30
                if line_idx >= max_lines:
                    break
            
            # Position: align baseline to ruled line
            baseline_y = MARGIN_TOP + line_idx * LINE_H
            word_y = baseline_y - new_h + 6  # Sit just above the line
            
            # Natural wobble
            wobble_y = random.uniform(-1.5, 1.5)
            
            # Composite (alpha-aware)
            paste_y = max(0, int(word_y + wobble_y))
            paste_x = max(0, int(x))
            page.paste(word_resized, (paste_x, paste_y), word_resized)
            
            # Word spacing with jitter
            space = random.randint(10, 18)
            x += new_w + space
        
        line_idx += 1
        if line_idx >= max_lines:
            break
    
    # Save
    if str(out_path).endswith(".pdf"):
        page.save(out_path, "PDF", resolution=150)
    else:
        page.save(out_path)
    
    print(f"[clean] saved to {out_path}")
    return {
        "output": out_path,
        "words": total_words,
        "lines": line_idx,
        "ink": ink_name,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Clean HWT output and composite on notebook paper")
    ap.add_argument("input", help="Raw HWT output image (grayscale)")
    ap.add_argument("output", help="Output file path (.png or .pdf)")
    ap.add_argument("--ink", default="blue", choices=list(INKS.keys()))
    ap.add_argument("--line-height", type=int, default=38)
    args = ap.parse_args()
    
    result = composite_on_notebook(
        args.input, args.output,
        ink_name=args.ink,
        target_line_height=args.line_height,
    )
    print(f"Done: {result['words']} words, {result['lines']} lines")


if __name__ == "__main__":
    main()
