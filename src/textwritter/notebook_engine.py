#!/usr/bin/env python3
"""Realistic Notebook ('Copy with Lines') Handwriting Engine.

Renders natural, fluid handwriting directly onto ruled notebook paper (lined copy)
with blue horizontal lines, red vertical margin, header area, subtle paper grain,
and natural human writing variations (baseline wobble, word spacing, ink variations).

Lightning fast (<0.1s on CPU) compared to heavy neural patch generators.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

STYLES = {
    "kalam": {
        "name": "Kalam (Student Notebook)",
        "file": "Kalam.ttf",
        "default_size": 36,
        "line_height": 56,
        "y_offset_ratio": 0.82,
    },
    "caveat": {
        "name": "Caveat (Flowing Cursive)",
        "file": "Caveat.ttf",
        "default_size": 42,
        "line_height": 56,
        "y_offset_ratio": 0.78,
    },
    "indie": {
        "name": "Indie Flower (Casual Hand)",
        "file": "IndieFlower.ttf",
        "default_size": 34,
        "line_height": 56,
        "y_offset_ratio": 0.80,
    },
    "patrick": {
        "name": "Patrick Hand (Neat Notes)",
        "file": "PatrickHand.ttf",
        "default_size": 36,
        "line_height": 56,
        "y_offset_ratio": 0.82,
    },
}

INKS = {
    "blue": (24, 62, 132),       # Classic blue ballpoint / gel pen
    "black": (26, 28, 32),       # Black ink pen
    "darkblue": (15, 38, 85),    # Pilot dark blue
    "red": (185, 30, 40),        # Red grading / correction pen
}


PAGE_SIZE = (1200, 1600)
PAGE_TOP = 170
PAGE_LEFT = 190
PAGE_RIGHT = 1100
PAGE_LINE_H = 56


def make_blank_ruled_page(
    paper_type: str = "ruled",
    date_str: str = "___/___/20___",
    page_no: str = "________",
) -> Image.Image:
    """Empty student notebook page (ivory paper, red margin, blue ruling)."""
    width, height = PAGE_SIZE
    paper_color = (253, 251, 247)
    img = Image.new("RGB", (width, height), paper_color)
    grain = np.random.normal(0, 1.8, (height, width, 3)).astype(np.int16)
    base = np.array(img).astype(np.int16)
    img = Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    left_margin = PAGE_LEFT

    if paper_type == "ruled":
        header_font_path = FONTS_DIR / "PatrickHand.ttf"
        header_font = (
            ImageFont.truetype(str(header_font_path), 24)
            if header_font_path.exists()
            else ImageFont.load_default()
        )
        draw.text((width - 320, 60), f"Date: {date_str}", fill=(160, 160, 180), font=header_font)
        draw.text((width - 320, 95), f"Page No: {page_no}", fill=(160, 160, 180), font=header_font)
        draw.line([(left_margin, 0), (left_margin, height)], fill=(225, 110, 110), width=2)
        draw.line([(left_margin - 6, 0), (left_margin - 6, height)], fill=(238, 155, 155), width=1)
        y = PAGE_TOP
        while y < height - 80:
            draw.line([(0, y), (width, y)], fill=(185, 210, 235), width=1)
            y += PAGE_LINE_H
    elif paper_type == "grid":
        grid_size = 40
        for gx in range(0, width, grid_size):
            draw.line([(gx, 0), (gx, height)], fill=(215, 230, 245), width=1)
        for gy in range(0, height, grid_size):
            draw.line([(0, gy), (width, gy)], fill=(215, 230, 245), width=1)
    return img


def lines_per_page() -> int:
    height = PAGE_SIZE[1]
    return (height - PAGE_TOP - 100) // PAGE_LINE_H


def render_notebook_copy(
    text: str,
    out_path: str | Path,
    style_key: str = "kalam",
    ink_name: str = "blue",
    paper_type: str = "ruled",      # 'ruled' (copy with lines), 'blank', 'grid'
    date_str: str = "___/___/20___",
    page_no: str = "________",
    wobble: bool = True,
) -> dict:
    """Render text on ruled notebook paper ('copy with lines') with realistic ink and lines."""
    img = make_blank_ruled_page(paper_type=paper_type, date_str=date_str, page_no=page_no)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    top_margin = PAGE_TOP
    left_margin = PAGE_LEFT
    right_margin = PAGE_RIGHT
    line_height = PAGE_LINE_H

    # 3. Load Handwriting Font
    style_info = STYLES.get(style_key, STYLES["kalam"])
    font_file = FONTS_DIR / style_info["file"]
    font_size = style_info["default_size"]
    y_ratio = style_info["y_offset_ratio"]

    if font_file.exists():
        font = ImageFont.truetype(str(font_file), font_size)
    else:
        font = ImageFont.load_default()

    base_ink = INKS.get(ink_name, INKS["blue"])

    # 4. Word-Wrap & Natural Handwriting Layout
    paragraphs = text.split("\n")
    current_line_idx = 0
    max_lines = (height - top_margin - 100) // line_height

    for para in paragraphs:
        words = para.split(" ")
        current_x = left_margin + 20

        for word in words:
            if not word:
                continue

            bbox = font.getbbox(word)
            word_w = bbox[2] - bbox[0]
            word_h = bbox[3] - bbox[1]

            # Line wrapping
            if current_x + word_w > right_margin:
                current_line_idx += 1
                current_x = left_margin + 20
                if current_line_idx >= max_lines:
                    break

            # Sits on the notebook ruling line
            line_baseline = top_margin + (current_line_idx * line_height)
            
            # Subtle human wobble
            wobble_y = random.uniform(-1.8, 1.8) if wobble else 0.0
            draw_y = line_baseline - (font_size * y_ratio) + wobble_y

            # Natural ink pressure variation
            r, g, b = base_ink
            ink_var = random.randint(-12, 12) if wobble else 0
            actual_ink = (
                max(0, min(255, r + ink_var)),
                max(0, min(255, g + ink_var)),
                max(0, min(255, b + ink_var)),
            )

            # Draw the word
            draw.text((current_x, draw_y), word, fill=actual_ink, font=font)

            # Space width with micro-jitter
            space_bbox = font.getbbox(" ")
            space_w = space_bbox[2] - space_bbox[0]
            spacing_jitter = random.uniform(-1.0, 1.5) if wobble else 0.0
            current_x += word_w + space_w + spacing_jitter

        # Paragraph break moves to next line
        current_line_idx += 1
        if current_line_idx >= max_lines:
            break

    # Save output
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "out": out_path,
        "style": style_info["name"],
        "lines_written": current_line_idx,
        "ink": ink_name,
        "paper": paper_type,
    }
