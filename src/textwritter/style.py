"""Shared style preparation for both engines (HWT and One-DM).

Turns "a photo of handwriting, or nothing at all" into a folder of word-crop PNGs:
  * photo given        -> deskew, binarize, segment lines then words (OpenCV)
  * no photo           -> bundled real-handwriting samples (committed assets),
                          with samples/sample_hello.png as a last-resort print-font
                          demo so a run still produces output.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from .runtime import ROOT, SAMPLE_STYLES_DIR

_FALLBACK_SAMPLE = ROOT / "samples" / "sample_hello.png"


def segment_words(gray) -> list:
    """Split a handwriting photo into word crops (OpenCV only, no ML)."""
    import cv2
    import numpy as np

    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords):
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.3:
            h, w = bw.shape
            m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            bw = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)

    def bands(profile, min_gap: int):
        out, start, gap = [], None, 0
        for i, v in enumerate(profile):
            if v > 0:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= min_gap:
                    out.append((start, i - gap + 1))
                    start = None
        if start is not None:
            out.append((start, len(profile)))
        return out

    rows = bw.sum(axis=1) // 255
    words = []
    for y0, y1 in bands(rows, max(4, bw.shape[0] // 60)):
        strip = bw[y0:y1]
        cols = strip.sum(axis=0) // 255
        for x0, x1 in bands(cols, max(6, (y1 - y0) // 3)):
            if (x1 - x0) >= 2 and (y1 - y0) >= 4:
                words.append(strip[:, x0:x1])
    return words


def save_words(words, out_dir: Path, prefix: str = "w") -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, w in enumerate(words):
        p = out_dir / f"{prefix}{i:02d}.png"
        cv2.imwrite(str(p), w)
        paths.append(p)
    return paths


def bundled_samples(extract: Callable[[], list[Path]] | None = None,
                    verbose: bool = True) -> list[Path]:
    """Committed sample styles, else extract once, else the printed sample image."""
    pngs = sorted(SAMPLE_STYLES_DIR.glob("*.png")) if SAMPLE_STYLES_DIR.exists() else []
    if pngs:
        return pngs
    if extract is not None:
        try:
            got = extract()
            if got:
                return got
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[style] bundled-style extraction failed ({e})")
    return [p for p in [_FALLBACK_SAMPLE] if p.exists()]


def prepare_style(style: str | Path | None, work_dir: Path,
                  extract: Callable[[], list[Path]] | None = None,
                  verbose: bool = True) -> tuple[Path, str, list[str]]:
    """Build the style word folder. Returns (dir, source, warnings)."""
    import cv2

    out_dir = Path(work_dir) / "style_words"
    warnings: list[str] = []

    if style:
        style = Path(style)
        if style.exists():
            gray = cv2.imread(str(style), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                words = segment_words(gray)
                if len(words) >= 3:
                    save_words(words, out_dir)
                    if verbose:
                        print(f"[style] {len(words)} word crops from {style.name}")
                    return out_dir, f"photo:{style.name}", warnings
                warnings.append(
                    f"only {len(words)} word(s) found in {style.name}; using "
                    "bundled style instead (a full line works far better)")
            else:
                warnings.append(f"could not read image {style}; using bundled style")
        else:
            warnings.append(f"style file not found ({style}); using bundled style")

    samples = bundled_samples(extract=extract, verbose=verbose)
    if not samples:
        raise RuntimeError(
            "No style photo given and no bundled style available. Pass one: "
            "--style /path/to/your_handwriting.(jpg|png)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(samples[:15]):
        shutil.copy2(p, out_dir / f"{i:02d}_{p.name}")
    if verbose:
        print(f"[style] using {min(len(samples), 15)} bundled handwriting samples")
    if samples[0].parent == _FALLBACK_SAMPLE.parent:
        warnings.append("bundled style fell back to a printed sample — output "
                        "will look typed; pass --style with real handwriting")
    return out_dir, "bundled", warnings
