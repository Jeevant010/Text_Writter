#!/usr/bin/env python3
"""Turn a scanned assignment PDF into one PNG the generator can use as style."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from textwritter.runtime import ROOT  # noqa: E402


def pdf_page_to_png(pdf: Path, page: int, out: Path, dpi: int = 200) -> Path:
    """Render 1-based `page` of a PDF to PNG. Tries pdftoppm, then PyMuPDF."""
    pdf = Path(pdf).expanduser().resolve()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    prefix = out.parent / f"{out.stem}_p{page}"
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), "-f", str(page), "-l", str(page),
             str(pdf), str(prefix)],
            check=True,
        )
        hits = sorted(prefix.parent.glob(f"{prefix.name}*.png"))
        if not hits:
            raise RuntimeError(f"pdftoppm ran but wrote no PNG next to {prefix}")
        shutil.copy2(hits[0], out)
        if hits[0].resolve() != out.resolve():
            hits[0].unlink(missing_ok=True)
        return out

    try:
        import fitz  # PyMuPDF
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf"],
                       check=False)
        import fitz

    doc = fitz.open(pdf)
    if page < 1 or page > len(doc):
        raise ValueError(f"{pdf} has {len(doc)} pages; asked for {page}")
    pix = doc[page - 1].get_pixmap(dpi=dpi)
    pix.save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export one PDF page as a handwriting style PNG.")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "samples" / "my_handwriting.png")
    args = ap.parse_args()
    path = pdf_page_to_png(args.pdf, args.page, args.out)
    print(f"wrote {path}")
    print("Use this as --style. Prefer a page with many English words, not only math.")


if __name__ == "__main__":
    main()
