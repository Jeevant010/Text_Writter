# Agent notes — Text_Writter

Read this before changing handwriting, assignment pages, or engines.

## What this repo actually is

Two different products share one tree. Do not mix them up.

| Path | What it is | When to use it |
|---|---|---|
| `src/textwritter/assignment_engine.py` | CPU notebook pages: Caveat/Kalam + math markup + ink + scan look | Full assignments (CS743 A2, etc.). Fast. Looks like a cursive **font** on ruled paper, not a clone of the user's hand. |
| `src/textwritter/write_assignment.py` | HWT English on ruled paper; math-heavy lines fall back to a font | Only when a real style PNG exists and the user wants neural English. Slow. Still cannot clone formulas. |
| `src/textwritter/quicktest.py` | One line of HWT / One-DM from a style photo | Style-cloning experiments. Not for 20-page math assignments. |
| `src/textwritter/web_server.py` | Studio UI, defaults to font “Ruled Copy” | Do not treat this as style cloning. |

**User goal that drove the assignment engine:** a convincing *human-looking* notebook (blue pen, ruled copy), not pixel-perfect cloning of their hand. Friends correctly called out typed-looking output when letters were disconnected from the paper lines.

## Canonical assignment command

Always set `PYTHONPATH=src` (or run as a module). `fontTools` is optional.

```bash
cd /home/jeevant/Desktop/Text_Writter
PYTHONPATH=src python3 src/textwritter/assignment_engine.py \
  samples/assignment2_solutions.txt \
  output/a2/Assignment2_solutions.pdf \
  --style caveat --ink blue
```

Styles: `caveat` (cursive, default for A2), `kalam`, `indie`, `patrick`.
Ink: `blue`, `black`, `darkblue`, `red`.

Content lives in a plain text file. Latest A2 solutions: `samples/assignment2_solutions.txt`.
Output: `output/a2/Assignment2_solutions.pdf` plus `output/a2/a2_pageNN.png` if the output suffix is `.png`.

## Markup (assignment_engine)

- `# heading` — underlined header
- `^{...}` superscript, `_{...}` subscript (nested braces OK: `w^{(2)}`)
- `\\to \\in \\times \\partial \\Sigma \\sigma \\nabla \\eta \\ell` → Unicode
- leading spaces = indent
- `---` separator
- `__word__` underline, `~~word~~` strike

Do **not** put nested markup in a way that needs three visual sizes; inner `^`/`_` is flattened.

## Hard-won rendering rules (do not regress)

1. **Draw whole words**, not per-letter tiles. Per-character rotate/scale **breaks Caveat ligatures** and reads as typing. Variation is word-level (`_draw_connected`).
2. **Baseline = ruled line.** Use `LINE_SPACING` for both paper rules and text slots. Place glyphs with `font.getmetrics()` ascent, not `MARGIN_TOP + 10` plus large `wobble_y`.
3. **Glyph coverage:** Caveat has no math. If `fontTools` is missing, `_covers` must **not** assume “unknown = covered” (that produced `□` boxes). Compare a raster against U+FFFF / blank.
4. **Kalam `1` looks like `l`.** Prefer Caveat for math-heavy assignments. Caveat capital `L` looks like `(` — `CONFUSABLE` swaps that letter to the fallback face.
5. Blur/scan is a light finish. It cannot hide disconnected glyphs.
6. This is still a font. Next fidelity jump is the **glyph bank** (`glyphs.py`) for `→ ∂ Σ ∇ σ …`, or HWT for English-only lines.

## Glyph bank (user writes the 19+ symbols)

```bash
PYTHONPATH=src python3 src/textwritter/glyphs.py sheet
PYTHONPATH=src python3 src/textwritter/glyphs.py slice --image samples/glyph_sheet_filled.jpg
PYTHONPATH=src python3 src/textwritter/glyphs.py check
```

Highest-frequency fallbacks on A2: `→` (hundreds), `∂`, then `Σ ✓ ∇ δ σ ℓ`.

## Style cloning (separate track)

Needs a **photo of real handwriting**, not a screenshot of the web UI.

```bash
python src/textwritter/extract_style.py --pdf "/path/to/scan.pdf" --page 1
python src/textwritter/quicktest.py --style samples/my_handwriting.png --engine hwt
```

Prefer a page with English words. Math-only pages are a bad style source.

## Layout of code that matters

```
src/textwritter/assignment_engine.py   # ruled pages + math (CPU)
src/textwritter/glyphs.py              # print/slice user symbol sheet
src/textwritter/notebook_engine.py    # older ruled-page helper (HWT compositor)
src/textwritter/write_assignment.py     # HWT multi-page
src/textwritter/extract_style.py      # PDF page → style PNG
samples/assignment2_solutions.txt     # CS743 A2 worked answers
docs/ASSIGNMENT_PAGES.md              # human how-to
```

## Rules that keep this project fast

1. Never train from scratch.
2. Do not promise biometric cloning of the user's hand from a font pipeline.
3. One-DM is GPU-only and too slow for 27-page assignments; HWT is the neural English engine; assignment_engine is the assignment workhorse.
