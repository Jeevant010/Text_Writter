# Option 03 — TrOCR Judge (Quality Gate)

**Not a generator — the automated "is this readable?" judge for the re-roll loop.**

## How it works (technical)

- **Model:** `microsoft/trocr-base-handwritten` (or `-large-`) on Hugging Face —
  Microsoft's TrOCR, a ViT encoder + text transformer decoder trained for handwritten
  text recognition (HTR).
- **Role in the system:** after One-DM/HWT generates a line, TrOCR *reads it back*.
  We compute **CER** (Character Error Rate) between the target text and what TrOCR
  actually read. If CER > threshold → the line is illegible → auto re-roll with a
  new seed. This is the loop's objective quality gate — zero human effort.
- **Why not an LLM (Gemini) as judge?** TrOCR is free, local, ~1 s, deterministic, and
  purpose-built for handwriting OCR. An LLM adds API latency + cost for worse
  character-level precision. Gemini's job is content and tweak-parsing, not judging.

## Verified accuracy

- TrOCR-large is a standard benchmark topper on the IAM handwriting dataset
  (CER in the low single digits on clean IAM lines). For judging, `base` is enough;
  use `large` if borderline calls look noisy.

## Speed

~0.2 s/line on GPU, ~1–2 s/line on CPU.

## Usage — for coders

```bash
python experiments/03_trocr_judge/judge.py \
    --image out/onedm_test.png \
    --expect "Hello world"
# prints: recognized text, CER, PASS/FAIL vs threshold
```

## Usage — for non-coders

You never touch this directly. It runs silently after every generation: if the system
"reads back" your generated line and finds too many mistakes, it automatically redraws
that line before you ever see it. It's the reason bad output rarely reaches your eyes.
