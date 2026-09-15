# Colab & GPU-PC Guide — Run In Either Place, Lose Nothing

You don't need Colab for **inference** — pretrained checkpoints run anywhere, and the
CPU path (HWT) runs on a plain laptop. Colab or a GPU PC is for **speed and the
optional fine-tunes**. Everything below uses the **same commands in both places**;
only the wrapper differs (notebook cell on Colab, terminal on a PC).

## What to run where

| Machine | Creation | Engine picked | Fine-tune |
|---|---|---|---|
| Laptop, no NVIDIA GPU | `python src/textwritter/quicktest.py` | HWT (CPU) | refused — says to use a GPU PC or Colab |
| GPU PC (friend's, RTX) | `python src/textwritter/quicktest.py` | One-DM (GPU) | `python training/finetune.py --model hwt --data …` |
| Google Colab | `notebooks/00_quick_test.ipynb` → Run all | One-DM (GPU) | `notebooks/02_hwt_finetune.ipynb` → Run all |

Force an engine any time with `--engine hwt|onedm` (or `TW_ENGINE=hwt`). Auto means
"One-DM if a CUDA GPU is present, else HWT".

## Colab tiers (verify — Google changes these)

| Tier | GPU | Session limit | Practical note |
|---|---|---|---|
| Free | T4 (16 GB) | ~12 h max, often 4–8 h, disconnects on idle | Fine for our 2–6 h fine-tunes |
| Pay-as-you-go / Pro | T4/A100 options | Longer, priority | Only if free tier keeps cutting you off |

**The rule that makes tiers irrelevant: checkpoint every epoch off the machine.**
`training/finetune.py` does this for you — on Colab it copies each epoch's checkpoint to
Google Drive, on a PC to `checkpoints/<run>/`. If a Colab session dies, reopen the
notebook and Run-All: the resume logic finds the newest checkpoint and continues from
that epoch. You can hop free tiers indefinitely this way.

```
Windows/PC:  checkpoints/<run>/*.pth
Colab:       /content/drive/MyDrive/textwritter_checkpoints/<run>/*.pth
Every epoch: newest checkpoint copied off the machine automatically
```

Nothing important ever lives only on Colab's throwaway disk. **Do not** keep the only
copy of anything in `/content` — it evaporates.

## Runbook when a Colab tier ends

1. Open the same notebook (saved in your Drive / this repo).
2. `Runtime → Change runtime type → GPU` (whatever tier you have now).
3. Run all cells top-to-bottom.
4. The run prints `[resume] … -> resuming` and continues from the last checkpoint.

## First-run downloads (one time, then cached)

| What | Size | Where it goes |
|---|---|---|
| HWT bundle (`files.zip`) | ~685 MB | `experiments/02_hwt/_vendor/HWT/files/` |
| TrOCR judge | ~300 MB | Hugging Face cache |
| One-DM checkpoint + English data | multi-GB | `checkpoints/` (folder bundles) |

## Fine-tune time budget (free T4)

| Job | Time | Worth it? |
|---|---|---|
| HWT fine-tune (15 epochs) | ~2–4 h | Only if pretrained inference visibly misses your style |
| One-DM fine-tune (20 epochs, low LR) | ~2–6 h | Same — and it needs the One-DM data bundle + OCR weights |
| TrOCR judge | 0 h (pretrained) | Always |

## Two warnings

1. **Try pretrained inference BEFORE any fine-tuning.** One-DM is one-shot by design
   and HWT has an official custom-handwriting demo. There's a good chance you never
   need these flows — they're insurance.
2. **IAM dataset license** is research-only and requires registration. Fine-tuning on
   *your own* pages (what these flows do) avoids redistributing anything.
