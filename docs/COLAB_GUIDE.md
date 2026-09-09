# Colab Guide — Training Without Losing Work

You don't need Colab for **inference** (pretrained checkpoints run anywhere).
Colab is for the two **optional fine-tune** notebooks and for trying GPU generation
free before your RTX 4060 is set up.

## Tiers (as of 2026 — verify, Google changes these)

| Tier | GPU | Session limit | Practical note |
|---|---|---|---|
| Free | T4 (16 GB) | ~12 h max, often 4–8 h, disconnects on idle | Fine for our 2–6 h fine-tunes |
| Pay-as-you-go / Pro | T4/A100 options | Longer, priority | Only if free tier keeps cutting you off |

**The rule that makes tiers irrelevant: checkpoint to Google Drive every epoch.**
Both notebooks in [`notebooks/`](../notebooks/) already do this — if a session dies,
reopen the notebook in any tier, Run-All, and it resumes from the last Drive
checkpoint. You can hop free tiers indefinitely this way.

## The checkpoint contract (already implemented in the notebooks)

```
Every epoch end:   local .pth  --copy-->  Drive/MyDrive/textwritter_checkpoints/<model>/
On notebook start: newest Drive .pth  --copy-->  local  ->  trainer resumes from it
```

Nothing important ever lives only on Colab's throwaway disk. **Do not** store the
only copy of anything in `/content` — it evaporates.

## Session migration runbook (when a tier ends)

1. Open the same notebook (it's saved in your Drive / this repo).
2. `Runtime → Change runtime type → GPU` (whatever tier you have now).
3. Run all cells top-to-bottom.
4. The resume cell prints `[resume] found <name>.pth on Drive -> resuming` and
   training continues from that epoch. Done.

## Fine-tune time budget (realistic, free T4)

| Job | Time | Worth it? |
|---|---|---|
| One-DM fine-tune (20 epochs, low LR) | ~2–6 h | Only if one-shot inference visibly fails on your style |
| HWT fine-tune (15 epochs) | ~2–4 h | Same — try official custom demo first |
| TrOCR judge | 0 h (pretrained) | Always |

## Two warnings

1. **Try pretrained inference BEFORE any fine-tuning.** One-DM is one-shot by design
   and HWT has an official custom-handwriting demo. There's a good chance you never
   open these notebooks for real — they're insurance.
2. **IAM dataset license** is research-only and requires registration. Fine-tuning on
   *your own* pages (what the notebooks do) avoids redistributing anything.
