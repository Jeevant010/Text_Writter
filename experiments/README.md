# Experiments — Model Working Options

Each folder is a **self-contained working option** for the handwriting engine. Each has:

- `README.md` — how the model works, verified accuracy (from the papers/repos), and
  **usage docs for non-coders**.
- A runnable Python test script (testing lives in `.py` files; training lives in
  [`notebooks/`](../notebooks/) on Colab).

| # | Option | Best at | Hardware | Status |
|---|---|---|---|---|
| 01 | [One-DM](01_onedm/) | Best one-shot fidelity (1 sample → your style) | GPU (RTX 4060 / Colab) | **Try first** |
| 02 | [HWT](02_hwt/) | Fast few-shot, CPU-friendly, official Colab demo | CPU or GPU | Fallback / dev |
| 03 | [TrOCR Judge](03_trocr_judge/) | Read-back quality gate for the re-roll loop | CPU or GPU | Always on |

## Which one produces the final product?

One path only — **One-DM on GPU** (Option 01) is the production engine. HWT is the
CPU/dev fallback behind the same interface. TrOCR is not a generator; it judges output.

## Ground rules for all experiments

1. **Never train from scratch.** Load the authors' pretrained checkpoints.
2. **Try inference before any fine-tuning.** Both models are designed for one/few-shot
   conditioning — your photo is an input, not a training set. Fine-tune only if the
   pretrained result visibly fails on your style.
3. All scripts auto-detect GPU (`torch.cuda.is_available()`) and fall back to CPU.
