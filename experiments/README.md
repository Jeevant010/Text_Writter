# Experiments — Model Working Options

Each folder is a **self-contained working option** for the handwriting engine. Each has:

- `README.md` — how the model works, verified accuracy (from the papers/repos), and
  **usage docs for non-coders**.
- `engine.py` — the reusable adapter (style prep + generation) used by the pipeline.
- A runnable `test_inference.py` CLI over that engine. Training lives in
  [`training/finetune.py`](../training/finetune.py), which runs on Colab **or** a GPU PC.

| # | Option | Best at | Hardware | Status |
|---|---|---|---|---|
| 01 | [One-DM](01_onedm/) | Best one-shot fidelity (1 sample → your style) | **NVIDIA GPU** (local PC or Colab) | GPU engine |
| 02 | [HWT](02_hwt/) | Fast few-shot, **runs anywhere** (CPU ok) | CPU or GPU | Default / dev |
| 03 | [TrOCR Judge](03_trocr_judge/) | Read-back quality gate for the re-roll loop | CPU or GPU | Always on |

## Which engine runs?

`python src/textwritter/quicktest.py` auto-picks: **One-DM if a CUDA GPU is
available, HWT otherwise**. Both stay available on both kinds of machine — force one
with `--engine onedm` / `--engine hwt` (or the `TW_ENGINE` env var). One-DM refuses
cleanly on CPU with instructions rather than crashing.

## Ground rules for all experiments

1. **Never train from scratch.** Load the authors' pretrained checkpoints.
2. **Try inference before any fine-tuning.** Both models are designed for one/few-shot
   conditioning — your photo is an input, not a training set.
3. Scripts auto-detect the device (`torch.cuda.is_available()`); no CUDA → HWT on CPU.
4. **No style photo is never a blocker** — bundled handwriting samples in
   [`assets/sample_styles/`](../assets/sample_styles/) are used instead.
