# Option 01 — One-DM (One-Shot Diffusion Mimicker)

**The primary engine. Best quality per sample. This is the one to try first.**

## How it works (technical)

- **Paper:** *One-DM: One-Shot Diffusion Mimicker for Handwritten Text Generation*,
  **ECCV 2024** (Dai, Zhang, Ke, Guo, Huang). arXiv: 2409.04004
- **Repo:** https://github.com/dailenson/One-DM (MIT license, official pretrained
  checkpoint on Google Drive)
- **Pipeline:** your **single** reference writing image → a *style-enhanced module*
  (Laplacian high-frequency filter + gated CNN-transformer style encoders + contrastive
  learning) extracts your slant, ligatures and stroke patterns while suppressing paper
  background noise → fused with a content encoding of the target text → a conditional
  **latent diffusion** model (Stable Diffusion 1.5 VAE + custom UNet, DDIM sampling,
  default 50 steps) renders the text in your style.
- **Input needed from the user:** **one** reference sample image. Zero training,
  zero fine-tuning — the sample is inference-time conditioning only.

## Verified accuracy (from the paper/repo — not estimates)

- Outperforms previous state-of-the-art methods **that used 15× more reference samples**,
  despite using only one.
- Demonstrated on English (IAM), Chinese, and Japanese handwriting benchmarks.
- Diffusion + high-frequency style extraction = sharpest stroke fidelity of the options
  here; this is the closest to "a human can't tell" at a glance.
- Caveat: the open-source v1 generates **word-level** images (64 px height). Full lines
  are assembled word-by-word by our compositor. (The same team's **DiffBrush**, ICCV
  2025, does native full-line generation — a future upgrade path.)

## Speed

| Hardware | Per word/short line | Full page (~20 lines) |
|---|---|---|
| RTX 4060 8 GB | ~2–6 s | ~2–4 min |
| Colab free T4 | ~4–10 s | ~3–6 min |
| i5-1235U CPU | impractical (~min/word) | use HWT instead |

## Usage — for coders

```bash
python experiments/01_onedm/test_inference.py \
    --style samples/my_handwriting.jpg \
    --text "Hello world" \
    --out out/onedm_test.png
```

The script clones the official repo, downloads the pretrained checkpoint, preprocesses
your photo (deskew + binarize + crop), and runs inference. First run downloads ~5 GB
(checkpoint + SD 1.5 VAE).

## Usage — for non-coders

1. Write **one line** of text on plain white paper (no lines/grids). Black or blue pen.
2. Photograph it in good light, flat, no shadows.
3. Give that photo + the text you want written to the person running the system
   (or use the notebook). You get back a PNG of the text **in your handwriting**.
4. If a word looks off, ask for a "re-roll" — the system regenerates just that word
   with different randomness. Repeat until happy.
5. Tips for best results: write naturally (don't try to be neat), include mixed
   upper/lowercase, avoid printed block letters if you normally write joined-up.

## Known limitations

- Word-level output (line assembly is our compositor's job).
- English only in the open-source checkpoint.
- Needs a GPU for usable speed.
