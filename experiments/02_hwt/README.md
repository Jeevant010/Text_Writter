# Option 02 — HWT (Handwriting Transformers)

**Runs anywhere — CPU or GPU. The default engine when no NVIDIA GPU is present.**

## How it works (technical)

- **Paper:** *Handwriting Transformers*, **ICCV 2021** (Bhunia et al., MBZUAI).
- **Repo:** https://github.com/ankanbhunia/Handwriting-Transformers — ships with an
  **official Colab demo for custom handwriting** (`demo_custom_handwriting.ipynb`) and
  a Hugging Face live demo.
- **Pipeline:** a handful (≈5–15) of your cropped word images → transformer
  **encoder** captures global style (slant, ink width) + local style (letter shapes,
  ligatures) via self-attention → transformer **decoder** does style-content
  entanglement per query character → convolutional decoder renders the word image.
  A cycle-loss keeps style consistent.
- **Input needed from the user:** a few word images (we auto-crop them from one
  paragraph photo). No training.

## Verified accuracy (from the paper)

- **FID 19.40** on IAM image quality (vs ScrabbleGAN 20.72, Davis et al. 20.65 — lower
  is better; under ~25 looks natural to human eyes).
- Style-mimicry FID 106.97–114.10 across the four IAM protocols, beating GANwriting
  by up to **16.5 FID** on the hardest unseen-style + unseen-words setting.
- **Human study: preferred 81% of the time** over prior styled-generation methods.
- Generalizes to unseen words *and* unseen writing styles (OOV-U) — exactly our use case.

## Speed

| Hardware | Per word | Full page (~20 lines) |
|---|---|---|
| i5-1235U CPU | ~1–3 s | ~1–2.5 min |
| RTX 4060 / Colab T4 | ~0.1–0.3 s | ~10–20 s |

## Usage — for coders

```bash
# no photo needed — bundled handwriting samples are used automatically
python experiments/02_hwt/test_inference.py --text "The quick brown fox"

# with your own handwriting
python experiments/02_hwt/test_inference.py \
    --style samples/my_handwriting.jpg \
    --text "The quick brown fox" \
    --out out/hwt_test.png
```

First run clones the official repo and downloads the ~685 MB bundle (weights + IAM
samples); after that it's cached and generation takes seconds per word on CPU.

## Which engine runs?

`quicktest.py` prefers One-DM on a CUDA GPU and falls back to HWT otherwise; force it
with `--engine hwt` (or `TW_ENGINE=hwt`). HWT is the right choice on the i5 laptop, or
on a GPU machine when you want speed over maximum fidelity.

## Usage — for non-coders

Same as Option 01, except: write **one full paragraph** (3–4 lines) instead of one
line — HWT likes a few word samples, so a paragraph gives it more to learn your style
from. Everything else (photo tips, re-rolls) is identical.

## When to use which

| Situation | Use |
|---|---|
| You have the RTX 4060 / Colab GPU and want max fidelity | One-DM (01) |
| You're on the i5 laptop, or iterating fast on pipeline code | HWT (02) |
| Production server | One-DM, HWT kept as degradation fallback |
