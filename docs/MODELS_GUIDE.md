# Models Guide — Verified Comparison (fact-checked against papers/repos)

Numbers below come from the papers and official repos — not estimates.
Your earlier chat contained a few hallucinated figures; this table replaces them.

## Generators

| | One-DM ⭐ | HWT | GANwriting | WordStylist |
|---|---|---|---|---|
| Paper | **ECCV 2024** | **ICCV 2021** | CVPRW 2020 | LNCS 2023 |
| Repo | [dailenson/One-DM](https://github.com/dailenson/One-DM) (MIT) | [ankanbhunia/Handwriting-Transformers](https://github.com/ankanbhunia/Handwriting-Transformers) | [omni-us/research-GANwriting](https://github.com/omni-us/research-GANwriting) | HF weights available |
| Style input | **1 sample** (one-shot) | ~5–15 word crops | few samples | few samples |
| Architecture | Latent diffusion (SD1.5 VAE + UNet) + Laplacian style module | Transformer encoder-decoder | GAN | Latent diffusion |
| Verified result | Beats prior SOTA **using 15× fewer references**; English/Chinese/Japanese | FID **19.40** quality; style-FID beats GANwriting by **16.5** on hardest split; **preferred 81%** in human study | FID ~120 (per-writer protocol) — visibly blurrier | Competitive with One-DM era models |
| Speed (GPU) | ~2–6 s/word | ~0.1–0.3 s/word | <0.1 s/word | ~1–3 s/word |
| Speed (CPU) | impractical | **~1–3 s/word** ✅ | ~0.5 s/word | slow |
| Output granularity | words (v1) | words | words | words |
| Official quick demo | Drive checkpoint | **official custom-handwriting Colab** | — | HF space |

**Interpretation for our 70–80% perceptual target:** both One-DM and HWT sit in the
"looks natural at a glance" regime. One-DM wins on single-sample fidelity and edge
sharpness; HWT wins on speed and CPU tolerance. GANwriting is obsolete except as a
speed curiosity.

## The judge

| | TrOCR (`microsoft/trocr-base-handwritten`) |
|---|---|
| Role | Reads generated lines back → CER → auto re-roll trigger |
| Why not Gemini/LLM | Local, free, ~1 s, deterministic, character-precise |
| Cost to run | ~0.2 s/line GPU, ~1–2 s/line CPU |

## Datasets referenced

- **IAM** — English offline handwriting, the standard benchmark (research license,
  registration required). All models above are pretrained on it.
- **CROHME** — handwritten math expressions; reserved for Phase 4 math glyphs.
- **CVL / RIMES** — secondary HWT benchmarks (paper reports FID on both).

## Upgrade watch

- **DiffBrush** (same team as One-DM, ICCV 2025) — native **full-line** generation.
  When mature, it replaces our word-assembly compositor for lines. Not needed for v1.
