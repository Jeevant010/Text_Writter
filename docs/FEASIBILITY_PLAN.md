# Feasibility Study & Implementation Plan
## AI Handwriting Agent — Terminal-First Prototype

> **Scope of this prototype:** backend + CLI only. No web frontend. Output is PNG/PDF
> files on disk, viewed from the terminal workflow. The architecture keeps clean API
> boundaries so it can later be dropped behind the production website unchanged.

---

## 1. Executive Summary

| Question | Verdict |
|---|---|
| Can we synthesize a specific person's handwriting from a small sample? | **Yes** — few-shot style-conditioned generation is a solved research problem with open-source models. |
| Can we do it **without manual labor** (no per-user training, no tracing forms)? | **Yes** — one photo of one paragraph is enough for a style-conditioned model. Zero backpropagation per user. |
| Can we render arbitrary content (essays, notes, letters)? | **Yes** — line-by-line generation + page compositor. |
| Can we render math (equations, fractions, roots)? | **Yes, with caveats** — LaTeX engine computes layout only; handwritten glyphs are substituted into the layout boxes. Fidelity lower than prose for complex expressions. |
| Target 70–80% perceptual fidelity? | **Realistic** for prose. Math fidelity ~60–75% in v1. |
| Can it run on the local machine (i5-1235U, 14 GB RAM, no dGPU)? | **Yes** — HWT backend is CPU-feasible (~1–2.5 min/page). |
| Can it exploit the RTX 4060 (8 GB) machine? | **Yes** — one codebase, dual backends (HWT + One-DM), auto device detection. No cloud GPU needed. |

**Recommended core strategy:** *Style-conditioned Handwritten Text Generation (HWT) model*
as the engine, a *LaTeX layout engine* for math geometry, an *HTR-based self-check loop*
for quality, orchestrated as a thin pipeline (LangGraph optional) behind a Typer CLI.

---

## 2. THE Core Problem: Acquiring Real Handwriting Without Manual Labor

This is the crux of the whole product. Everything else (layout, pagination, export) is
ordinary engineering. There are exactly four known ways to capture a person's writing,
ranked by user effort:

### Option A — Manual glyph sheet (user fills a form)
User writes `a–z, A–Z, 0–9`, punctuation into labeled boxes; we slice and replay glyphs.

- **User labor:** HIGH (30–60 min of careful writing).
- **Fidelity:** High for letters, but output looks like a "ransom note" — repeated glyphs
  are identical, cursive joins are lost, ligatures break. Needs heavy jitter/ligature
  post-processing to not look fake.
- **Verdict:** Keep only as an *optional calibration fallback* for users who want max
  control. Never the default onboarding.

### Option B — Free-form pages + automatic glyph harvesting (OpenCV segmentation)
User uploads 2–4 pages; OpenCV binarizes, segments connected components, an OCR/CTC
classifier tags each glyph into a bank of variants.

- **User labor:** LOW-MEDIUM (they must write 2–4 pages).
- **Fragility:** Cursive/connected writing breaks naive connected-component segmentation;
  mislabeled glyphs poison the bank. Real engineering cost is high for unreliable gain.
- **Verdict:** Useful only as a *fidelity booster* layered on top of Option C (swap real
  harvested glyphs into generated lines where confident). Not the primary mechanism.

### Option C — Few-shot style-conditioned generation (RECOMMENDED CORE)
A pretrained model takes **(a) a style sample image** — one photo of one paragraph — and
**(b) a target text string**, and directly generates an image of that text in that style.
The model was trained on thousands of writers (IAM dataset), so it has already learned
"how handwriting varies across humans." The user's sample is only *conditioning* at
inference time — no training, no forms, no segmentation by us.

Candidate open-source models:

| Model | Style input needed | Type | CPU-feasible? | Notes |
|---|---|---|---|---|
| **HWT — Handwriting Transformers** (CVPR'21) | A few word/line images | Transformer encoder-decoder | **Yes** (seconds/line) | Strong style/content disentanglement; best CPU-first choice. |
| **One-DM** (AAAI'24) | **One** sample | Diffusion | Slow on CPU (min/line) | Highest fidelity from a single sample; use on GPU later. |
| **DiffusionPen** | A few samples | Diffusion | Slow on CPU | Good style control knobs. |
| **GANwriting** (CVPR'20) | A few samples | GAN | Yes (fast) | Older, blurrier, but very fast. |

- **User labor:** **near ZERO** — snap a photo of any handwritten paragraph.
- **Fidelity:** 70–85% perceptual match in the literature; matches our target exactly.
- **Verdict:** **This is the primary acquisition mechanism.** It is the direct,
  honest answer to "get the real person's handwriting without manual labour."

### Option D — Per-user fine-tuning (LoRA/DreamBooth on a diffusion backbone)
- **User labor:** zero, but **compute labor:** 10–60 min GPU training per user, storage
  per user, ops complexity.
- **Verdict:** A later premium tier ("ultra mode") behind cloud GPU. Not v1.

### Realistic speed & training expectations

**Inference per generated line (~10–12 words):**

| Backend | i5-1235U (CPU only) | RTX 4060 8 GB |
|---|---|---|
| HWT | ~2–6 s | ~0.1–0.3 s |
| One-DM (25–50 denoise steps) | ~3–8 min (impractical) | ~3–8 s |
| One-DM + few-step distillation (LCM/Turbo) | ~40–90 s (usable, quality dip) | ~0.5–1.5 s |
| TrOCR judge (read-back check) | ~1–2 s | ~0.2 s |

**Full page (~20 lines) end-to-end (generate + judge + compositor):**

- i5 laptop, HWT: **~1–2.5 min/page**
- RTX 4060, HWT: **~10–20 s/page**
- RTX 4060, One-DM: **~1–3 min/page** (best fidelity tier)

**Training time — important: v1 needs ZERO training.** We load pretrained weights and
the user's photo is only inference-time conditioning. Numbers below apply only if we
ever train/fine-tune our own model (e.g. custom language or premium tier):

| Task | On RTX 4060 8 GB |
|---|---|
| HWT from scratch (IAM dataset) | ~2–4 days |
| HWT fine-tune (domain adaptation) | ~2–6 h |
| One-DM from scratch | ~5–10 days (paper used datacenter GPUs) |
| Per-user LoRA (optional Tier-3 premium) | ~20–45 min per user |
| TrOCR judge | pretrained, no training |

### Decision: Layered acquisition ladder

```
Tier 1 (default, zero labor):  Option C — HWT style encoder from 1 photo of 1 paragraph
Tier 2 (optional booster):     Option B — harvested glyph bank swapped in for top-20 letters
Tier 3 (optional premium):     Option D — per-user LoRA (RTX 4060 / cloud GPU)
Fallback (accessibility):      Option A — guided glyph sheet, one-time
```

A new user is productive in **under 2 minutes**: upload one photo → we extract line crops
→ encode style vector → cache it → generate.

---

## 3. System Overview (Terminal-First)

```
                        ┌────────────────────────────────────────────┐
                        │                 Typer CLI                   │
                        │  tw enroll | tw write | tw reroll | tw show │
                        └───────────────────┬────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        ▼                                   ▼                               ▼
 [ENROLL PIPELINE]                 [WRITE PIPELINE]                  [QUALITY LOOP]
 photo → denoise/deskew            text ─┐                          generated line
  (OpenCV) → line crops                    │                          → HTR model reads
  → StyleEncoder → z_style                 ▼                          it back → CER vs
  → cache (json/npz)              [Content Router]                    target → pass /
                                   verbatim │ ai-prompt               auto re-roll
                                            │ (LLM optional)          (seed ± jitter)
                                            ▼                                ▲
                              blocks = [para | heading | math]               │
                                            │                                │
                        ┌───────────────────┴───────────────────┐            │
                        ▼                                       ▼            │
              [Prose Renderer]                        [Math Renderer]        │
              HWT(z_style, line_text)                 LaTeX→glyph boxes      │
              → transparent line PNG                  → handwrite each glyph │
                        │                             (HWT/glyph bank)      │
                        │                             → composite w/ jitter │
                        └───────────────────┬───────────────────┘            │
                                            ▼                                │
                                  [Page Compositor] ─────────────────────────┘
                                  paper texture, ruled lines, ink color,
                                  baseline drift, margin wander
                                            │
                                            ▼
                                  out/<doc>/page-01.png … + manifest.json
```

**Everything is a file.** The terminal shows progress; results open with `xdg-open` or
the `tw show` helper. `manifest.json` records every block's seed/params so any line can
be deterministically re-rolled later — including from the future website.

---

## 4. Content Modes

### 4.1 Prose (essays, assignments, letters, notes)
- **Verbatim mode:** user's exact text → word-wrap to page width → per-line generation
  with HWT conditioned on `z_style`.
- **AI-writer mode:** prompt → local/API LLM produces text → same pipeline. (Pluggable;
  prototype can stub with a local model or external API key.)
- Humanization at the compositor level: Gaussian word-spacing, baseline random-walk,
  left-margin wander, ±1–2° line rotation, ascender/descender collision nudging.

### 4.2 Math
- LLM/user supplies **LaTeX**. A layout engine (`matplotlib.mathtext` for a pure-Python
  start; `dvisvgm`/KaTeX as the robust upgrade) emits **glyph bounding boxes only** —
  positions/scales, never fonts.
- Each box is filled with a *handwritten* rendering of that symbol:
  - Latin/digits/common operators → HWT model conditioned on `z_style`.
  - Rare symbols (∫, ∑, √, ∂, Greek) → CROHME-derived handwritten glyph library,
    style-adapted (slant θ, stroke width w, jitter σ from the user profile).
- Humanization: Bézier tremolo on fraction bars/radical roofs, ±5–10% bracket asymmetry,
  non-rigid super/subscript offsets.

---

## 5. The "Iterate Until Satisfied" Loop (works in a terminal)

The loop you described doesn't need a website — it needs a judge and a re-roll command.

1. **Automatic judge:** a pretrained handwritten-text-recognition model
   (e.g. TrOCR-handwritten) reads each generated line. If character-error-rate vs the
   target text > threshold → auto re-roll with a new seed (max N tries). This catches
   illegible output with zero human effort.
2. **Manual re-roll (CLI):**
   `tw reroll out/hw1 --page 1 --line 4 --seed 42 --jitter 0.6 --slant -3`
3. **Global knobs:** `tw write … --messy 0.7 --ink blue_ballpoint --paper ruled`
4. All state lives in `manifest.json` → the same re-roll API will be called by the
   production website later (one HTTP wrapper away).

*LangGraph note:* a genuine cyclic graph exists here (generate → verify → re-roll →
approve). A thin LangGraph flow is reasonable and keeps your preferred stack; but the
pipeline is deliberately implemented so the graph is a replaceable orchestration shell
around pure functions — not a load-bearing dependency.

---

## 6. Tech Stack (grounded)

| Layer | Choice | Why |
|---|---|---|
| Language / CLI | Python 3.11+, **Typer** + Rich | Clean commands, terminal progress bars |
| Style encoder + generator | **HWT** (CPU default) + **One-DM** (RTX 4060) | dual backends, one `HandwritingBackend` interface, auto device selection |
| Read-back judge | **TrOCR-handwritten** (HF) | Automated CER quality gate |
| Math layout | `matplotlib.mathtext` → `dvisvgm`/KaTeX upgrade | boxes only, pure-Python start |
| CV preprocessing | OpenCV, scikit-image | deskew, binarize, line crops |
| Compositing | Pillow / numpy; textures bundled | paper, ink, drift |
| Orchestration | plain pipeline now; **LangGraph** shell for the verify loop | matches your stack preference without over-coupling |
| LLM (AI-writer mode) | pluggable provider (API key) | verbatim mode needs none |
| Storage | filesystem: `profiles/`, `out/`, JSON manifests | zero-infra prototype |
| Later | FastAPI wrapper → Celery/Redis GPU workers → S3 | production website |

**Hardware reality check:** the prototype is **dual-compatible** — one codebase, two
backends, device auto-detection (`torch.cuda.is_available()` → One-DM eligible, else
HWT on CPU). The i5 laptop is the dev machine (HWT, ~1–2.5 min/page); the RTX 4060 box
runs everything fast (HWT ~10–20 s/page, One-DM ~1–3 min/page). Both backends generate
small line images, so 8 GB VRAM is comfortable (fp16 + attention-slicing if needed).
All models together fit in ~6 GB RAM. No cloud GPU required at any phase of v1.

---

## 7. Proposed Repository Layout

```
Text_Writter/
├── docs/FEASIBILITY_PLAN.md        ← this file
├── pyproject.toml
├── src/textwritter/
│   ├── cli.py                      # tw enroll / write / reroll / show
│   ├── enroll/
│   │   ├── preprocess.py           # deskew, binarize, line segmentation
│   │   └── style_encoder.py        # → z_style cache
│   ├── content/
│   │   ├── router.py               # verbatim | ai-prompt → block list
│   │   └── llm_writer.py           # optional provider
│   ├── render/
│   │   ├── backends/
│   │   │   ├── base.py             # HandwritingBackend protocol
│   │   │   ├── hwt_backend.py      # default (CPU)
│   │   │   └── onedm_backend.py    # later (GPU)
│   │   ├── prose.py                # wrap, lines, humanization
│   │   ├── math_layout.py          # LaTeX → glyph boxes
│   │   ├── math_glyphs.py          # handwritten symbol substitution
│   │   └── compositor.py           # paper, ink, pagination, manifest
│   ├── quality/
│   │   ├── htr_judge.py            # TrOCR read-back, CER
│   │   └── reroll.py               # seed/param control
│   └── workflow/graph.py           # thin LangGraph verify loop (optional)
├── assets/{papers,inks,math_glyphs}
├── profiles/                       # z_style + glyph banks per user
└── out/                            # generated documents + manifests
```

---

## 8. Phased Plan

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Scaffold** | `pyproject`, Typer CLI skeleton, config, folders | `tw --help` runs |
| **1. Enrollment** | photo → deskew/binarize/line-crops; z_style cache (HWT encoder) | `tw enroll samples/me.jpg --as me` stores profile |
| **2. Prose engine** | HWT backend + wrapping + compositor + paper/ink | `tw write "essay…" --profile me` → readable page PNGs |
| **3. Quality loop** | TrOCR judge, auto re-roll, manifest, `tw reroll` | illegible line self-corrects; manual reroll works |
| **4. Math v1** | mathtext layout → handwritten glyph substitution | `x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}` renders acceptably |
| **5. AI-writer mode** | pluggable LLM content → pipeline | `tw write --prompt "essay on photosynthesis"` |
| **6. API seam** | FastAPI thin wrapper over the same services | website can call enroll/write/reroll |
| **7. One-DM / GPU tier** | One-DM backend on the RTX 4060 (+ optional per-user LoRA) | photoreal tier behind a flag |

**Explicit non-goals for the prototype:** diagrams/drawings, multi-column layouts,
non-Latin scripts, real-time generation, any web UI.

---

## 9. Honest Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Model weights (HWT/One-DM) availability/licensing | blocker | Verify licenses before Phase 1; GANwriting as fallback; all behind `HandwritingBackend` protocol so swapping is trivial |
| IAM-trained models = English only | scope limit | document limitation; collect CROHME-style data later |
| Math symbols unseen in user style | uncanny math | CROHME glyph library + style transfer (slant/width/jitter) |
| CPU latency (1–2 min/page) | UX | Rich progress, parallel line workers, caching; GPU later |
| "Ransom-note" look if glyph-bank path dominates | quality gate | glyph bank is *booster only*; HWT generation is default |
| Handwriting misuse (forgery) | ethical/legal | watermark metadata, consent notice in CLI, ToS in product |

---

## 10. Decisions (resolved)

1. **Backend:** **Dual-compatible** — HWT (CPU default, i5 laptop) + One-DM (RTX 4060)
   behind the `HandwritingBackend` protocol, auto-selected by device detection. Same
   script runs on both machines unchanged.
2. **Onboarding:** **Single photo by default + optional calibration glyph sheet** for
   users who want maximum fidelity.
3. **Math v1:** **Restricted set** — arithmetic, algebra, fractions, radicals,
   sub/superscripts, basic calculus (∫, Σ, ∂, π, common Greek). Full LaTeX deferred.
