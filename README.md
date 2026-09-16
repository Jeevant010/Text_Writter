# Text_Writter

AI agent that learns a person's handwriting from a small sample and renders arbitrary
content (prose + math) in that handwriting. Terminal-first prototype; production
website integration comes later.

## Run it anywhere (no Colab required)

One command creates handwriting from text. The engine is **auto-picked for the machine
you're on** — One-DM on an NVIDIA GPU (best fidelity), HWT everywhere else (CPU
included) — and you can always force either one. **No handwriting photo?** Bundled
samples are used, so a run always produces output.

| Your machine | What to run | Engine picked | One-time download |
|---|---|---|---|
| This laptop (no NVIDIA GPU) | `python src/textwritter/quicktest.py` | HWT (CPU, ~s/word) | ~685 MB bundle + ~300 MB judge |
| A friend's GPU PC | `python src/textwritter/quicktest.py` | **One-DM** (drop to `--engine hwt` for speed) | One-DM weights + English data |
| Google Colab (GPU) | open [`notebooks/00_quick_test.ipynb`](notebooks/00_quick_test.ipynb) → Run all | **One-DM** | same |

```bash
# create handwriting now (bundled style if you don't pass --style)
python src/textwritter/quicktest.py --text "The quick brown fox jumps over the lazy dog"

# with your own handwriting photo
python src/textwritter/quicktest.py --style samples/my_handwriting.jpg

# force an engine (auto | hwt | onedm); TW_ENGINE env var works too
python src/textwritter/quicktest.py --engine hwt
```

Every run prints a runtime card: where you are, which engine, what's best to run here.
If the judge model can't load, you still get the image — the verdict just says
"unjudged".

### Engine-by-engine scripts

```bash
# HWT — runs on CPU or GPU
python experiments/02_hwt/test_inference.py --text "The quick brown fox"

# One-DM — CUDA GPU only; --check reports readiness before any big download
python experiments/01_onedm/test_inference.py --check
python experiments/01_onedm/test_inference.py --text "Hello world"

# judge any image yourself
python experiments/03_trocr_judge/judge.py --image out/quicktest_hwt.png --expect "Hello world"
```

### Fine-tuning (optional, GPU — the "6-hour class" job)

Try pretrained inference first — fine-tuning is insurance, not step 1. The **same
command** runs on a GPU PC and on Colab; checkpoints resume/back up automatically
(Drive on Colab, local disk on a PC), so an interruption never loses the run.

```bash
# GPU PC and Colab (inside a notebook cell) run the identical command:
python training/finetune.py --model hwt --data my_handwriting.zip
python training/finetune.py --model onedm --data my_handwriting.zip   # needs One-DM bundle

# see what's required first, without training
python training/finetune.py --model hwt --check
```

Notebook front-ends for both environments:
[`notebooks/02_hwt_finetune.ipynb`](notebooks/02_hwt_finetune.ipynb) ·
[`notebooks/01_onedm_finetune.ipynb`](notebooks/01_onedm_finetune.ipynb).

### Ruled assignment pages (CPU — current working path)

For a multi-page homework PDF that *looks like* a student notebook (blue pen, Caveat,
math markup). This is **not** cloning your hand. Details: [docs/ASSIGNMENT_PAGES.md](docs/ASSIGNMENT_PAGES.md), [AGENTS.md](AGENTS.md).

```bash
PYTHONPATH=src python src/textwritter/assignment_engine.py \
  samples/assignment2_solutions.txt output/a2/Assignment2_solutions.pdf \
  --style caveat --ink blue
```

### Docs

| Doc | What it's for |
|---|---|
| [AGENTS.md](AGENTS.md) | What future agents must not regress (joins, ruled lines, two pipelines) |
| [docs/ASSIGNMENT_PAGES.md](docs/ASSIGNMENT_PAGES.md) | Assignment PDF how-to + markup + glyph sheet |
| [docs/FEASIBILITY_PLAN.md](docs/FEASIBILITY_PLAN.md) | The master plan (locked) |
| [docs/MODELS_GUIDE.md](docs/MODELS_GUIDE.md) | Fact-checked model comparison + accuracy |
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | Running/training on Colab **or** a GPU PC (same commands) |
| [docs/AGENT_SERVER_ROADMAP.md](docs/AGENT_SERVER_ROADMAP.md) | LangGraph loop, Gemini, server & MCP — and what NOT to build yet |
| [experiments/](experiments/) | Per-model working options, each with coder + non-coder usage docs |

## Layout

```
├── docs/                  # plan + guides (start here)
├── notebooks/             # 00 quick test · 01/02 optional fine-tunes (Colab or local)
├── experiments/           # 01 One-DM (GPU) · 02 HWT (everywhere) · 03 TrOCR judge
├── assets/sample_styles/  # bundled handwriting samples (used when you pass no photo)
├── samples/               # drop your handwriting photo here
└── src/textwritter/       # the product code
    ├── runtime.py              #   environment + engine detection
    ├── pipeline.py             #   create() → generate → judge → compare
    ├── quicktest.py            #   one-command HWT/One-DM CLI
    ├── assignment_engine.py  #   CPU ruled assignment pages + math
    ├── glyphs.py              #   print/slice handwritten symbol sheet
    ├── write_assignment.py    #   HWT multi-page (needs style PNG)
    └── agent/                  #   LangGraph satisfaction loop + Gemini client
```

## Rules that keep this project fast

1. Never train from scratch — pretrained checkpoints only.
2. Inference before fine-tuning — the fine-tune flows are insurance, not step 1.
3. One interface, two engines — HWT everywhere, One-DM when a CUDA GPU is present.
   Nothing is Colab-only; nothing is GPU-only unless the model itself requires it.
