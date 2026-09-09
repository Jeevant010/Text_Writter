# Text_Writter

AI agent that learns a person's handwriting from a small sample and renders arbitrary
content (prose + math) in that handwriting. Terminal-first prototype; production
website integration comes later.

## Start here

1. **See your own handwriting synthesized today (free GPU, no training):**
   upload [`notebooks/00_quick_test_colab.ipynb`](notebooks/00_quick_test_colab.ipynb)
   to Google Colab → Runtime → GPU → Run all.
2. Read the docs:

| Doc | What it's for |
|---|---|
| [docs/FEASIBILITY_PLAN.md](docs/FEASIBILITY_PLAN.md) | The master plan (locked) |
| [docs/MODELS_GUIDE.md](docs/MODELS_GUIDE.md) | Fact-checked model comparison + accuracy |
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | Optional fine-tuning + checkpoint-resume across Colab tiers |
| [docs/AGENT_SERVER_ROADMAP.md](docs/AGENT_SERVER_ROADMAP.md) | LangGraph loop, Gemini, server & MCP — and what NOT to build yet |
| [experiments/](experiments/) | Per-model working options, each with coder + non-coder usage docs |

## Layout

```
├── docs/                  # plan + guides (start here)
├── notebooks/             # Colab: 00 quick test · 01/02 optional fine-tunes (Drive resume)
├── experiments/           # 01 One-DM (GPU, primary) · 02 HWT (CPU, fallback) · 03 TrOCR judge
└── src/textwritter/       # the product code
    └── agent/             # LangGraph satisfaction loop + Gemini client
```

## Rules that keep this project fast

1. Never train from scratch — pretrained checkpoints only.
2. Inference before fine-tuning — the fine-tune notebooks are insurance, not step 1.
3. One production engine (One-DM on GPU); everything else is a fallback behind the
   same interface.
