# Agent, Server & MCP Roadmap — Including What You DON'T Need

You asked me to flag anything that's off or unnecessary. Here it is, honestly.

## 1. MCP server: build it LAST, not first

What you described — "users on my website, running on my server" — needs **FastAPI**,
not MCP. They're for different consumers:

| | Consumer | When you need it |
|---|---|---|
| **FastAPI wrapper** | Your website's frontend | **Production — this is the real one** |
| **MCP server** | LLM clients (Claude Desktop, Cursor) on *your* machine | Optional dev convenience — drive the agent conversationally while building |

I've scaffolded [`src/textwritter/mcp_server.py`](../src/textwritter/mcp_server.py)
(thin, 4 tools) so the option exists, but **do not invest in it now**. The FastAPI
wrapper over the same `agent/loop.py` functions is Phase 6 of the feasibility plan
and is what your dashboard will actually call.

## 2. Gemini in the loop: two jobs ONLY

- ✅ **AI-writer mode** — topic → prose + LaTeX (`gemini.write_content`)
- ✅ **Tweak parsing** — "make line 2 messier" → parameter deltas (`gemini.parse_tweak`)
- ❌ **NOT the legibility judge** — that's TrOCR: local, free, ~1 s, deterministic.
  Using Gemini there adds API latency + cost for *worse* character-level precision.

End users' machines do **zero** compute: generation + judging run on your server's
GPU/CPU; the only external call is server→Gemini API with *your* key. Exactly as you
wanted.

## 3. The LangGraph satisfaction loop (implemented)

[`src/textwritter/agent/loop.py`](../src/textwritter/agent/loop.py):

```
draft ──► render ──► judge ──┬─ CER>15% & tries<3 ─► auto_reroll ─► render
                             └─ ok ─► human_review (interrupt — graph FREEZES)
                                           │
                          ┌──── approve ───┴── feedback ────┐
                          ▼                                 ▼
                        export                        parse_tweak ─► render
```

- `interrupt_before=["human_review"]` = the "user takes as long as they want" part.
  With a checkpointer (SQLite locally, Postgres in prod), the frozen session survives
  server restarts — user closes the tab, comes back tomorrow, loop resumes.
- **Auto re-roll before human eyes:** TrOCR CER > 0.15 → new seed, up to 3× per block.
  Bad output rarely reaches the user — this is the quality gate doing quiet work.
- Every block carries `seed/jitter/slant` → any block is deterministically re-rollable.

## 4. Deployment shape (your server)

```
[Website dashboard] ──► [FastAPI] ──► [LangGraph loop] ──► [One-DM on RTX 4060]
                                            │                [TrOCR judge]
                                            └──► [Gemini API] (content + tweaks only)
```

- Server does ALL compute. Users upload a photo + type text. That's it.
- Sessions/checkpoints: Postgres checkpointer in prod, SQLite for dev.
- 8 GB VRAM fits One-DM (line-level images) + TrOCR comfortably.

## 5. Priority order (don't waste time on smaller things)

1. **Run `notebooks/00_quick_test_colab.ipynb`** — see YOUR handwriting via pretrained
   One-DM today. This is the single most validating step.
2. If good → build CLI around it (Phase 0–2). If style is off → optional fine-tune
   notebooks (they resume from Drive; see [COLAB_GUIDE](COLAB_GUIDE.md)).
3. Wire `agent/backends.py` to the working backend → the LangGraph loop comes alive.
4. FastAPI wrapper → website. (MCP only if you later want Claude/Cursor to drive it.)

## 6. Corrections to your earlier chat (fact-checked)

- One-DM is **ECCV 2024**, not AAAI 2024. HWT is **ICCV 2021**, not CVPR 2021.
- One-DM's open-source v1 generates **word-level** images — full lines are assembled
  by our compositor. Native full-line is the same team's **DiffBrush** (ICCV 2025) —
  a future upgrade, not needed now.
- The "FID 15.2 / 92% word accuracy" numbers in your chat were unverifiable.
  Verified numbers are in [MODELS_GUIDE](MODELS_GUIDE.md) and each experiment README.
- HWT already has an **official Colab for custom handwriting** — we build on it
  rather than reinventing.
