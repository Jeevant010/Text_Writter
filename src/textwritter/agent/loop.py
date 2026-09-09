"""LangGraph satisfaction loop — the "iterate until the user is happy" agent.

Flow:
    draft (Gemini, if AI-writer mode)
      -> render (One-DM/HWT backend)
      -> judge (TrOCR read-back, CER)
      -> auto re-roll if illegible (max N)
      -> HUMAN REVIEW (interrupt — user approves or requests tweaks)
      -> tweak parse (Gemini: "make it messier" -> parameter deltas)
      -> re-render only affected blocks
      -> loop until approved -> export

Runs on YOUR server: generation + judging are local GPU/CPU work.
Only the LLM calls (content drafting, tweak parsing) go to the Gemini API —
end users' machines do zero compute, exactly as required.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class Block(TypedDict):
    block_id: str
    kind: Literal["paragraph", "heading", "math"]
    text: str                      # prose text or LaTeX
    seed: int
    jitter: float                  # 0.0 tidy .. 1.0 messy
    slant_deg: float
    image_path: Optional[str]
    cer: Optional[float]           # TrOCR read-back score


class AgentState(TypedDict):
    session_id: str
    mode: Literal["verbatim", "ai_writer"]
    raw_input: str                 # user's text or topic prompt
    style_profile_id: str          # enrolled handwriting profile
    blocks: list[Block]
    pending_tweak: Optional[str]   # free-text user feedback, e.g. "line 2 messier"
    approved: bool
    reroll_count: int


# ---------------------------------------------------------------------------
# Node implementations (thin — real backends plug in here)
# ---------------------------------------------------------------------------

CER_THRESHOLD = 0.15
MAX_AUTO_REROLLS = 3


def draft_content(state: AgentState) -> dict:
    """AI-writer mode: Gemini turns a topic into prose+LaTeX blocks.
    Verbatim mode: pass the user's text through unchanged."""
    if state["mode"] == "verbatim":
        blocks = _split_into_blocks(state["raw_input"])
        return {"blocks": blocks}
    from . import gemini
    text = gemini.write_content(state["raw_input"])  # topic -> full text
    return {"blocks": _split_into_blocks(text)}


def render_blocks(state: AgentState) -> dict:
    """Generate each block via the active handwriting backend (One-DM/HWT)."""
    from . import backends
    for b in state["blocks"]:
        if b["image_path"] is None or b.get("_dirty", False):  # type: ignore
            b["image_path"] = backends.render_line(
                text=b["text"],
                style_id=state["style_profile_id"],
                seed=b["seed"], jitter=b["jitter"], slant_deg=b["slant_deg"],
            )
            b["_dirty"] = False  # type: ignore
    return {"blocks": state["blocks"]}


def judge_blocks(state: AgentState) -> dict:
    """TrOCR reads each rendered block back; CER scores legibility."""
    from . import judge
    for b in state["blocks"]:
        if b["image_path"]:
            b["cer"] = judge.cer_of(b["image_path"], b["text"])
    return {"blocks": state["blocks"]}


def auto_reroll(state: AgentState) -> dict:
    """Illegible blocks get a new seed automatically (bounded)."""
    rerolls = state["reroll_count"]
    for b in state["blocks"]:
        if b["cer"] is not None and b["cer"] > CER_THRESHOLD and rerolls < MAX_AUTO_REROLLS:
            b["seed"] += 1
            b["_dirty"] = True  # type: ignore
            rerolls += 1
    return {"blocks": state["blocks"], "reroll_count": rerolls}


def parse_tweak(state: AgentState) -> dict:
    """Gemini turns 'make line 2 messier and more slanted' into block deltas."""
    if not state["pending_tweak"]:
        return {}
    from . import gemini
    deltas = gemini.parse_tweak(state["pending_tweak"], state["blocks"])
    for d in deltas:
        b = state["blocks"][d["block_index"]]
        b["jitter"] = _clamp01(b["jitter"] + d.get("jitter_delta", 0.0))
        b["slant_deg"] += d.get("slant_delta", 0.0)
        b["seed"] += 1
        b["_dirty"] = True  # type: ignore
    return {"blocks": state["blocks"], "pending_tweak": None}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_judge(state: AgentState) -> str:
    illegible = any(b["cer"] is not None and b["cer"] > CER_THRESHOLD
                    for b in state["blocks"])
    if illegible and state["reroll_count"] < MAX_AUTO_REROLLS:
        return "auto_reroll"
    return "human_review"          # -> interrupt() in the compiled graph


def route_after_review(state: AgentState) -> str:
    if state["approved"]:
        return "export"
    return "parse_tweak"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Compile the LangGraph. `interrupt()` at human_review freezes the thread;
    the CLI (and later the website) resumes it with the user's verdict."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("draft", draft_content)
    g.add_node("render", render_blocks)
    g.add_node("judge", judge_blocks)
    g.add_node("auto_reroll", auto_reroll)
    g.add_node("parse_tweak", parse_tweak)
    g.add_node("export", lambda s: {"approved": True})

    g.set_entry_point("draft")
    g.add_edge("draft", "render")
    g.add_edge("render", "judge")
    g.add_conditional_edges("judge", route_after_judge,
                            {"auto_reroll": "auto_reroll", "human_review": "human_review"})
    g.add_edge("auto_reroll", "render")
    # human_review is an interrupt point — added at compile time below
    g.add_conditional_edges("human_review", route_after_review,
                            {"export": "export", "parse_tweak": "parse_tweak"})
    g.add_edge("parse_tweak", "render")
    g.add_edge("export", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],   # graph pauses; UI shows the page
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_into_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    for i, para in enumerate(p for p in text.split("\n\n") if p.strip()):
        blocks.append(Block(
            block_id=f"b{i}", kind="paragraph", text=para.strip(),
            seed=0, jitter=0.4, slant_deg=0.0, image_path=None, cer=None,
        ))
    return blocks


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
