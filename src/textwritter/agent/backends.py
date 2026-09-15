"""Backend protocol — the swap point between One-DM (GPU) and HWT (CPU).

Both official repos are driven as-is through their engine adapters; this module
exposes the one function the agent loop needs and lets runtime.py pick which
engine is active.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textwritter.pipeline import render_line as _render_line  # noqa: E402
from textwritter.runtime import resolve_engine  # noqa: E402


def render_line(text: str, style_id: str | None = None, seed: int = 0,
                jitter: float = 0.0, slant_deg: float = 0.0,
                engine: str = "auto", out: str | Path | None = None) -> str:
    """Render one line of handwriting; returns the path to the PNG.

    `style_id` may be a photo path; None uses the bundled handwriting samples.
    `engine="auto"` uses One-DM on a CUDA GPU and HWT everywhere else.
    `seed`/`jitter`/`slant_deg` are accepted for interface compatibility —
    deterministic parameter control lands with the enroll/profile step.
    """
    path = _render_line(text, style=style_id, engine=engine, out=out, seed=seed)
    return str(path)
