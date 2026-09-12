"""Gemini API client — the ONLY network-dependent part of the agent.

Used for two things, nothing else:
  1. AI-writer mode: topic -> prose (+ LaTeX for math)
  2. Tweak parsing: "make line 2 messier" -> structured parameter deltas

Legibility judging is NOT done here — that's TrOCR, local and free.
Set GEMINI_API_KEY in the environment. Free tier is sufficient.
"""
from __future__ import annotations

import json
import os
from typing import Any

_MODEL = "gemini-2.0-flash"  # cheap + fast; swap for pro if drafting quality matters


def _client():
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set — get one at ai.google.dev (free)")
    genai.configure(api_key=key)
    return genai.GenerativeModel(_MODEL)


def write_content(topic: str) -> str:
    """Topic prompt -> full document text (prose, LaTeX blocks for math)."""
    prompt = (
        "Write a clear, well-structured document on the following topic. "
        "Use plain paragraphs. Put any mathematics in LaTeX between $$ markers. "
        "No markdown headers, no bullet symbols — pure prose and equations.\n\n"
        f"TOPIC: {topic}"
    )
    return _client().generate_content(prompt).text


def parse_tweak(feedback: str, blocks: list[dict]) -> list[dict[str, Any]]:
    """Free-text feedback -> [{block_index, jitter_delta, slant_delta}, ...]."""
    prompt = (
        "You map handwriting-tweak requests to JSON parameter deltas.\n"
        "Parameters: jitter_delta in [-1,1] (messiness change), "
        "slant_delta in degrees (e.g. -5 to +5).\n"
        f"Blocks (index: first 40 chars): "
        f"{[{'index': i, 'preview': b['text'][:40]} for i, b in enumerate(blocks)]}\n"
        f"User feedback: {feedback!r}\n"
        "Return ONLY a JSON array like "
        '[{"block_index": 1, "jitter_delta": 0.2, "slant_delta": -3}].'
    )
    raw = _client().generate_content(prompt).text
    try:
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except (json.JSONDecodeError, AttributeError):
        return []  # unparseable feedback -> no-op, user can rephrase
