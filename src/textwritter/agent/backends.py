"""Backend protocol — the swap point between One-DM (GPU) and HWT (CPU).

Both official repos are driven as-is; this module picks which one is active.
"""
from __future__ import annotations


def render_line(text: str, style_id: str, seed: int,
                jitter: float, slant_deg: float) -> str:
    """Render one line of handwriting; returns path to the PNG.

    TODO(phase-2): wire to experiments/01_onedm (GPU) or 02_hwt (CPU),
    selected by torch.cuda.is_available(). Until then this is a stub that
    raises loudly instead of silently producing nothing.
    """
    raise NotImplementedError(
        "No handwriting backend wired yet. "
        "Run experiments/01_onedm/test_inference.py or 02_hwt first, "
        "then wrap it here."
    )
