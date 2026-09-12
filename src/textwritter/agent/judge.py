"""TrOCR judge adapter for the agent loop — wraps experiments/03_trocr_judge."""
from __future__ import annotations


def cer_of(image_path: str, expected_text: str) -> float:
    """Character Error Rate of a rendered block vs its intended text."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                         / "experiments" / "03_trocr_judge"))
    from judge import cer, recognize  # noqa: E402
    return cer(expected_text, recognize(Path(image_path)))
