"""Creation pipeline — one call that renders text in a handwriting style,
judges it, and reports. Engine-agnostic: HWT anywhere, One-DM on GPU.

Used by `textwritter.quicktest` (CLI) and by the notebooks, so Colab and a local
GPU PC run the exact same code.

Nothing here is allowed to hard-fail on missing optional pieces:
  * no style photo        -> bundled handwriting samples
  * no CUDA + onedm       -> validated earlier with clear guidance
  * judge model unavailable -> output is still produced, verdict "unjudged"
"""
from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path

from .runtime import (ENGINES, OUT_DIR, ROOT, detect_environment,
                      print_runtime_card, resolve_engine, validate_engine)

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog"

_ENGINE_FILES = {
    "hwt": ROOT / "experiments" / "02_hwt" / "engine.py",
    "onedm": ROOT / "experiments" / "01_onedm" / "engine.py",
}
_JUDGE_FILE = ROOT / "experiments" / "03_trocr_judge" / "judge.py"


def _load(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"engine module missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_judge(image: Path, expected: str, device: str = "auto") -> dict:
    """TrOCR read-back. Never raises — returns judged=False with a reason."""
    try:
        judge = _load(_JUDGE_FILE, "tw_trocr_judge")
        recognized = judge.recognize(image, device=device)
        score = judge.cer(expected, recognized)
        return {"judged": True, "recognized": recognized, "cer": score,
                "pass": score <= judge.DEFAULT_THRESHOLD,
                "threshold": judge.DEFAULT_THRESHOLD}
    except Exception as e:  # noqa: BLE001 - judging is optional, never fatal
        return {"judged": False, "reason": f"{type(e).__name__}: {e}"}


def _compare_image(style_dir: Path | None, generated: Path, out: Path) -> Path | None:
    """Stack a style sample above the generated image for a quick eyeball."""
    try:
        from PIL import Image
        gen = Image.open(generated).convert("RGB")
        tiles = []
        if style_dir:
            words = sorted(Path(style_dir).glob("*.png"))[:5]
            for w in words:
                im = Image.open(w).convert("RGB")
                scale = gen.height / max(1, im.height)
                tiles.append(im.resize((max(1, int(im.width * scale)), gen.height)))
        if tiles:
            total_w = sum(t.width for t in tiles) + 10 * (len(tiles) - 1)
            strip = Image.new("RGB", (total_w, gen.height), "white")
            x = 0
            for t in tiles:
                strip.paste(t, (x, 0))
                x += t.width + 10
            width = max(strip.width, gen.width)
            canvas = Image.new("RGB", (width, strip.height + gen.height + 12), "white")
            canvas.paste(strip, (0, 0))
            canvas.paste(gen, (0, strip.height + 12))
        else:
            canvas = gen
        canvas.save(out)
        return out
    except Exception:  # noqa: BLE001 - cosmetic only
        return None


def create(text: str | None = None, style: str | None = None, engine: str = "auto",
           out: str | Path | None = None, device: str | None = None,
           steps: int = 50, judge: bool = True) -> dict:
    """Render `text` in the handwriting style; returns a result dict."""
    text = (text or DEFAULT_TEXT).strip()
    chosen = print_runtime_card(engine)
    problem = validate_engine(chosen)
    if problem:
        raise RuntimeError(problem)

    out_path = Path(out) if out else OUT_DIR / f"quicktest_{chosen}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"_{chosen}_work"

    print(f"[run] engine={chosen} device={device or 'auto'} text={text!r}")
    eng = _load(_ENGINE_FILES[chosen], f"tw_engine_{chosen}")

    style_dir, source, warnings = eng.prepare_style(style, work)
    for w in warnings:
        print(f"[run] note: {w}")

    kwargs = {"device": device} if device else {}
    if chosen == "onedm":
        kwargs["steps"] = steps
    result = eng.generate(text, style_dir, out_path, **kwargs)
    result.update({"engine": chosen, "style_source": source, "warnings": warnings,
                   "out": out_path})

    if judge:
        verdict = run_judge(out_path, result.get("text", text), device=device or "auto")
    else:
        verdict = {"judged": False, "reason": "judging disabled"}
    result["verdict"] = verdict

    compare = _compare_image(style_dir, out_path, out_path.with_name(
        out_path.stem + "_compare.png"))
    result["compare"] = compare

    _report(result)
    return result


def _report(result: dict) -> None:
    v = result["verdict"]
    print("\n" + "-" * 62)
    print(f"engine     : {result['engine']} ({ENGINES[result['engine']]['label']})")
    print(f"style      : {result['style_source']}")
    print(f"expected   : {result.get('text', '')}")
    if v.get("judged"):
        print(f"recognized : {v['recognized']}")
        print(f"CER        : {v['cer']:.3f}  (threshold {v['threshold']})")
        print(f"verdict    : {'PASS' if v['pass'] else 'FAIL — re-roll with a new seed'}")
    else:
        print(f"verdict    : UNJUDGED ({v.get('reason', 'judge unavailable')})")
    print(f"image      : {result['out']}")
    if result.get("compare"):
        print(f"compare    : {result['compare']}")
    print("-" * 62)


def render_line(text: str, style=None, engine: str = "auto", out=None,
                device: str | None = None, seed: int = 0, steps: int = 50) -> Path:
    """One rendered image for a block of text — the backend hook used by the agent.

    `seed` is accepted for interface compatibility (deterministic re-rolls land
    with the One-DM wiring); HWT samples style words per run.
    """
    chosen = resolve_engine(engine)
    eng = _load(_ENGINE_FILES[chosen], f"tw_engine_{chosen}")
    style_dir, _, _ = eng.prepare_style(style, OUT_DIR / "_render_work")
    out_path = Path(out) if out else OUT_DIR / f"line_{abs(hash((text, seed))) % 10**8}.png"
    res = eng.generate(text, style_dir, out_path, device=device,
                       **({"steps": steps} if chosen == "onedm" else {}))
    return res["image"]
