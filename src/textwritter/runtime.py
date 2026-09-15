"""Runtime detection + engine selection — the single source of truth so the SAME
code runs unchanged on:
  - a local PC with an NVIDIA GPU  (local_gpu — full speed)
  - Google Colab                  (colab — Drive-backed checkpoints)
  - a CPU-only laptop             (local_cpu — dev/testing via the HWT engine)

Every entry point (scripts, notebooks, training) imports this module first and
prints the runtime card, then calls resolve_engine() so generation always picks
something that works on THIS machine — while still letting the user force any
engine with --engine / TW_ENGINE.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/Jeevant010/Text_Writter"

# repo root: <repo>/src/textwritter/runtime.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT_DIR = ROOT / "out"
ASSETS_DIR = ROOT / "assets"
SAMPLE_STYLES_DIR = ASSETS_DIR / "sample_styles"

# Engines. `kind` is the flag value users type; hwt is the everywhere engine,
# onedm is the GPU-fidelity engine (auto-chosen on CUDA machines).
ENGINES = {
    "hwt": {
        "label": "HWT (Handwriting Transformers)",
        "needs_cuda": False,
        "blurb": "few-shot transformer, CPU+GPU, ~685 MB one-time download",
    },
    "onedm": {
        "label": "One-DM (One-Shot Diffusion Mimicker)",
        "needs_cuda": True,
        "blurb": "highest single-photo fidelity, GPU-only, big auto-downloads",
    },
}


@dataclass
class Environment:
    kind: str                    # "colab" | "local_gpu" | "local_cpu"
    device: str                  # "cuda" | "cpu"
    gpu_name: str | None
    can_train: bool              # CUDA present?
    extras: dict = field(default_factory=dict)


def detect_environment() -> Environment:
    """Auto-detect where we're running. Order: Colab -> local CUDA -> CPU."""
    # Colab sets this reliably (module present and/or /content exists).
    in_colab = "google.colab" in sys.modules or os.path.exists("/content/.config")

    device, gpu_name = "cpu", None
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch missing/broken -> CPU path
        pass

    if in_colab:
        # On Colab the repo is cloned to /content/Text_Writter (see notebooks).
        drive_root = Path("/content/drive/MyDrive")
        return Environment("colab", device, gpu_name, can_train=(device == "cuda"),
                           extras={"drive_root": drive_root})
    if device == "cuda":
        return Environment("local_gpu", device, gpu_name, can_train=True)
    return Environment("local_cpu", device, gpu_name, can_train=False)


def repo_root() -> Path:
    """Repo root on this machine (Colab: /content/Text_Writter after clone)."""
    if ROOT.exists():
        return ROOT
    return Path.cwd()  # paranoia fallback for exotic launches


def add_src_to_path() -> None:
    """Make `import textwritter...` work without installing this repo."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def read_string_constant(path: Path, name: str) -> str:
    """Read `name = '<str>'` from a Python file without importing it.

    AST-based on purpose: these vendored constants contain escaped quotes, which
    regexes mangle.
    """
    import ast

    try:
        tree = ast.parse(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return ""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    return ""


def resolve_engine(choice: str | None = None) -> str:
    """Pick the engine to run. Order: --engine/TW_ENGINE override, else auto.

    auto  -> onedm on CUDA machines (best fidelity), hwt everywhere else.
    """
    choice = (choice or os.environ.get("TW_ENGINE", "")).strip().lower()
    if choice in ("onedm", "hwt"):
        engine = choice
    else:
        env = detect_environment()
        engine = "onedm" if env.device == "cuda" else "hwt"
    return engine


def validate_engine(engine: str) -> str | None:
    """None if OK on this machine, else a short reason string."""
    env = detect_environment()
    if engine == "onedm" and env.device != "cuda":
        return ("One-DM needs an NVIDIA CUDA GPU — not available here "
                "(device=%s). Run with --engine hwt on this machine, or run "
                "One-DM on Colab / a GPU PC." % env.device)
    return None


def engine_hint(engine: str) -> str:
    """One-line 'what is best to run and test creation' guidance per machine."""
    env = detect_environment()
    if env.kind == "colab":
        return ("Colab GPU detected — this is the best place for the One-DM "
                "(max-fidelity) creation test. Free tier is fine for inference.")
    if env.kind == "local_gpu":
        return ("GPU PC detected — run the quick creation test; use --engine "
                "onedm for best fidelity or hwt for speed.")
    return ("CPU-only machine (dev/testing) — the HWT engine runs here (~s per "
            "word). For max-fidelity One-DM output, run on Colab or a GPU PC.")


def print_runtime_card(engine: str | None = None) -> str:
    """Print where we are + which engine we'll use; returns chosen engine."""
    env = detect_environment()
    tag = {"colab": "GOOGLE COLAB", "local_gpu": "LOCAL GPU PC",
           "local_cpu": "LOCAL CPU (dev/testing)"}[env.kind]
    chosen = resolve_engine(engine) if engine in (None, "auto") else engine
    print(f"[env] {tag} | device={env.device} | gpu={env.gpu_name or 'none'}")
    print(f"[env] engine = {chosen} "
          f"({ENGINES[chosen]['label']}; {ENGINES[chosen]['blurb']})")
    if not env.can_train:
        print("[env] no CUDA here — fine-tuning is off; this machine still does "
              "HWT inference and judging.")
    return chosen
