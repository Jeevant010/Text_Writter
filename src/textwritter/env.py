"""Checkpoint + download management for every script and notebook.

Rules (project-wide):
  1. Checkpoint already on disk  -> use it, download nothing.
  2. Else auto-fetch it         -> single-file gdown, or gdown --folder for the
                                   One-DM bundles (checkpoint + English data).
  3. Else CLEARLY guide the user -> never a raw stack trace. The caller decides
                                   how to proceed (often: run the HWT engine).

Same code runs on Colab (Drive-backed) and local machines (disk-backed) — that is
the whole point of this module.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .runtime import ROOT, Environment, detect_environment

CKPT_ROOT = Path(os.environ.get("TW_CKPT_DIR", str(ROOT / "checkpoints")))
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# Registry of known checkpoints / bundles. `gdrive` is a file id; `gdrive_folder`
# is a Google Drive folder id (downloaded with `gdown --folder`). Folder ids are
# overridable via env so a stale id never bricks a run.
CHECKPOINTS: dict[str, dict] = {
    "hwt_pretrained": {
        "filename": "iam_model.pth",
        "gdrive": "16g9zgysQnWk7-353_tMig92KsZsrcM6k",  # official bundle (INSTALL.md)
        "note": "Official HWT bundle (files.zip: iam_model.pth + IAM pickles)",
    },
    "onedm_pretrained": {
        "filename": "one_dm_pretrained.pth",
        "gdrive_folder": os.environ.get(
            "ONEDM_CKPT_FOLDER_ID", "10KOQ05HeN2kaR2_OCZNl9D_Kh1p8BDaa"),
        "subfolder": "onedm_ckpt",
        "match": "ckpt",  # inside the folder, the file whose name contains this
        "note": ("Official One-DM checkpoint (Google Drive folder, per the repo "
                 "README Model Zoo; file One-DM-ckpt.pt)"),
    },
    "onedm_data": {
        "filename": None,  # a folder bundle, not a single file
        "gdrive_folder": os.environ.get(
            "ONEDM_DATA_FOLDER_ID", "108TB-z2ytAZSIEzND94dyufybjpqVyn6"),
        "subfolder": "onedm_data",
        "note": ("Official One-DM English dataset bundle (IAM64 layout + text "
                 "lists), per the repo README datasets link"),
    },
}


def _gdown_install() -> None:
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass
    attempts = [
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        ["uv", "pip", "install", "--python", sys.executable, "gdown"],
    ]
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True)
            import gdown  # noqa: F401
            return
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("gdown is required for checkpoint downloads — install it: "
                       "pip install gdown")


def _download_folder(info: dict, dest: Path) -> Path | None:
    """gdown --folder into dest; returns dest or None on failure."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] downloading folder {info['gdrive_folder']} -> {dest} "
          "(large; one-time)")
    _gdown_install()
    import gdown
    try:
        gdown.download_folder(id=info["gdrive_folder"], output=str(dest),
                              quiet=False)
    except Exception as e:  # noqa: BLE001 - gdown raises many things
        print(f"[ckpt] folder download failed: {e}")
        print(f"[ckpt] fix: open the folder link manually, or set the env var "
              f"(see {info['note']}).")
        return None
    return dest


def _find_by_match(folder: Path, needle: str) -> Path | None:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and needle.lower() in p.name.lower():
            return p
    return None


def get_checkpoint(name: str, interactive: bool = True) -> Path | None:
    """Return path to checkpoint/bundle `name`, obtaining it if necessary.

    1. Already on disk            -> return it (nothing downloaded).
    2. gdrive id configured       -> download via gdown (single file).
    3. gdrive_folder configured   -> download via gdown --folder, locate file.
    4. Otherwise                  -> print exact manual steps; return None
                                     (caller decides how to proceed).
    """
    info = CHECKPOINTS[name]
    filename = info.get("filename")
    if filename:
        path = CKPT_ROOT / filename
        if path.exists():
            print(f"[ckpt] found {path} ({path.stat().st_size / 1e6:.0f} MB) — using it")
            return path
    else:
        # folder-style bundle (onedm_data): report as ready if folder non-empty
        path = CKPT_ROOT / info["subfolder"]
        if path.exists() and any(path.iterdir()):
            print(f"[ckpt] found bundle folder {path} — using it")
            return path

    if info.get("gdrive"):
        print(f"[ckpt] {filename} not present — downloading…")
        _gdown_install()
        import gdown
        gdown.download(id=info["gdrive"], output=str(path), quiet=False)
        return path if path.exists() else None

    if info.get("gdrive_folder"):
        print(f"[ckpt] {info['subfolder']} not present — downloading folder…")
        bundle = _download_folder(info, CKPT_ROOT / info["subfolder"])
        if bundle is None:
            return None
        if filename:
            match = _find_by_match(bundle, info["match"])
            if match is None:
                print(f"[ckpt] downloaded folder, but no file matching "
                      f"{info['match']!r} found in {bundle}. Contents:")
                for p in sorted(bundle.rglob("*"))[:30]:
                    print("  ", p)
                return None
            shutil.copy2(match, CKPT_ROOT / filename)
            print(f"[ckpt] {match.name} -> {CKPT_ROOT / filename}")
            return CKPT_ROOT / filename
        return bundle

    # No automatic route — guide the human (30 seconds of manual work, once).
    print(f"\n[ckpt] '{name}' is missing and has no download configured.")
    print(f"       {info['note']}")
    print("       Fix (either):")
    print(f"         a) drop the needed file into {CKPT_ROOT}/ yourself, or")
    print("         b) set the right env var (ONEDM_CKPT_FOLDER_ID / "
          "ONEDM_DATA_FOLDER_ID) and re-run.\n")
    if interactive:
        manual = input("Paste a local path to the file now, or Enter to abort: ").strip()
        if manual and Path(manual).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manual, path)
            return path
    return None


# ---------------------------------------------------------------------------
# Resume: works identically on Colab (Drive) and local (disk)
# ---------------------------------------------------------------------------

def resume_source(env: Environment, run_name: str) -> Path | None:
    """Newest checkpoint to resume FROM, or None to start from pretrained base."""
    if env.kind == "colab":
        drive_dir = env.extras["drive_root"] / "textwritter_checkpoints" / run_name
        ckpts = sorted(drive_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime) \
            if drive_dir.exists() else []
        if ckpts:
            local = CKPT_ROOT / ckpts[-1].name
            shutil.copy2(ckpts[-1], local)
            print(f"[resume] Drive -> {local.name}")
            return local
    else:
        ckpts = sorted((CKPT_ROOT / run_name).glob("*.pth"),
                       key=lambda p: p.stat().st_mtime) \
            if (CKPT_ROOT / run_name).exists() else []
        if ckpts:
            print(f"[resume] local -> {ckpts[-1].name}")
            return ckpts[-1]
    print("[resume] no prior checkpoint — starting from pretrained base")
    return None


def backup_checkpoint(env: Environment, run_name: str, src: Path) -> None:
    """Persist a checkpoint so a crash/tier-switch never loses it."""
    if env.kind == "colab":
        dst_dir = env.extras["drive_root"] / "textwritter_checkpoints" / run_name
    else:
        dst_dir = CKPT_ROOT / run_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    print(f"[checkpoint] backed up -> {dst_dir / src.name}")
