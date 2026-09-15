#!/usr/bin/env python3
"""Unified fine-tune — runs UNCHANGED on a GPU PC or on Google Colab.

    # on a friend's GPU PC:
    python training/finetune.py --model hwt --data my_handwriting.zip

    # on Colab (same command — env auto-detected, checkpoints go to Drive):
    !python training/finetune.py --model hwt --data my_handwriting.zip

    # resume after any interruption (same command again — it resumes):
    python training/finetune.py --model hwt --data my_handwriting.zip

    # see what's needed without training:
    python training/finetune.py --model onedm --check

Rules this enforces (project-wide):
  * never train from scratch — always start from the official pretrained weights
  * never hard-crash on a missing checkpoint — get it or guide the user
  * CPU-only machines: refuse to train, but say exactly where to run instead
  * every epoch's checkpoint is copied off the machine (Drive on Colab, disk on PC),
    so a tier switch or crash never costs the whole run
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textwritter.env import (CHECKPOINTS, backup_checkpoint,  # noqa: E402
                             get_checkpoint, resume_source)
from textwritter.runtime import detect_environment  # noqa: E402

ENGINE_DIRS = {
    "hwt": ROOT / "experiments" / "02_hwt",
    "onedm": ROOT / "experiments" / "01_onedm",
}
PRETRAINED = {"hwt": "hwt_pretrained", "onedm": "onedm_pretrained"}
WRITER = "me"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_engine(model: str):
    import importlib.util

    path = ENGINE_DIRS[model] / "engine.py"
    spec = importlib.util.spec_from_file_location(f"tw_engine_{model}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_zip_dataset(data: Path, work: Path) -> list[tuple[Path, str]]:
    """Extract `<images> + labels.txt` (one 'file.png<TAB>text' per line)."""
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data) as z:
        z.extractall(work)
    labels = next((p for p in work.rglob("labels.txt")), None)
    if labels is None:
        raise SystemExit(f"[!] no labels.txt inside {data}")
    items = []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.rsplit(None, 1)
        if len(parts) < 2:
            continue
        name, text = parts[0].strip(), parts[-1].strip()
        img = next((p for p in work.rglob(Path(name).name)), None)
        if img is None:
            print(f"[data] warning: image not found for {name!r} — skipping")
            continue
        items.append((img, text))
    if not items:
        raise SystemExit(f"[!] no usable image/label pairs found in {data}")
    print(f"[data] {len(items)} handwriting samples from {data.name}")
    return items


def _watch_and_backup(env, run: str, dirs: list[Path], stop: threading.Event) -> None:
    """Copy new checkpoints off-machine while training runs (tier-drop insurance)."""
    seen: set[Path] = set()
    while not stop.is_set():
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*.pth")) + sorted(d.rglob("*.pt")):
                if p not in seen and p.stat().st_size > 0:
                    try:
                        backup_checkpoint(env, run, p)
                    except Exception as e:  # noqa: BLE001
                        print(f"[checkpoint] backup failed for {p.name}: {e}")
                    seen.add(p)
        stop.wait(60)


def _run_monitored(cmd: list[str], cwd: Path, env, run: str,
                   watch_dirs: list[Path]) -> int:
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_and_backup,
                               args=(env, run, watch_dirs, stop), daemon=True)
    watcher.start()
    print("+", " ".join(str(c) for c in cmd))
    try:
        return subprocess.run(cmd, cwd=cwd).returncode
    finally:
        stop.set()
        watcher.join(timeout=5)


def _patch_params(vendor: Path, updates: dict[str, str]) -> None:
    """Rewrite simple `KEY = value` assignments in the vendored params.py."""
    params = vendor / "params.py"
    backup = vendor / "params.py.tw_orig"
    if not backup.exists():
        shutil.copy2(params, backup)
    text = backup.read_text()
    for key, value in updates.items():
        pattern = re.compile(rf"^(\s*){key}\s*=.*$", re.M)
        if not pattern.search(text):
            text += f"\n{key} = {value}\n"
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}{key} = {value}", text, count=1)
    params.write_text(text)


# ---------------------------------------------------------------------------
# HWT fine-tune (fully wired)
# ---------------------------------------------------------------------------

_HWT_TRAIN_DRIVER = r'''
import argparse, json, os, shutil, sys, types
import torch

try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from params import *
from data.dataset import TextDataset, TextDatasetval
from models.model import TRGAN

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--resume", default="")
ap.add_argument("--run-dir", required=True)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
device = torch.device(a.device)
os.makedirs(a.run_dir, exist_ok=True)

dataset = torch.utils.data.DataLoader(
    TextDataset(num_examples=NUM_EXAMPLES), batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=False)
val = torch.utils.data.DataLoader(
    TextDatasetval(num_examples=NUM_EXAMPLES), batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=False)

model = TRGAN()
if a.resume:
    model.load_state_dict(torch.load(a.resume, map_location="cpu"))
    print("resumed from " + a.resume)
else:
    model.netG.load_state_dict(torch.load(a.base, map_location="cpu"))
    print("initialised generator from pretrained " + a.base)
model.to(device)
model.train()

step = 0
for epoch in range(a.epochs):
    for data in dataset:
        model._set_input(data)
        model.optimize_G_only(); model.optimize_G_step()
        model._set_input(data)
        model.optimize_D_OCR(); model.optimize_D_OCR_step()
        model._set_input(data)
        model.optimize_G_WL(); model.optimize_G_step()
        model._set_input(data)
        model.optimize_D_WL(); model.optimize_D_WL_step()
        step += 1
    out = os.path.join(a.run_dir, "model.pth")
    torch.save(model.state_dict(), out)
    print("TW_EPOCH {} saved {}".format(epoch, out), flush=True)
print("TW_DONE")
'''


def train_hwt(args, env, base: Path | None, resume: Path | None) -> int:
    eng = _load_engine("hwt")
    vendor = eng.ensure_repo()  # fetches the official bundle (weights + IAM pickles)
    weights = vendor / "files" / "iam_model.pth"
    if not weights.exists():
        raise SystemExit(f"[!] HWT weights missing at {weights}")
    base = weights
    pickle_path = vendor / "files" / "custom.pickle"
    _build_hwt_pickle(args.data, pickle_path)

    _patch_params(vendor, {
        "DATASET_PATHS": "'files/custom.pickle'",
        "EXP_NAME": f"'{args.run}'",
        "EPOCHS": str(args.epochs),
        "batch_size": str(args.batch),
        "G_LR": repr(args.lr), "D_LR": repr(args.lr),
        "W_LR": repr(args.lr), "OCR_LR": repr(args.lr),
        "RESUME": "False",
    })

    driver = vendor / "_tw_train_driver.py"
    driver.write_text(_HWT_TRAIN_DRIVER)
    run_dir = vendor / "saved_models" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, driver.name,
           "--base", str(base), "--resume", str(resume or ""),
           "--run-dir", str(run_dir), "--epochs", str(args.epochs),
           "--device", env.device]
    rc = _run_monitored(cmd, cwd=vendor, env=env, run=args.run, watch_dirs=[run_dir])
    if rc != 0:
        print(f"[!] HWT training exited with code {rc}")
    else:
        ckpt = run_dir / "model.pth"
        if ckpt.exists():
            backup_checkpoint(env, args.run, ckpt)
    return rc


def _build_hwt_pickle(data: Path, dest: Path) -> None:
    import pickle

    from PIL import Image

    items = _read_zip_dataset(data, dest.parent / "_tw_data")
    samples = [{"img": Image.open(p).convert("L"), "label": text} for p, text in items]
    bundle = {"train": {WRITER: samples}, "test": {WRITER: samples}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[data] packed {len(samples)} samples -> {dest}")


# ---------------------------------------------------------------------------
# One-DM fine-tune (best effort: needs the official bundle + OCR weights)
# ---------------------------------------------------------------------------

def train_onedm(args, env, base: Path, resume: Path | None) -> int:
    eng = _load_engine("onedm")
    vendor = eng.ensure_repo()

    ckpt_folder = CHECKPOINTS["onedm_pretrained"]
    from textwritter.env import CKPT_ROOT

    bundle = CKPT_ROOT / ckpt_folder["subfolder"]
    ocr = next(iter(bundle.rglob("vae_HTR138.pth")), None) if bundle.exists() else None
    if ocr is None:
        print("[onedm] OCR/VAE weights (vae_HTR138.pth) not found in the official "
              "checkpoint folder.")
        print("        Fix: let the checkpoint folder download (remove "
              f"{bundle} and re-run), or place vae_HTR138.pth there.")
        return 1

    eng._content_asset(vendor)  # data/unifont.pickle (printed glyph conditions)
    data_root = eng.DATA_ROOT
    _lay_onedm_training_data(args.data, data_root)
    cfg = eng._write_config(vendor, data_root)
    # The trainer reads train lists from a fixed path; point it at our data root.
    shutil.copy2(data_root / "IAM64_train.txt", vendor / "data" / "IAM64_train.txt")

    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nnodes=1", "--nproc_per_node=1", "train_finetune.py",
           "--cfg", f"configs/{cfg.name}",
           "--one_dm", str(resume or base),
           "--ocr_model", str(ocr),
           "--log", args.run]
    watch = [vendor / "Saved"]
    rc = _run_monitored(cmd, cwd=vendor, env=env, run=args.run, watch_dirs=watch)
    if rc != 0:
        print(f"[!] One-DM training exited with code {rc}")
    return rc


def _lay_onedm_training_data(data: Path, data_root: Path) -> None:
    import cv2
    import numpy as np

    from textwritter.style import segment_words

    items = _read_zip_dataset(data, data_root.parent / "_tw_data")
    style_dir = data_root / "IAM64-new" / "train" / WRITER
    lap_dir = data_root / "IAM64_laplace" / "train" / WRITER
    for d in (style_dir, lap_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # labels.txt refers to cropped words; if an image is a line, segment it first,
    # otherwise use the crop as-is. Every sample is written as a 64 px-high word.
    lines = []
    idx = 0
    for src, text in items:
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        words = segment_words(gray) if min(gray.shape) > 90 else [gray]
        for w in words or [gray]:
            h, ww = w.shape
            if h < 2 or ww < 2:
                continue
            img = cv2.resize(w, (max(1, int(64 * ww / h)), 64), interpolation=cv2.INTER_AREA)
            name = f"w{idx:05d}"
            cv2.imwrite(str(style_dir / f"{name}.png"), img)
            lap = cv2.Laplacian(img.astype("float32"), cv2.CV_32F, ksize=3)
            lap = (np.abs(lap) / (np.abs(lap).max() or 1) * 255).astype("uint8")
            cv2.imwrite(str(lap_dir / f"{name}.png"), lap)
            lines.append(f"{WRITER},{name} {text.split()[0] if text.split() else 'x'}")
            idx += 1
    (data_root / "IAM64_train.txt").write_text("\n".join(lines) + "\n")
    print(f"[onedm] laid out {idx} training crops under {data_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _check(model: str) -> int:
    env = detect_environment()
    print(f"[env] {env.kind} | device={env.device} | gpu={env.gpu_name or 'none'}")
    print(f"[env] training allowed here: {env.can_train}")
    if model == "hwt":
        vendor = ENGINE_DIRS["hwt"] / "_vendor" / "HWT"
        weights = vendor / "files" / "iam_model.pth"
        print(f"[ckpt] HWT bundle: {'present' if weights.exists() else 'MISSING'} "
              f"({weights})")
        print("[bits] the bundle (~685 MB) is fetched automatically on first run; "
              "wandb is not required.")
        ready = weights.exists()
    else:
        ready = True
        for name in [PRETRAINED[model], "onedm_data"]:
            info = CHECKPOINTS[name]
            path = ROOT / "checkpoints" / (info["filename"] or info["subfolder"])
            ok = path.exists()
            ready &= ok
            print(f"[ckpt] {name}: {'present' if ok else 'MISSING'} ({info['note']})")
        print("[bits] One-DM fine-tune additionally needs vae_HTR138.pth from the "
              "official checkpoint folder.")
    return 0 if (ready and env.can_train) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["onedm", "hwt"], required=True)
    ap.add_argument("--data", type=Path, help="zip of images + labels.txt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5, help="low LR = adapt, don't forget")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--run", default=None, help="run name (default: <model>-finetune)")
    ap.add_argument("--check", action="store_true",
                    help="report prerequisites without training")
    args = ap.parse_args()
    args.run = args.run or f"{args.model}-finetune"

    if args.check:
        sys.exit(_check(args.model))

    env = detect_environment()
    print(f"[env] {env.kind} | device={env.device} | gpu={env.gpu_name or 'none'}")

    if not env.can_train:
        sys.exit(
            "\n[!] No CUDA GPU here — training refused.\n"
            "    Two options (the SAME command works on both):\n"
            "      • friend's GPU PC:  python training/finetune.py "
            f"--model {args.model} --data {args.data}\n"
            "      • Colab:            open notebooks/0"
            f"{'1' if args.model == 'onedm' else '2'}_* and Run-All\n"
            "    This machine can still create handwriting (CPU): "
            "python src/textwritter/quicktest.py --engine hwt\n")

    if args.data is None or not args.data.exists():
        sys.exit(f"[!] handwriting data zip not found: {args.data}\n"
                 "    Expected inside: cropped line/word images + labels.txt "
                 "(one 'file.png<TAB>text' per line).")

    # HWT ships as one bundle the engine fetches itself; only One-DM uses the
    # single-file checkpoint registry here.
    base = None
    if args.model == "onedm":
        base = get_checkpoint(PRETRAINED["onedm"])
        if base is None:
            sys.exit("[!] pretrained checkpoint unavailable — see instructions above.")
    resume = resume_source(env, args.run)
    print(f"[train] model={args.model} start_from={resume or base or 'official bundle'} "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch}")

    started = time.time()
    rc = train_hwt(args, env, base, resume) if args.model == "hwt" \
        else train_onedm(args, env, base, resume)
    print(f"[train] finished in {time.time() - started:.0f}s (exit {rc})")
    sys.exit(rc)


if __name__ == "__main__":
    main()
