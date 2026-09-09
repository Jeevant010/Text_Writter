#!/usr/bin/env python3
"""Builds the two Colab notebooks (valid nbformat v4 JSON, guaranteed).

Run once:  python notebooks/_build_notebooks.py
Notebooks are for TRAINING (optional fine-tuning). Inference/testing lives in
experiments/*.py per project convention.
"""
import json
from pathlib import Path

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------------------
# Shared checkpoint-resume cell (Google Drive)
# ---------------------------------------------------------------------------
RESUME_CELL = '''\
# ============================================================
# CHECKPOINT RESUME — survives Colab tier switches
# ------------------------------------------------------------
# Every epoch, the trainer copies its latest checkpoint to YOUR
# Google Drive. If Colab disconnects / your free tier ends:
#   1. Open THIS notebook in a new Colab session (any tier).
#   2. Run all cells top-to-bottom again.
#   3. The trainer detects the Drive checkpoint and RESUMES
#      from the last epoch instead of starting over.
# Nothing is ever stored only on Colab's throwaway disk.
# ============================================================
from google.colab import drive
import shutil, pathlib

drive.mount('/content/drive')
DRIVE_DIR = pathlib.Path('/content/drive/MyDrive/textwritter_checkpoints') / MODEL_NAME
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CKPT = pathlib.Path('checkpoints')
LOCAL_CKPT.mkdir(exist_ok=True)

def resume_from_drive():
    """Copy newest Drive checkpoint back to local disk. Returns path or None."""
    ckpts = sorted(DRIVE_DIR.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    if ckpts:
        dst = LOCAL_CKPT / ckpts[-1].name
        shutil.copy2(ckpts[-1], dst)
        print(f'[resume] found {ckpts[-1].name} on Drive -> resuming')
        return dst
    print('[resume] no Drive checkpoint -> starting from pretrained base')
    return None

def save_to_drive(src: pathlib.Path):
    shutil.copy2(src, DRIVE_DIR / src.name)
    print(f'[checkpoint] {src.name} backed up to Drive')
'''

# ---------------------------------------------------------------------------
# Notebook 1: One-DM optional fine-tune
# ---------------------------------------------------------------------------
onedm = dict(NB)
onedm["cells"] = [
    md("# One-DM — OPTIONAL fine-tune (Colab, GPU)\n"
       "\n"
       "**Only run this if the pretrained One-DM fails on your handwriting.**\n"
       "One-DM is one-shot by design — try `experiments/01_onedm/test_inference.py`\n"
       "with the official checkpoint FIRST. 90% chance you never need this notebook.\n"
       "\n"
       "- Repo: https://github.com/dailenson/One-DM (MIT, ECCV 2024)\n"
       "- Base: official pretrained checkpoint (never train from scratch)\n"
       "- Checkpoints: auto-backed-up to Google Drive, auto-resumed on reconnect"),
    code("# 1. Verify GPU (Runtime -> Change runtime type -> T4 GPU)\n"
         "!nvidia-smi"),
    code("MODEL_NAME = 'onedm'\n"
         "REPO_URL = 'https://github.com/dailenson/One-DM'\n"
         "\n"
         "# 2. Clone repo + install\n"
         "!git clone --depth 1 $REPO_URL /content/One-DM\n"
         "%cd /content/One-DM\n"
         "!pip install -q -r requirements.txt gdown"),
    code(RESUME_CELL),
    md("## 3. Download the official PRETRAINED checkpoint\n"
       "\n"
       "Copy the Google Drive file id from the One-DM README (posted 2024-10-24)\n"
       "and paste it below. This is the base we fine-tune FROM — not from scratch."),
    code("PRETRAINED_GDRIVE_ID = 'PASTE_FROM_ONEDM_README'  # <-- fill this\n"
         "!gdown $PRETRAINED_GDRIVE_ID -O checkpoints/one_dm_pretrained.pth"),
    md("## 4. Your handwriting data\n"
       "\n"
       "Upload a zip of cropped line images + a `labels.txt` (one line per image:\n"
       "`filename.png<TAB>transcription`). 10–30 lines is plenty for adaptation.\n"
       "Format matches the IAM layout the repo's dataloader expects."),
    code("from google.colab import files\n"
         "import zipfile, pathlib\n"
         "\n"
         "uploaded = files.upload()  # pick your my_handwriting.zip\n"
         "DATA = pathlib.Path('/content/data')\n"
         "DATA.mkdir(exist_ok=True)\n"
         "for name in uploaded:\n"
         "    with zipfile.ZipFile(name) as z:\n"
         "        z.extractall(DATA)\n"
         "!ls $DATA | head"),
    md("## 5. Fine-tune (short: 2–6 h on a free T4)\n"
       "\n"
       "The training command below follows the repo's `train.py`. Key idea:\n"
       "`--resume` points at the Drive-restored checkpoint if one exists."),
    code("resume_ckpt = resume_from_drive()\n"
         "BASE = str(resume_ckpt or 'checkpoints/one_dm_pretrained.pth')\n"
         "\n"
         "# Low LR = adapt to your style without forgetting general handwriting\n"
         "!python train.py \\\n"
         "    --cfg configs/IAM64.yml \\\n"
         "    --pretrain $BASE \\\n"
         "    --lr 1e-5 \\\n"
         "    --batch_size 8 \\\n"
         "    --epochs 20 \\\n"
         "    --save_dir checkpoints \\\n"
         "    2>&1 | tail -20\n"
         "\n"
         "# Back up whatever the trainer produced\n"
         "for ckpt in sorted(pathlib.Path('checkpoints').glob('*.pth')):\n"
         "    save_to_drive(ckpt)"),
    md("## 6. Done — use it\n"
       "\n"
       "Download the newest `.pth` from Drive (`MyDrive/textwritter_checkpoints/onedm/`)\n"
       "and pass it to `experiments/01_onedm/test_inference.py --checkpoint ...`.\n"
       "\n"
       "**If Colab cut you off mid-run:** reopen notebook, Run-All — it resumes\n"
       "from the last Drive checkpoint automatically."),
]
OUT = Path(__file__).resolve().parent
(OUT / "01_onedm_finetune_colab.ipynb").write_text(
    json.dumps(onedm, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Notebook 2: HWT optional fine-tune
# ---------------------------------------------------------------------------
hwt = dict(NB)
hwt["cells"] = [
    md("# HWT — OPTIONAL fine-tune (Colab, GPU)\n"
       "\n"
       "**Only if pretrained HWT underfits your style.** Try the official custom-\n"
       "handwriting demo first: github.com/ankanbhunia/Handwriting-Transformers\n"
       "(it has a ready-made `demo_custom_handwriting.ipynb` — often all you need).\n"
       "\n"
       "- Repo: https://github.com/ankanbhunia/Handwriting-Transformers (ICCV 2021)\n"
       "- Base: official iam_model.pth (never from scratch)\n"
       "- Checkpoints: Drive-backed-up, auto-resumed"),
    code("!nvidia-smi"),
    code("MODEL_NAME = 'hwt'\n"
         "\n"
         "# 2. Clone + official pretrained bundle (models + IAM pickle)\n"
         "!git clone --depth 1 https://github.com/ankanbhunia/Handwriting-Transformers /content/HWT\n"
         "%cd /content/HWT\n"
         "!pip install -q --upgrade --no-cache-dir gdown\n"
         "!gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k && unzip -o files.zip && rm files.zip"),
    code(RESUME_CELL),
    md("## 3. Your handwriting data\n"
       "\n"
       "HWT custom training expects a pickle of `{writer: [{img, label}, ...]}`\n"
       "(see repo INSTALL.md). Upload word-crops + labels as described there;\n"
       "this cell converts a simple zip into that pickle."),
    code("from google.colab import files\n"
         "uploaded = files.upload()  # my_words.zip with images + labels.txt\n"
         "\n"
         "import zipfile, pathlib, pickle\n"
         "from PIL import Image\n"
         "DATA = pathlib.Path('/content/data'); DATA.mkdir(exist_ok=True)\n"
         "for name in uploaded:\n"
         "    with zipfile.ZipFile(name) as z: z.extractall(DATA)\n"
         "\n"
         "labels = {}\n"
         "for line in (DATA / 'labels.txt').read_text().splitlines():\n"
         "    fn, txt = line.split('\\t')\n"
         "    labels[fn.strip()] = txt.strip()\n"
         "\n"
         "samples = [{'img': Image.open(DATA / fn).convert('RGB'), 'label': txt}\n"
         "           for fn, txt in labels.items()]\n"
         "bundle = {'train': [{'me': samples}], 'test': [{'me': samples[:5]}]}\n"
         "with open('files/custom.pickle', 'wb') as f:\n"
         "    pickle.dump(bundle, f)\n"
         "print(f'{len(samples)} word samples packed -> files/custom.pickle')"),
    md("## 4. Fine-tune from iam_model.pth (2–4 h on T4)"),
    code("resume_ckpt = resume_from_drive()\n"
         "BASE = str(resume_ckpt or 'files/iam_model.pth')\n"
         "\n"
         "!python train.py \\\n"
         "    --dataname custom \\\n"
         "    --modelname custom_model \\\n"
         "    --pretrained $BASE \\\n"
         "    --lr 1e-5 \\\n"
         "    --epochs 15 \\\n"
         "    2>&1 | tail -20\n"
         "\n"
         "import pathlib\n"
         "for ckpt in sorted(pathlib.Path('files').glob('custom_model*.pth')):\n"
         "    save_to_drive(ckpt)"),
    md("## 5. Use it\n"
       "\n"
       "Download `custom_model*.pth` from Drive, place in HWT `files/`, point the\n"
       "repo's inference at it. Colab cut you off? Run-All resumes from Drive."),
]
(OUT / "02_hwt_finetune_colab.ipynb").write_text(
    json.dumps(hwt, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Notebook 0: Quick test — REAL One-DM inference on Colab GPU (no training)
# ---------------------------------------------------------------------------
quick = dict(NB)
quick["cells"] = [
    md("# Quick Test — Your Handwriting via One-DM (Colab GPU, NO training)\n"
       "\n"
       "The fastest honest way to see your own handwriting synthesized:\n"
       "pretrained One-DM + your one photo. Nothing here trains anything.\n"
       "\n"
       "Steps: GPU check → clone official repo → pretrained checkpoint →\n"
       "upload your photo → preprocess → generate → TrOCR read-back judge → compare."),
    code("!nvidia-smi  # Runtime -> Change runtime type -> T4 GPU"),
    code("# Clone official One-DM (MIT) + deps\n"
         "!git clone --depth 1 https://github.com/dailenson/One-DM /content/One-DM\n"
         "%cd /content/One-DM\n"
         "!pip install -q -r requirements.txt gdown opencv-python"),
    md("## Pretrained checkpoint\n"
       "Copy the Google Drive file id from the One-DM README (posted 2024-10-24).\n"
       "**Never train from scratch** — this checkpoint is the whole point."),
    code("GDRIVE_ID = 'PASTE_FROM_ONEDM_README'  # <-- fill this\n"
         "!gdown $GDRIVE_ID -O one_dm_pretrained.pth\n"
         "# SD 1.5 VAE downloads automatically on first run (from Hugging Face)"),
    md("## Upload ONE photo of your handwriting\n"
       "One clean line on plain white paper, good light, no shadows."),
    code("from google.colab import files\n"
         "uploaded = files.upload()\n"
         "SAMPLE = list(uploaded.keys())[0]\n"
         "print('style sample:', SAMPLE)"),
    md("## Preprocess: deskew → binarize → crop → 64px height"),
    code("import cv2, numpy as np\n"
         "\n"
         "img = cv2.imread(SAMPLE, cv2.IMREAD_GRAYSCALE)\n"
         "bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]\n"
         "coords = np.column_stack(np.where(bw > 0))\n"
         "angle = cv2.minAreaRect(coords)[-1]\n"
         "angle = -(90 + angle) if angle < -45 else -angle\n"
         "h, w = bw.shape\n"
         "M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)\n"
         "bw = cv2.warpAffine(bw, M, (w, h), borderValue=0)\n"
         "ys, xs = np.where(bw > 0)\n"
         "crop = bw[ys.min():ys.max()+1, xs.min():xs.max()+1]\n"
         "crop = cv2.resize(crop, None, fx=64/crop.shape[0], fy=64/crop.shape[0],\n"
         "                  interpolation=cv2.INTER_AREA)\n"
         "cv2.imwrite('style_ref.png', 255 - crop)\n"
         "print('style_ref.png ready')"),
    md("## Generate (edit TARGET_TEXT)"),
    code("TARGET_TEXT = 'The quick brown fox jumps over the lazy dog'\n"
         "\n"
         "!python test.py \\\n"
         "    --cfg configs/IAM64.yml \\\n"
         "    --one_dm one_dm_pretrained.pth \\\n"
         "    --generate_type iv_u \\\n"
         "    --device cuda \\\n"
         "    --sampling_timesteps 50 \\\n"
         "    --sample_method ddim \\\n"
         "    --dir /content/outputs\n"
         "\n"
         "import glob\n"
         "print('generated:', glob.glob('/content/outputs/**/*.png', recursive=True)[:5])"),
    md("## TrOCR read-back judge (auto quality gate)\n"
       "If CER > 0.15 the line is illegible — re-roll with a different seed/steps."),
    code("from transformers import TrOCRProcessor, VisionEncoderDecoderModel\n"
         "from PIL import Image\n"
         "\n"
         "processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')\n"
         "judge = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')\n"
         "\n"
         "def read_back(path):\n"
         "    px = processor(Image.open(path).convert('RGB'), return_tensors='pt').pixel_values\n"
         "    return processor.batch_decode(judge.generate(px), skip_special_tokens=True)[0]\n"
         "\n"
         "def cer(a, b):\n"
         "    a, b = a.lower().strip(), b.lower().strip()\n"
         "    prev = list(range(len(b)+1))\n"
         "    for i, ca in enumerate(a, 1):\n"
         "        cur = [i]\n"
         "        for j, cb in enumerate(b, 1):\n"
         "            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb)))\n"
         "        prev = cur\n"
         "    return prev[-1] / max(1, len(a))\n"
         "\n"
         "gen = sorted(glob.glob('/content/outputs/**/*.png', recursive=True))[-1]\n"
         "recognized = read_back(gen)\n"
         "score = cer(TARGET_TEXT, recognized)\n"
         "print(f'expected  : {TARGET_TEXT}')\n"
         "print(f'recognized: {recognized}')\n"
         "print(f'CER {score:.3f} ->', 'PASS' if score <= 0.15 else 'FAIL, re-roll')"),
    md("## Compare: your original vs generated"),
    code("import matplotlib.pyplot as plt\n"
         "fig, axes = plt.subplots(2, 1, figsize=(12, 4))\n"
         "axes[0].imshow(Image.open(SAMPLE)); axes[0].set_title('Your sample')\n"
         "axes[1].imshow(Image.open(gen)); axes[1].set_title('Generated')\n"
         "for ax in axes: ax.axis('off')\n"
         "plt.tight_layout(); plt.show()"),
]
(OUT / "00_quick_test_colab.ipynb").write_text(
    json.dumps(quick, indent=1, ensure_ascii=False))

print("notebooks written:",
      [p.name for p in OUT.glob("*.ipynb")])
