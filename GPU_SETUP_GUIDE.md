# GPU Setup Guide — Run Text_Writter on a Friend's RTX 4060

> **What this does:** Takes Text_Writter to any Windows PC with an NVIDIA GPU
> and generates genuinely realistic handwriting using One-DM (diffusion model).
> Everything is automated — one setup, then one-click generation.

---

## Prerequisites (Friend's PC)

| Requirement | How to Check | Install Link |
|---|---|---|
| **Windows 10/11** | — | — |
| **NVIDIA GPU** (RTX 4060 or similar) | Open Task Manager → Performance → GPU | — |
| **NVIDIA Drivers** (latest) | Run `nvidia-smi` in cmd | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| **Python 3.10 or 3.11** | Run `python --version` in cmd | [python.org/downloads](https://www.python.org/downloads/) |
| **Git** (optional, for cloning) | Run `git --version` | [git-scm.com](https://git-scm.com/download/win) |

> **IMPORTANT:** When installing Python, check **☑ "Add Python to PATH"** on the
> first screen of the installer. Without this, nothing works.

---

## Step-by-Step Instructions

### Step 1 — Transfer the Project

**Option A: USB Drive (Simplest)**
```
1. Copy out/Text_Writter_portable.zip to a USB drive
2. On friend's PC: extract the zip to Desktop (or anywhere)
```

**Option B: Git Clone (if friend has internet + git)**
```cmd
git clone https://github.com/Jeevant010/Text_Writter.git
cd Text_Writter
```

---

### Step 2 — Run Setup (Once)

Open **Command Prompt** (or PowerShell) in the Text_Writter folder:

```cmd
gpu_setup.bat
```

This automatically:
- Creates a Python virtual environment (`.venv`)
- Installs PyTorch **with CUDA 12.1 GPU support**
- Installs all project dependencies
- Verifies GPU detection

**Expected output at the end:**
```
  Environment: local_gpu
  Device: cuda
  GPU: NVIDIA GeForce RTX 4060
```

> Setup takes ~5-10 minutes (downloads ~3 GB of PyTorch + CUDA libraries).
> Only needed once — after this, everything runs instantly.

---

### Step 3 — Generate Handwriting

#### Option A: Quick CLI Test
```cmd
run_generate.bat
```
This runs `quicktest.py` with the default text. On an RTX 4060, it **auto-selects
One-DM** (the diffusion model that generates realistic handwriting). The output
opens automatically.

**Custom text:**
```cmd
run_generate.bat "Hello world, this is my handwriting" onedm
```

#### Option B: Web Studio (Browser UI)
```cmd
run_web.bat
```
Opens `http://localhost:8000` in your browser with the full interactive studio:
- Type any text
- Pick handwriting style, ink color, paper type
- Switch between Notebook mode (instant) and Neural mode (One-DM GPU)
- Download the generated page as PNG

---

### Step 4 — Use Your Own Handwriting (Optional)

To make it generate **in YOUR specific handwriting style**:

1. Write a few lines on plain white paper with a dark pen
2. Photograph straight-on, good lighting, no shadows
3. Save the photo as `samples/my_handwriting.jpg`

Then run:
```cmd
.venv\Scripts\activate
python src/textwritter/quicktest.py --style samples/my_handwriting.jpg --text "Your text here"
```

One-DM is **one-shot** — it learns your style from that single photo, no training
needed. The output will mimic your slant, pressure, and letter shapes.

---

### Step 5 — Fine-Tune for Best Results (Optional, 2-6 hours)

If the pretrained model doesn't capture your style well enough:

**Prepare your data:**
1. Write 10-20 different words/lines on white paper (black/blue pen)
2. Photograph each word clearly
3. Save as `word_01.png`, `word_02.png`, etc.
4. Create `labels.txt`:
   ```
   word_01.png	the
   word_02.png	quick
   word_03.png	brown
   ```
   (tab-separated: filename, then the word in that image)
5. Zip everything → `samples/my_handwriting.zip`

**Run fine-tuning:**
```cmd
run_finetune.bat
```

This takes **2-4 hours on RTX 4060** for HWT, or **4-6 hours** for One-DM.
Checkpoints are saved every epoch — you can stop and resume anytime.

---

## What Gets Downloaded (First Run Only)

| Asset | Size | Purpose |
|---|---|---|
| PyTorch + CUDA | ~3 GB | Deep learning framework with GPU support |
| HWT bundle | ~685 MB | Handwriting Transformer model weights + IAM data |
| One-DM checkpoint | ~1-2 GB | Diffusion model weights |
| One-DM English data | ~500 MB | Character conditions (unifont.pickle + word lists) |
| TrOCR judge | ~300 MB | Microsoft handwriting OCR for quality scoring |

Total first-run download: **~5-6 GB**. After that, everything is cached locally.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python, check "Add to PATH" |
| `nvidia-smi not found` | Install latest NVIDIA drivers |
| `CUDA not available` (PyTorch says False) | Make sure NVIDIA drivers are up-to-date; reinstall PyTorch: `.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |
| `gdown` fails on large files | Download the checkpoint manually from the Google Drive link in `docs/MODELS_GUIDE.md` and put it in `checkpoints/` |
| Out of GPU memory | Use `--steps 25` instead of default 50 (faster, slightly lower quality) |
| Fine-tune runs out of VRAM | RTX 4060 has 8 GB — enough for inference, may need `--batch-size 1` for training |

---

## File Map (What You'll See After Running)

```
Text_Writter/
├── gpu_setup.bat          ← Run this FIRST (one-time setup)
├── run_generate.bat       ← Generate handwriting (one click)
├── run_web.bat            ← Start web studio in browser
├── run_finetune.bat       ← Optional: fine-tune on your handwriting
├── samples/               ← Put your handwriting photos here
├── out/                   ← Generated images appear here
│   ├── quicktest_onedm.png    ← One-DM output (GPU, high quality)
│   └── quicktest_hwt.png      ← HWT output (CPU fallback)
└── checkpoints/           ← Model weights (auto-downloaded)
```
