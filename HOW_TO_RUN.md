# HOW TO RUN — read this file first

Everything below is copy-paste. Open a terminal, paste the line, press Enter.

---

## STEP 0 — Do this once per terminal session

```bash
cd /home/jeevant/Desktop/Text_Writter
source .venv/bin/activate
```

After that, the command `python` means this project's Python. If you ever see
`ModuleNotFoundError`, you forgot this step — run those two lines again.

> Don't want to activate? Then replace `python` with `.venv/bin/python` in every
> command below. Same thing.

---

## STEP 1 — Create handwriting (the main thing)

```bash
python src/textwritter/quicktest.py
```

That's it. It writes the default sentence in handwriting and saves an image.

**What you will see:**

```
[env] LOCAL CPU (dev/testing) | device=cpu | gpu=none
[env] engine = hwt (...)
[style] using 15 bundled handwriting samples
[hwt] wrote out/quicktest_hwt.png
expected   : The quick brown fox jumps over the lazy dog
recognized : They quick brown fox jumps over the lazy dog
CER        : 0.116  (threshold 0.15)
verdict    : PASS
image      : out/quicktest_hwt.png
```

- `verdict: PASS` = the machine read back the image and it is legible.
- `image` = your handwriting picture. Open it:

```bash
xdg-open out/quicktest_hwt.png
```

There is also `out/quicktest_hwt_compare.png` — top is the style sample, bottom is
what was generated. Compare them with your eyes.

---

## STEP 2 — Use YOUR handwriting instead of the samples

1. Write one full line (or a short paragraph) on plain white paper. Black or blue pen.
2. Photograph it straight on, good light, no shadow, no lines/grid paper.
3. Save the photo into the `samples/` folder (any name, `.jpg` or `.png`).

Then:

```bash
python src/textwritter/quicktest.py --style samples/YOUR_PHOTO.jpg --text "Anything you want written"
```

If the photo is unclear it will tell you and use the bundled samples instead — it
never crashes.

---

## STEP 3 — Common options

```bash
# write your own text
python src/textwritter/quicktest.py --text "Hello, this is my handwriting"

# save to a specific file
python src/textwritter/quicktest.py --out out/mytest.png

# skip the read-back check (faster)
python src/textwritter/quicktest.py --no-judge

# force an engine
python src/textwritter/quicktest.py --engine hwt      # default here, CPU
python src/textwritter/quicktest.py --engine onedm    # needs NVIDIA GPU (this PC has none)
```

`--engine auto` (the default) picks One-DM on a GPU machine and HWT here. This laptop
has no NVIDIA GPU, so **use `hwt` on this machine.**

---

## STEP 4 — Judge any image by itself

```bash
python experiments/03_trocr_judge/judge.py --image out/quicktest_hwt.png --expect "The quick brown fox jumps over the lazy dog"
```

Prints what the machine "read" and the error rate. Low error = readable handwriting.

---

## STEP 5 — The notebook (same thing, nicer view)

```bash
python -m jupyter notebook
```

Then open `notebooks/00_quick_test.ipynb` in the browser and press **Run All**.
It works the same on this laptop and on Colab.

---

## Running on Colab or a GPU PC (for best quality)

On Colab: upload/open `notebooks/00_quick_test.ipynb` → Runtime → Change runtime type
→ **GPU** → Run All. The notebook clones this repo by itself. On a GPU machine you get
One-DM automatically (best fidelity).

Fine-tuning (the long GPU job) is optional and refuses to run on this CPU laptop with a
clear message. When you have a GPU:

```bash
python training/finetune.py --model hwt --data my_handwriting.zip
```

Same command on Colab and on a GPU PC. Only needed if the pretrained result looks wrong.

---

## Where things are

| Path | What |
|---|---|
| `out/quicktest_hwt.png` | the generated handwriting image |
| `out/quicktest_hwt_compare.png` | side-by-side: your style vs generated |
| `samples/` | put your handwriting photos here |
| `assets/sample_styles/` | bundled handwriting used when you pass no photo |
| `experiments/02_hwt/_vendor/` | downloaded model code + weights (ignore this) |
| `checkpoints/` | fine-tune checkpoints (created when you train) |

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | You skipped STEP 0. Run `source .venv/bin/activate`. |
| `command not found: python` | Use `.venv/bin/python` instead of `python`. |
| "No CUDA GPU here — training refused" | Correct. Training needs a GPU PC or Colab. Creation still works here. |
| "One-DM needs an NVIDIA CUDA GPU" | You passed `--engine onedm` on this PC. Use `--engine hwt`. |
| It says "using bundled style instead" | Your photo was unreadable or too small. Write a full clean line and retake the photo. |
| First run is slow | Only the very first time (downloads the model). After that it's cached. |

---

## The one-line cheat sheet

```bash
cd /home/jeevant/Desktop/Text_Writter && source .venv/bin/activate
python src/textwritter/quicktest.py --style samples/YOUR_PHOTO.jpg --text "your text here"
xdg-open out/quicktest_hwt.png
```
