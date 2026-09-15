#!/bin/bash
# ============================================================
#  Package Text_Writter for transfer to a GPU PC
#  Creates a clean zip without .venv, .git, __pycache__, etc.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="Text_Writter"
OUT_DIR="$SCRIPT_DIR/out"
ZIP_NAME="${REPO_NAME}_portable.zip"

echo ""
echo "  ========================================"
echo "   Packaging $REPO_NAME for GPU PC"
echo "  ========================================"
echo ""

cd "$SCRIPT_DIR"

# Create zip excluding heavy/unnecessary dirs
echo "[1/2] Creating portable zip..."
zip -r "$OUT_DIR/$ZIP_NAME" . \
    -x ".venv/*" \
    -x ".git/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*/*/__pycache__/*" \
    -x "*.pyc" \
    -x ".commandcode/*" \
    -x "experiments/02_hwt/_vendor/*" \
    -x "experiments/01_onedm/_vendor/*" \
    -x "checkpoints/*.pth" \
    -x "checkpoints/onedm_*/*" \
    -x "out/_*" \
    -x "out/notebook_*.png" \
    -x ".vscode/*"

SIZE=$(du -sh "$OUT_DIR/$ZIP_NAME" | cut -f1)
echo ""
echo "[2/2] Done!"
echo ""
echo "  File: $OUT_DIR/$ZIP_NAME"
echo "  Size: $SIZE"
echo ""
echo "  ========================================"
echo "   Transfer Instructions:"
echo "  ========================================"
echo ""
echo "  1. Copy '$ZIP_NAME' to the GPU PC (USB / cloud drive / etc.)"
echo "  2. Extract the zip to any folder (e.g., Desktop)"
echo "  3. Open a terminal (cmd / PowerShell) in that folder"
echo "  4. Run:  gpu_setup.bat"
echo "  5. Then: run_generate.bat"
echo ""
echo "  The setup script installs Python deps + CUDA PyTorch automatically."
echo "  Model weights (~1-2 GB) are downloaded on first run."
echo ""
