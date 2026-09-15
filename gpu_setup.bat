@echo off
REM ============================================================
REM  Text_Writter GPU Setup — Run this ONCE on the RTX 4060 PC
REM  This installs Python dependencies with CUDA GPU support
REM ============================================================

title Text_Writter GPU Setup

echo.
echo  ========================================================
echo   Text_Writter — GPU Setup for NVIDIA RTX 4060
echo  ========================================================
echo.

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo   Download Python 3.11 from: https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

python --version
echo.

REM Check NVIDIA GPU
echo [1/6] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] nvidia-smi not found. Make sure NVIDIA drivers are installed.
    echo   Download from: https://www.nvidia.com/Download/index.aspx
    echo.
    pause
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
echo.

REM Create virtual environment
echo [2/6] Creating Python virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists, reusing
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install PyTorch with CUDA 12.1 support (works with RTX 4060)
echo [4/6] Installing PyTorch with CUDA support (this may take a few minutes)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

REM Verify CUDA
echo.
echo [CHECK] Verifying CUDA availability...
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo.

REM Install remaining requirements
echo [5/6] Installing project dependencies...
pip install -r requirements.txt --quiet

REM Install additional utilities
pip install gdown --quiet

REM Test import
echo [6/6] Verifying installation...
python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from textwritter.runtime import detect_environment; env = detect_environment(); print(f'  Environment: {env.kind}'); print(f'  Device: {env.device}'); print(f'  GPU: {env.gpu_name}')"

echo.
echo  ========================================================
echo   Setup Complete!
echo  ========================================================
echo.
echo   Now run one of these:
echo.
echo   1. Quick test (generate handwriting):
echo      run_generate.bat
echo.
echo   2. Start web studio (browser UI):
echo      run_web.bat
echo.
echo   3. Fine-tune on your handwriting:
echo      run_finetune.bat
echo.
pause
