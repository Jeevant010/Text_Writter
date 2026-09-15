@echo off
REM ============================================================
REM  Generate Handwriting — One Click (GPU: One-DM, CPU: HWT)
REM ============================================================

title Text_Writter — Generate

call .venv\Scripts\activate.bat

echo.
echo  ========================================
echo   Text_Writter — Handwriting Generator
echo  ========================================
echo.

REM Default: auto-picks One-DM on GPU, HWT on CPU
set TEXT="The quick brown fox jumps over the lazy dog"
set ENGINE=auto

REM Check if user passed arguments
if not "%~1"=="" set TEXT=%1
if not "%~2"=="" set ENGINE=%2

echo  Text: %TEXT%
echo  Engine: %ENGINE%
echo.

python src/textwritter/quicktest.py --text %TEXT% --engine %ENGINE%

echo.
echo  Output saved. Opening image...
if exist out\quicktest_onedm.png (
    start out\quicktest_onedm.png
) else if exist out\quicktest_hwt.png (
    start out\quicktest_hwt.png
)

echo.
echo  ---
echo  Custom usage:
echo    run_generate.bat "Your text here" onedm
echo    run_generate.bat "Your text here" hwt
echo.
pause
