@echo off
REM ============================================================
REM  Fine-tune on YOUR handwriting (GPU required)
REM  Needs: my_handwriting.zip in samples/ folder
REM ============================================================

title Text_Writter — Fine-Tune

call .venv\Scripts\activate.bat

echo.
echo  ========================================
echo   Text_Writter — Fine-Tune Your Style
echo  ========================================
echo.

REM Check if handwriting data exists
if not exist "samples\my_handwriting.zip" (
    echo  [!] No handwriting data found.
    echo.
    echo  HOW TO PREPARE YOUR HANDWRITING DATA:
    echo  =====================================
    echo.
    echo  1. Write 10-20 lines of text on plain white paper (black/blue pen)
    echo     Use different words, not the same sentence repeated.
    echo.
    echo  2. Photograph each line or word clearly (good light, no shadow)
    echo.
    echo  3. Save the photos as word_01.png, word_02.png, etc.
    echo.
    echo  4. Create a labels.txt file with one line per image:
    echo     word_01.png	the
    echo     word_02.png	quick
    echo     word_03.png	brown
    echo     (tab-separated: filename then the word written in that image)
    echo.
    echo  5. Zip everything into: samples\my_handwriting.zip
    echo     (images + labels.txt in the zip root)
    echo.
    echo  6. Run this script again.
    echo.
    pause
    exit /b 1
)

echo  Found: samples\my_handwriting.zip
echo.
echo  Select model to fine-tune:
echo    1. HWT  (faster, ~2-4 hours on RTX 4060)
echo    2. One-DM (slower, ~4-6 hours, best quality)
echo.
set /p MODEL_CHOICE="Enter 1 or 2: "

if "%MODEL_CHOICE%"=="1" (
    set MODEL=hwt
) else (
    set MODEL=onedm
)

echo.
echo  Starting fine-tune with model: %MODEL%
echo  This will take 2-6 hours. Checkpoints are saved every epoch.
echo  You can safely close and resume later (same command picks up where it left off).
echo.

python training/finetune.py --model %MODEL% --data samples/my_handwriting.zip

echo.
echo  Fine-tuning complete!
echo  Run run_generate.bat to test with your new model.
echo.
pause
