@echo off
REM ============================================================
REM  Web Studio — Opens browser UI for handwriting generation
REM ============================================================

title Text_Writter — Web Studio

call .venv\Scripts\activate.bat

echo.
echo  ========================================
echo   Text_Writter — Web Studio Server
echo  ========================================
echo.
echo  Starting server on http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

REM Open browser after 2 second delay
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

set PYTHONPATH=src
python src/textwritter/web_server.py --host 127.0.0.1 --port 8000

pause
