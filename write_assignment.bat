@echo off
REM HWT assignment pages (needs samples\my_handwriting.png).
REM For the CPU Caveat+math path used for CS743 A2, use write_pages.bat instead.
title Text_Writter — Assignment pages
call .venv\Scripts\activate.bat
if not exist "samples\my_handwriting.png" if not exist "samples\my_handwriting.jpg" (
  echo Put samples\my_handwriting.png first.
  echo Export a PDF page:
  echo   python src\textwritter\extract_style.py --pdf "C:\path\your.pdf" --page 1
  pause
  exit /b 1
)
set STYLE=samples\my_handwriting.png
if exist samples\my_handwriting.jpg set STYLE=samples\my_handwriting.jpg
if "%~1"=="" (
  echo Usage: write_assignment.bat notes.txt
  echo Put your assignment English in notes.txt then run this again.
  pause
  exit /b 1
)
python src\textwritter\write_assignment.py --engine hwt --style %STYLE% --text-file %1
pause
