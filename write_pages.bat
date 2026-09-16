@echo off
REM CPU notebook pages (Caveat + math). Not handwriting cloning.
title Text_Writter — assignment_engine
call .venv\Scripts\activate.bat
if "%~1"=="" (
  echo Usage: write_pages.bat notes.txt [out.pdf]
  echo Example: write_pages.bat samples\assignment2_solutions.txt output\a2\out.pdf
  pause
  exit /b 1
)
set IN=%~1
set OUT=%~2
if "%OUT%"=="" set OUT=output\assignment.pdf
set PYTHONPATH=src
python src\textwritter\assignment_engine.py "%IN%" "%OUT%" --style caveat --ink blue
pause
