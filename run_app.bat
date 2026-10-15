@echo off
REM Double-click this file to start the AI Answer Evaluator app.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
