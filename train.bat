@echo off
set TEST=False
call .venv\Scripts\activate
python app.py
if %ERRORLEVEL% neq 0 pause
