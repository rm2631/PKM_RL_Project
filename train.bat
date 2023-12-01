@echo off
set TEST=False
set WANDB_SILENT=True
call .venv\Scripts\activate
python app.py
if %ERRORLEVEL% neq 0 pause
