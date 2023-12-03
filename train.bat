@echo off
set TEST=False
set WANDB_SILENT=True
call .venv\Scripts\activate
python run_train.py
if %ERRORLEVEL% neq 0 pause
