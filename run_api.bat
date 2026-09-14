@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" api\main.py
