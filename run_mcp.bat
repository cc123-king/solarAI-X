@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" api\mcpserver.py --host 127.0.0.1 --port 12346
