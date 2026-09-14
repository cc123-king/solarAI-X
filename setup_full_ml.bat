@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    exit /b 1
)

echo Installing optional AutoGluon and FlagEmbedding dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements-ml.txt
if errorlevel 1 exit /b 1

echo Full ML dependencies installed.
endlocal
