@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Checking Python...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 was not found. Please install Python 3.11 or 3.12 first.
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 exit /b 1

echo [3/4] Installing base dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-dev.txt
if errorlevel 1 exit /b 1

echo [4/4] Preparing local configuration...
if not exist ".env" copy ".env.example" ".env" >nul
if not exist "outputs" mkdir outputs
if not exist "models" mkdir models

echo.
echo Base setup completed.
echo Edit .env and add your own DEEPSEEK_API_KEY before using chat.
echo Run setup_full_ml.bat when AutoGluon prediction and RAG model support are needed.
endlocal
