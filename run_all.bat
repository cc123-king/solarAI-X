@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    exit /b 1
)

start "SolarAI-X MCP" cmd /k call "%~dp0run_mcp.bat"
timeout /t 3 /nobreak >nul
start "SolarAI-X API" cmd /k call "%~dp0run_api.bat"
timeout /t 3 /nobreak >nul
start "SolarAI-X UI" cmd /k call "%~dp0run_ui.bat"

echo Services are starting:
echo MCP: http://127.0.0.1:12346/sse
echo API docs: http://127.0.0.1:8001/docs
echo UI: http://127.0.0.1:5173
