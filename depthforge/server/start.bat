@echo off
rem Starts the DepthForge server and opens it in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo DepthForge is not set up yet. Run this first:
  echo   powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8765"
".venv\Scripts\python.exe" app.py
pause
