@echo off
title TypeSafe Jev Quantitative Analyst - Cockpit Server
color 0A
cls
echo =====================================================================
echo  POWERING ON: TypeSafe Jev Quantitative Trading Cockpit
echo  Launching Interactive Web Cockpit on http://localhost:8000
echo =====================================================================
echo.

cd /d "%~dp0"

:: If an existing session is running, clean it up first
if exist "app.pid" python stop_analyst.py >nul 2>&1

:: Verify Python is available
python --version >nul 2>&1
if errorlevel 1 goto :no_python

echo Starting Real-Time Analysis Server ^& WebSockets...
echo Connecting to MetaTrader 5 Terminal...
echo.

echo =====================================================================
echo  Cockpit UI available at: http://localhost:8000
echo  (The browser will open automatically once live analysis is ready)
echo  Press Ctrl+C in this window or double-click stop.bat to power off.
echo =====================================================================
echo.

python server.py
goto :end

:no_python
echo [ERROR] Python was not found on your system PATH!
echo Please ensure Python 3 is installed and added to PATH.
pause
exit /b 1

:end
echo.
echo =====================================================================
echo  Application server stopped.
echo =====================================================================
pause
