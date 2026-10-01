@echo off
title Stop TypeSafe Jev Trading Analyst
color 0C
cls
echo =====================================================================
echo  POWERING OFF: TypeSafe Jev + MT5 Quantitative Trading Analyst
echo =====================================================================
echo.

cd /d "%~dp0"

:: Use the dedicated python process manager
python stop_analyst.py

echo.
echo =====================================================================
echo  Application successfully powered off.
echo =====================================================================
ping 127.0.0.1 -n 3 >nul
