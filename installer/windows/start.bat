@echo off
setlocal
cd /d "%~dp0"
title Lattice volunteer

if not exist ".venv\Scripts\python.exe" (
  echo Lattice is not set up on this computer yet.
  echo Double-click install.bat first, then start.bat again.
  pause
  exit /b 1
)

echo Lattice volunteer is starting.
echo Keep this window open while you volunteer. Your laptop trains only when a job needs it.
echo To stop volunteering, close this window or press Ctrl+C.
echo.
".venv\Scripts\python.exe" -m node.client

echo.
echo Lattice has stopped. You can close this window.
pause