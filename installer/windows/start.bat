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

rem Open the Lattice window. pythonw runs it without an extra black console window.
start "" ".venv\Scripts\pythonw.exe" -m node.app