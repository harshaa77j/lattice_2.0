@echo off
setlocal
cd /d "%~dp0"
title Lattice volunteer setup
echo Lattice volunteer setup
echo =======================
echo.

rem --- 1. Find Python 3.10 or newer --------------------------------------
rem The "py" launcher comes with the python.org installer; plain "python" covers other installs.
set "PY="
for %%V in (3.10 3.11 3.12 3.13) do (
  if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10 or newer was not found on this computer.
  echo Your browser will open the Python download page.
  echo Install Python, tick "Add python.exe to PATH" in the installer, then run install.bat again.
  start "" "https://www.python.org/downloads/windows/"
  pause
  exit /b 1
)
echo Using Python: %PY%

rem --- 2. A private environment just for Lattice -------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment for Lattice...
  %PY% -m venv .venv || goto :failed
)
echo Installing PyTorch and other packages. This downloads about 300 MB and takes a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements-node.txt || goto :failed

rem --- 3. Check everything and download the training data once -----------
".venv\Scripts\python.exe" -m node.prepare || goto :failed

echo.
echo Setup complete. Double-click start.bat whenever you want to volunteer.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Read the message above, fix the problem, and run install.bat again.
echo It is safe to run it again: finished steps are skipped or repeated quickly.
pause
exit /b 1