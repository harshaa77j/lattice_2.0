#!/bin/bash
# Lattice volunteer setup for macOS. Double-click this file to run it.
cd "$(dirname "$0")" || exit 1

finish() { echo; read -n 1 -s -r -p "Press any key to close this window."; echo; exit "$1"; }
failed() {
  echo
  echo "Setup did not finish. Read the message above, fix the problem, and run install.command again."
  echo "It is safe to run it again: finished steps are skipped or repeated quickly."
  finish 1
}

echo "Lattice volunteer setup"
echo "======================="
echo

# --- 1. Find Python 3.10 or newer ------------------------------------------
# macOS's built-in python3 is usually 3.9, which is too old; python.org installers add python3.1x.
PY=""
for cmd in python3.10 python3.11 python3.12 python3.13 python3; do
  if command -v "$cmd" >/dev/null 2>&1 && \
     "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$cmd"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found on this Mac."
  echo "Your browser will open the Python download page. Install Python, then run install.command again."
  open "https://www.python.org/downloads/macos/"
  finish 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

# --- 2. A private environment just for Lattice -----------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating a private Python environment for Lattice..."
  "$PY" -m venv .venv || failed
fi
echo "Installing PyTorch and other packages. This downloads about 300 MB and takes a few minutes..."
.venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
.venv/bin/python -m pip install -r requirements-node.txt || failed

# --- 3. Check everything and download the training data once ---------------
.venv/bin/python -m node.prepare || failed

echo
echo "Setup complete. Double-click start.command whenever you want to volunteer."
finish 0