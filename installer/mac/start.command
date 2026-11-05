#!/bin/bash
# Start volunteering for Lattice. Double-click this file to run it.
cd "$(dirname "$0")" || exit 1

if [ ! -x ".venv/bin/python" ]; then
  echo "Lattice is not set up on this Mac yet."
  echo "Double-click install.command first, then start.command again."
  read -n 1 -s -r -p "Press any key to close this window."
  echo
  exit 1
fi

echo "Opening the Lattice window..."
echo "You can minimise this Terminal window; closing the Lattice window stops volunteering."

# Keep the Mac awake while the app runs (it may still sleep if the lid is closed).
caffeinate -i .venv/bin/python -m node.app

echo "Lattice has stopped. You can close this window."