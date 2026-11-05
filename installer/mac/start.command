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

echo "Lattice volunteer is starting."
echo "Keep this window open while you volunteer. Your Mac trains only when a job needs it."
echo "To stop volunteering, close this window or press Ctrl+C."
echo

# Keep the Mac awake while volunteering (it may still sleep if the lid is closed).
caffeinate -i .venv/bin/python -m node.client

echo
echo "Lattice has stopped. You can close this window."