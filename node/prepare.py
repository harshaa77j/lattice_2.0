import sys
import shutil
import time

import requests

MIN_PYTHON = (3, 10)

def step(text):
    print(f"\n\u2192 {text}")


def ok(text):
    print(f"  \u2713 {text}")


def problem(text, fix):
    print(f"  \u2717 {text}\n    How to fix: {fix}")
    sys.exit(1)


def main():
    print("Lattice volunteer setup check")

    step("Python")
    v = sys.version_info
    if v < MIN_PYTHON:
        problem(f"Python {v.major}.{v.minor} is too old.", "install Python 3.10 or newer from python.org.")
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    step("PyTorch")
    try:
        import torch
        import torchvision
    except ImportError as e:
        problem(f"PyTorch is not installed ({e.name}).", "run the installer again; it installs PyTorch.")
    ok(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}")

    from node.client import load_config, make_session   # also validates node/.env
    from shared.protocol import NOT_REGISTERED

    config = load_config()
    ok(f"This machine will train on: {config['device']} ({config['cpu_count']} CPU cores)")

    step(f"Server {config['server']}")
    try:
        r = requests.get(f"{config['server']}/health", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        problem(f"The server can't be reached ({type(e).__name__}).",
                "check your internet connection, or ask the organiser whether the server is running.")
    ok("Server is reachable")

    # An unregistered node asking for work gets a 404 if the token is right, a 401 if it's wrong.
    r = make_session(config).get(f"{config['server']}/task/{config['node_id']}", timeout=10)
    if r.status_code == 401:
        problem("The server rejected this machine's token.", "download the volunteer package again.")
    if r.status_code not in (200, 404):
        problem(f"Unexpected answer from the server ({r.status_code}).", "try again in a minute.")
    if r.status_code == 404 and r.json().get("detail") != NOT_REGISTERED:
        problem("The server didn't recognise the request.", "download the volunteer package again.")
    ok(f"Token accepted; this machine will join as '{config['node_id']}'")

    step("Disk space for training data")
    free_gb = shutil.disk_usage(".").free / 1e9
    if free_gb < 1:
        problem(f"Only {free_gb:.1f} GB of disk space is free.", "free up at least 1 GB, then run the installer again.")
    ok(f"{free_gb:.0f} GB free. When a job starts, Lattice downloads only this computer's share of its data "
       "(usually 20\u201370 MB) from the Lattice server, and reuses it for the rest of the job.")


if __name__ == "__main__":
    main()