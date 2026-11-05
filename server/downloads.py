"""Builds the volunteer download: one zip with everything a laptop needs to join.

The zip contains the volunteer code (node/ and shared/), requirements-node.txt,
the platform's install and start scripts, a README, and a node/.env with this
server's address and token already filled in. Nothing from the server side
(server/, frontend/, tests) is included.
"""
import io
import stat
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = "lattice-volunteer"                       # folder name inside the zip
PLATFORMS = {
    "windows": {"dir": "installer/windows", "scripts": ["install.bat", "start.bat"]},
    "mac": {"dir": "installer/mac", "scripts": ["install.command", "start.command"]},
}

README = {
"windows": """Lattice volunteer for Windows
=============================

1. Right-click the downloaded zip and choose "Extract All", then open the extracted folder.
2. Double-click install.bat. If Windows shows "Windows protected your PC",
   click "More info", then "Run anyway". Setup takes a few minutes the first time.
3. When it says "Setup complete", double-click start.bat to volunteer.
   Close that window (or press Ctrl+C) to stop at any time.
""",
"mac": """Lattice volunteer for macOS
===========================

1. Double-click the downloaded zip to unzip it, then open the folder.
2. The first time only: right-click install.command, choose "Open", then "Open" again.
   (macOS asks because the script is not from the App Store.) Setup takes a few minutes.
3. When it says "Setup complete", right-click start.command, choose "Open" (first time only).
   After that you can simply double-click it. Close the window to stop at any time.
""",
}

ABOUT = """
What Lattice does on your computer
----------------------------------
- While the start window is open, it waits for training work and uses your
  processor (or Apple GPU) to train on a share of a public image dataset (CIFAR-10).
- It never runs code sent by anyone: jobs only choose a model from a fixed library.
- It sends back only model weights, never your files.
- Everything lives in this folder. To remove Lattice, close the window and delete the folder.
"""


def _add_text(zf: zipfile.ZipFile, name: str, text: str, executable: bool = False, crlf: bool = False):
    data = text.replace("\r\n", "\n")
    if crlf:
        data = data.replace("\n", "\r\n")          # Windows batch files need CRLF line endings
    info = zipfile.ZipInfo(f"{TOP}/{name}")
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16  # keeps the executable bit when macOS unzips
    zf.writestr(info, data.encode("utf-8"))


def build_volunteer_zip(platform: str, server_url: str, token: str) -> bytes:
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform '{platform}'")
    spec = PLATFORMS[platform]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("shared", "node"):
            for path in sorted((REPO / folder).glob("*.py")):
                _add_text(zf, f"{folder}/{path.name}", path.read_text(encoding="utf-8"))
        _add_text(zf, "requirements-node.txt", (REPO / "requirements-node.txt").read_text(encoding="utf-8"))
        for script in spec["scripts"]:
            text = (REPO / spec["dir"] / script).read_text(encoding="utf-8")
            _add_text(zf, script, text, executable=(platform == "mac"), crlf=(platform == "windows"))
        _add_text(zf, "README.txt", README[platform] + ABOUT, crlf=(platform == "windows"))
        _add_text(zf, "node/.env",
                  "# Filled in by the Lattice server when you downloaded this package.\n"
                  f"LATTICE_SERVER={server_url.rstrip('/')}\n"
                  f"LATTICE_TOKEN={token}\n"
                  "# NODE_ID=my-laptop   (optional; defaults to this computer's name)\n")
    return buf.getvalue()