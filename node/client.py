import logging
import os
import re
import socket
import sys
import threading
import time

import requests
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from pathlib import Path
from shared.datasets import ArrayDataset, unpack_arrays

from shared.data import get_cifar10, get_slice
from shared.models import get_spec
from shared.protocol import (
    NODE_ID_DISALLOWED, NODE_ID_MAX_LEN, NODE_TOKEN_HEADER, NOT_REGISTERED, is_valid_node_id,
)
from shared.serialization import bytes_to_state_dict, state_dict_to_bytes
from shared.training import train_one_epoch

REQUEST_TIMEOUT = 5          
TRANSFER_TIMEOUT = 120       
REGISTER_RETRY_DELAY = 3
HEARTBEAT_INTERVAL = 5
POLL_INTERVAL = 2

log = logging.getLogger("lattice.node")

class Rejected(Exception):
    """The server rejected registration (4xx): retrying won't help."""

def node_id_from_hostname() -> str:
    name = socket.gethostname().split(".")[0]
    name = re.sub(NODE_ID_DISALLOWED, "-", name)
    return name[:NODE_ID_MAX_LEN].strip("-_")

def fail(message: str):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

def pick_device() -> str:
    forced = os.getenv("LATTICE_DEVICE", "").strip().lower()
    if forced:
        if forced not in ("cpu", "mps", "cuda"):
            fail(f"LATTICE_DEVICE must be cpu, mps or cuda (got '{forced}').")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config() -> dict:
    load_dotenv()
    token = (os.getenv("LATTICE_TOKEN") or "").strip()
    if not token:
        fail("LATTICE_TOKEN is missing or empty. Set it in node/.env.")
    server = (os.getenv("LATTICE_SERVER") or "").strip().rstrip("/")
    if not server:
        fail("LATTICE_SERVER is missing or empty. Set it in node/.env.")
    if not server.startswith(("http://", "https://")):
        fail(f"LATTICE_SERVER must start with http:// or https:// (got '{server}').")
    node_id = (os.getenv("NODE_ID") or "").strip()
    if not node_id:
        node_id = node_id_from_hostname()
        log.warning("NODE_ID not set. Using hostname: '%s'", node_id)
    if not is_valid_node_id(node_id):
        fail(f"node id '{node_id}' is invalid. Use letters, digits, - and _ only "
             f"(max {NODE_ID_MAX_LEN} characters). Set NODE_ID in node/.env.")
    return {"server": server, "token": token, "node_id": node_id,
            "cpu_count": os.cpu_count() or 1, "device": pick_device()}

def make_session(config: dict) -> requests.Session:
    session = requests.Session()
    session.headers.update({NODE_TOKEN_HEADER: config["token"]})
    return session

def is_not_registered(response) -> bool:
    try:
        data = response.json()
        return isinstance(data, dict) and data.get("detail") == NOT_REGISTERED
    except ValueError:
        return False

def register(config: dict, session: requests.Session):
    url = f"{config['server']}/register"
    body = {"node_id": config["node_id"], "cpu_count": config["cpu_count"], "device": config["device"]}
    while True:
        try:
            r = session.post(url, json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                log.info("Registered as %s with %s (%s, %d cores)", config["node_id"],
                         config["server"], config["device"], config["cpu_count"])
                return
            if 400 <= r.status_code < 500:
                raise Rejected(f"Server rejected registration ({r.status_code}): {r.text}")
            log.warning("Server returned %s: %s", r.status_code, r.text)
        except requests.RequestException as e:
            log.warning("Network error while registering: %s", type(e).__name__)
        log.info("Retrying registration in %d seconds...", REGISTER_RETRY_DELAY)
        time.sleep(REGISTER_RETRY_DELAY)

def heartbeat_loop(config: dict):
    session = make_session(config)      # its own session: never shared between threads
    url = f"{config['server']}/heartbeat/{config['node_id']}"
    while True:
        try:
            r = session.post(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                log.debug("heartbeat ok")
            elif r.status_code == 404 and is_not_registered(r):
                log.info("Server doesn't know this node (restarted?); re-registering")
                try:
                    register(config, session)
                except Rejected as e:
                    log.error("Re-registration rejected: %s", e)
                    time.sleep(60)
            else:
                log.warning("Heartbeat got %s: %s", r.status_code, r.text)
        except requests.RequestException as e:
            log.warning("Heartbeat network error: %s", type(e).__name__)
        except Exception as e:
            log.exception("Heartbeat: unexpected error %s", type(e).__name__)
        time.sleep(HEARTBEAT_INTERVAL)

# --------------------------------------------------------------- training
SLICE_TIMEOUT = 900          # a slice can be ~50 MB on a slow connection
CACHE_DIR = Path("data") / "lattice"
_builtin: dict = {}          # the old CIFAR-10 path, per augmentation setting
_loaded: dict = {}           # (dataset, slice_id, num_slices) -> (images, labels) kept in memory


def builtin_cifar(augment: bool):
    """The original path: CIFAR-10 from torchvision. Used only for dataset 'cifar10'."""
    if augment not in _builtin:
        _builtin[augment] = get_cifar10(train=True, augment=augment)
    return _builtin[augment]


def stored_slice(config: dict, session: requests.Session, dataset: str, slice_id: int, num_slices: int):
    """This volunteer's share of a dataset stored on the server: downloaded once, then cached on disk."""
    key = (dataset, slice_id, num_slices)
    if key in _loaded:
        return _loaded[key]
    path = CACHE_DIR / f"{dataset}__{slice_id}of{num_slices}.npz"
    if path.is_file():
        log.info("Using this computer's saved share of %s", dataset)
    else:
        log.info("Downloading this computer's share of %s (only once for this job)...", dataset)
        start = time.perf_counter()
        r = session.get(f"{config['server']}/datasets/{dataset}/slice",
                        params={"slice_id": slice_id, "num_slices": num_slices}, timeout=SLICE_TIMEOUT)
        r.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old in CACHE_DIR.glob(f"{dataset}__*.npz"):    # a different share of the same dataset
            old.unlink(missing_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)                                  # complete files only, even if interrupted
        log.info("Downloaded %.1f MB in %.0f s", len(r.content) / 1e6, time.perf_counter() - start)
    _loaded.clear()                                        # keep at most one share in memory
    _loaded[key] = unpack_arrays(path.read_bytes())
    return _loaded[key]


def training_data(config: dict, session: requests.Session, task: dict, augment: bool):
    dataset = task.get("dataset", "cifar10")
    if dataset == "cifar10":
        return get_slice(builtin_cifar(augment), task["slice_id"], task["num_slices"])
    images, labels = stored_slice(config, session, dataset, task["slice_id"], task["num_slices"])
    return ArrayDataset(images, labels, augment=augment)


def run_task(config: dict, session: requests.Session, task: dict):
    spec = get_spec(task["model"])
    device = config["device"]
    rnd, slice_id, num_slices = task["round"], task["slice_id"], task["num_slices"]

    part = training_data(config, session, task, spec.augment)   # before the weights: may take a while

    r = session.get(f"{config['server']}/weights", timeout=TRANSFER_TIMEOUT)
    if r.status_code == 409:
        log.info("Round %d closed before training started; waiting for the next one", rnd)
        return
    r.raise_for_status()
    if r.headers.get("X-Lattice-Round") != str(rnd):
        log.info("Weights are for a different round; skipping round %d", rnd)
        return

    model = spec.make_model(task.get("num_classes", 10))
    model.load_state_dict(bytes_to_state_dict(r.content))
    model.to(device)
    progress = (rnd - 1) / task["max_rounds"] if task.get("max_rounds") else 0.0
    optimizer = spec.make_optimizer(model, progress)
    lr = optimizer.param_groups[0]["lr"]
    loader = DataLoader(part, batch_size=spec.batch_size, shuffle=True, num_workers=0)

    log.info("Round %d: training slice %d of %d (%d images, %s, %d epoch(s), lr %.4f)",
             rnd, slice_id + 1, num_slices, len(part), spec.name, task["local_epochs"], lr)
    start = time.perf_counter()
    for epoch in range(task["local_epochs"]):
        loss = train_one_epoch(model, loader, optimizer, device)
        log.info("Round %d: epoch %d loss %.3f", rnd, epoch + 1, loss)
    took = time.perf_counter() - start

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    up = session.post(
        f"{config['server']}/update/{config['node_id']}",
        params={"round": rnd, "n_samples": len(part)},
        data=state_dict_to_bytes(state),
        headers={"Content-Type": "application/octet-stream"},
        timeout=TRANSFER_TIMEOUT,
    )
    if up.status_code == 200:
        log.info("Round %d: trained in %.0f s, update accepted", rnd, took)
    elif up.status_code == 409:
        log.warning("Round %d: update not used (%s)", rnd, up.json().get("detail"))
    else:
        log.warning("Round %d: upload got %s: %s", rnd, up.status_code, up.text)

def work_loop(config: dict):
    session = make_session(config)
    url = f"{config['server']}/task/{config['node_id']}"
    waiting_logged = False
    while True:
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404 and is_not_registered(r):
                time.sleep(POLL_INTERVAL)          # the heartbeat thread re-registers
                continue
            r.raise_for_status()
            task = r.json()
            if task.get("status") == "train":
                waiting_logged = False
                run_task(config, session, task)
                continue                            # ask again right away
            if not waiting_logged:
                log.info("Waiting for work...")
                waiting_logged = True
        except requests.RequestException as e:
            log.warning("Network error: %s", type(e).__name__)
        except Exception:
            log.exception("Task failed; waiting for the next one")
        time.sleep(POLL_INTERVAL)

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    config = load_config()
    try:
        try:
            register(config, make_session(config))
        except Rejected as e:
            fail(str(e))
        threading.Thread(target=heartbeat_loop, args=(config,), name="heartbeat", daemon=True).start()
        work_loop(config)
    except KeyboardInterrupt:
        log.info("Stopping node client.")

if __name__ == "__main__":
    main()