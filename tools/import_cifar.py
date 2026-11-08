import argparse
import io
import os
import tempfile
import time

import numpy as np
import requests
from dotenv import load_dotenv
from torchvision.datasets import CIFAR10

from shared.protocol import ADMIN_KEY_HEADER


def main():
    load_dotenv("server/.env")
    p = argparse.ArgumentParser(description="Upload CIFAR-10 as a stored Lattice dataset.")
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--name", default="CIFAR-10")
    args = p.parse_args()
    key = os.getenv("LATTICE_ADMIN_KEY", "").strip()
    if not key:
        raise SystemExit("LATTICE_ADMIN_KEY is not set (server/.env or the environment).")

    train = CIFAR10("./data", train=True, download=False)
    test = CIFAR10("./data", train=False, download=False)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, train_images=np.asarray(train.data, np.uint8),
                            train_labels=np.asarray(train.targets, np.uint16),
                            test_images=np.asarray(test.data, np.uint8),
                            test_labels=np.asarray(test.targets, np.uint16))
        path = tmp.name
    size_mb = os.path.getsize(path) / 1e6
    print(f"Packed {len(train.targets)} training and {len(test.targets)} test images ({size_mb:.0f} MB). Uploading...")

    start = time.perf_counter()
    try:
        with open(path, "rb") as f:
            r = requests.post(f"{args.server.rstrip('/')}/admin/datasets/import",
                              params={"name": args.name, "classes": ",".join(train.classes)},
                              headers={ADMIN_KEY_HEADER: key, "Content-Type": "application/octet-stream"},
                              data=f, timeout=(10, 1800))
    finally:
        os.unlink(path)
    if r.status_code != 200:
        raise SystemExit(f"Import failed ({r.status_code}): {r.text}")
    meta = r.json()
    print(f"Done in {time.perf_counter() - start:.0f} s: dataset id '{meta['id']}', "
          f"{meta['num_classes']} classes, {meta['train_count']} train / {meta['test_count']} test")


if __name__ == "__main__":
    main()