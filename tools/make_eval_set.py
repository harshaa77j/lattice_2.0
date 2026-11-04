"""One-time script: save the CIFAR-10 test set as a compact .npz file for the server.

    python -m tools.make_eval_set

Uses the copy already downloaded in ./data, so nothing is fetched from the internet.
"""
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

OUT = Path("server/eval/cifar10_test.npz")


def main():
    test = CIFAR10(root="./data", train=False, download=False)
    images = np.asarray(test.data, dtype=np.uint8)          # (10000, 32, 32, 3), raw pixels
    labels = np.asarray(test.targets, dtype=np.uint8)       # (10000,)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, images=images, labels=labels)
    print(f"saved {len(labels)} test images to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()