import argparse
import time

import torch
from torch.utils.data import DataLoader

from shared.data import get_cifar10, get_slice
from shared.models import DEFAULT_MODEL, get_spec
from shared.training import train_one_epoch

def time_one_epoch(spec, dataset, device):
    """Train a fresh model for one epoch on `dataset`. Returns seconds taken."""
    torch.manual_seed(0)
    model = spec.make_model().to(device)
    optimizer = spec.make_optimizer(model)
    loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=True, num_workers=0)
    start = time.perf_counter()
    train_one_epoch(model, loader, optimizer, device)
    return time.perf_counter() - start

def main():
    parser = argparse.ArgumentParser(description="Time one training epoch.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--slices", type=int, default=3, help="volunteers sharing the data")
    args = parser.parse_args()

    spec = get_spec(args.model)
    train = get_cifar10(train=True, augment=spec.augment)
    runs = [
        ("full dataset", train),
        (f"one slice of {args.slices}", get_slice(train, slice_id=0, num_slices=args.slices)),
    ]
    print(f"model {spec.name}, device {args.device}")
    for label, dataset in runs:
        seconds = time_one_epoch(spec, dataset, args.device)
        print(f"{label}: {len(dataset)} images in {seconds:.1f} s ({len(dataset) / seconds:.0f} images/s)")

if __name__ == "__main__":
    main()