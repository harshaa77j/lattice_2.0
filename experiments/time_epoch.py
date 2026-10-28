import argparse
import time

import torch
from torch.utils.data import DataLoader

from shared.data import get_cifar10, get_slice
from shared.model import SmallCNN
from shared.training import train_one_epoch

BATCH_SIZE = 64

def time_one_epoch(dataset, device):
    torch.manual_seed(0) 
    model = SmallCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    start = time.perf_counter()
    train_one_epoch(model, loader, optimizer, device)
    return time.perf_counter() - start

def main():
    parser = argparse.ArgumentParser(description="Time one training epoch.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    train = get_cifar10(train=True)
    runs = [
        ("full dataset", train),
        ("one-fifth slice", get_slice(train, slice_id=0, num_slices=5)),
    ]
    for label, dataset in runs:
        seconds = time_one_epoch(dataset, args.device)
        print(f"{label}: {len(dataset)} in {seconds:.1f} ({len(dataset)/seconds:.0f} images/sec)")

if __name__ == "__main__":
    main()