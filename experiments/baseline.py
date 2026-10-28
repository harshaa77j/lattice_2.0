import argparse
import csv
import socket
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.data import get_cifar10
from shared.model import SmallCNN
from shared.training import train_one_epoch, evaluate

BATCH_SIZE = 64
RESULTS_DIR = Path("experiments/results")

def main():
    parser = argparse.ArgumentParser(description="Single-laptop baseline: train until target accuracy.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--target", type=float, default=0.60, help="stop at this test accuracy (fraction)")
    parser.add_argument("--max-epochs", type=int, default=30, help="safety cap")
    args = parser.parse_args()

    torch.manual_seed(0)
    train_loader = DataLoader(get_cifar10(train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(get_cifar10(train=False), batch_size=256, shuffle=False)
    
    model = SmallCNN().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{socket.gethostname()}_{args.device}.csv"
    print(f"Logging to {out_path}")

    start = time.perf_counter()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy", "elapsed_s"])

        for epoch in range(1, args.max_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, args.device)
            accuracy = evaluate(model, test_loader, args.device)
            elapsed = time.perf_counter() - start
            writer.writerow([epoch, round(loss, 4), round(accuracy, 4), round(elapsed, 1)])
            f.flush()
            print(f"epoch {epoch}: loss {loss}, test accuracy {accuracy}, {elapsed/60:.1f} min")
            if accuracy >= args.target: 
                print(f"Reached {accuracy:.1%} at epoch {epoch}, after {elapsed / 60:.1f} min.")
                break
        else:
            print(f"Did not reach {args.target:.0%} within {args.max_epochs} epochs.")

if __name__ == "__main__":
    main()