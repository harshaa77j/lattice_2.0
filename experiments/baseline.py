import argparse
import csv
import socket
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.data import get_cifar10
from shared.models import DEFAULT_MODEL, get_spec
from shared.training import evaluate, train_one_epoch

RESULTS_DIR = Path("experiments/results")

def main():
    parser = argparse.ArgumentParser(description="Single-laptop baseline: train until target accuracy.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--target", type=float, default=0.85, help="stop at this test accuracy (fraction)")
    parser.add_argument("--max-epochs", type=int, default=30, help="schedule length and safety cap")
    args = parser.parse_args()

    spec = get_spec(args.model)
    torch.manual_seed(0)
    train_loader = DataLoader(get_cifar10(train=True, augment=spec.augment),
                              batch_size=spec.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(get_cifar10(train=False), batch_size=256, shuffle=False, num_workers=0)

    model = spec.make_model().to(args.device)
    optimizer = spec.make_optimizer(model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    out_path = RESULTS_DIR / f"baseline_{spec.name}_{host}_{args.device}.csv"
    print(f"model {spec.name}, device {args.device}, target {args.target:.0%}. Logging to {out_path}")

    start = time.perf_counter()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "test_accuracy", "elapsed_s"])

        for epoch in range(1, args.max_epochs + 1):
            lr = spec.lr_at((epoch - 1) / args.max_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = train_one_epoch(model, train_loader, optimizer, args.device)
            accuracy = evaluate(model, test_loader, args.device)
            elapsed = time.perf_counter() - start
            writer.writerow([epoch, round(lr, 5), round(loss, 4), round(accuracy, 4), round(elapsed, 1)])
            f.flush()
            print(f"epoch {epoch}: lr {lr:.4f}, loss {loss:.3f}, test {accuracy:.1%}, {elapsed / 60:.1f} min")
            if accuracy >= args.target:
                print(f"Reached {accuracy:.1%} at epoch {epoch}, after {elapsed / 60:.1f} min.")
                break
        else:
            print(f"Did not reach {args.target:.0%} within {args.max_epochs} epochs.")

if __name__ == "__main__":
    main()