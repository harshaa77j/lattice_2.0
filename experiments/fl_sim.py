"""Single-machine FedAvg simulation for the paper (reviewer comment 4).

Simulates K clients on one machine, with IID or non-IID (Dirichlet label-skew)
partitions of CIFAR-10, and compares FedAvg with centralized training given the
same number of passes over the data. Clients train one after another on the same
device; only the data partitioning and averaging are simulated, not the network.

    caffeinate -i python -m experiments.fl_sim --device mps --rounds 20
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from server.aggregator import fedavg
from shared.data import get_cifar10, slice_data
from shared.models import get_spec
from shared.training import evaluate, train_one_epoch

OUT = Path("experiments/results/fl_sim.csv")


def dirichlet_partition(labels, k, alpha, seed):
    """Split indices among k clients; each class is divided by Dirichlet(alpha) proportions.
    Small alpha = strongly non-IID (each client sees few classes)."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(k))) * len(idx)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(idx, cuts)):
            parts[client].extend(chunk.tolist())
    return parts


def run(spec, train, test_loader, parts, rounds, device, label, seed, writer, f):
    torch.manual_seed(seed)
    global_model = spec.make_model()
    state = {k: v.clone() for k, v in global_model.state_dict().items()}
    start = time.perf_counter()
    best = 0.0
    for r in range(1, rounds + 1):
        updates = []
        for part in parts:
            if not part:
                continue                                    # a client with no data sits out
            model = spec.make_model()
            model.load_state_dict(state)
            model.to(device)
            opt = spec.make_optimizer(model, progress=(r - 1) / rounds)
            loader = DataLoader(Subset(train, part), batch_size=spec.batch_size, shuffle=True, num_workers=0)
            train_one_epoch(model, loader, opt, device)
            updates.append(({k: v.detach().cpu() for k, v in model.state_dict().items()}, len(part)))
        state = fedavg(updates)
        global_model.load_state_dict(state)
        acc = evaluate(global_model.to(device), test_loader, device)
        global_model.to("cpu")
        best = max(best, acc)
        elapsed = time.perf_counter() - start
        writer.writerow([label, seed, r, round(acc, 4), round(elapsed, 1)])
        f.flush()
        print(f"{label:<22} seed {seed} round {r:2d}: {acc:.1%}  ({elapsed / 60:.1f} min)")
    return best


def main():
    p = argparse.ArgumentParser(description="FedAvg simulation: clients x data distribution.")
    p.add_argument("--model", default="cnn_small")
    p.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, nargs="+", default=[2, 5, 10])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.1], help="Dirichlet alphas (non-IID)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()

    spec = get_spec(args.model)
    train = get_cifar10(train=True, augment=spec.augment)
    labels = np.asarray(train.targets)
    test_loader = DataLoader(get_cifar10(train=False), batch_size=256, shuffle=False, num_workers=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "seed", "round", "test_accuracy", "elapsed_s"])
        for seed in args.seeds:
            # Centralized reference: one "client" holding all data = plain training, same schedule.
            summary.append(("centralized", seed, run(spec, train, test_loader, [list(range(len(train)))],
                                                     args.rounds, args.device, "centralized", seed, w, f)))
            for k in args.clients:
                settings = [(f"K={k} IID", [slice_data(len(train), i, k) for i in range(k)])]
                settings += [(f"K={k} Dir({a})", dirichlet_partition(labels, k, a, seed)) for a in args.alphas]
                for label, parts in settings:
                    sizes = sorted(len(pt) for pt in parts)
                    print(f"\n{label}: client sizes min {sizes[0]}, max {sizes[-1]}")
                    summary.append((label, seed, run(spec, train, test_loader, parts, args.rounds,
                                                     args.device, label, seed, w, f)))
    print("\nBest test accuracy per setting:")
    for label, seed, best in summary:
        print(f"  {label:<22} seed {seed}: {best:.1%}")


if __name__ == "__main__":
    main()