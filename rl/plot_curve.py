"""Training curve: mean episode reward vs timesteps, across seeds.
    python -m rl.plot_curve --seeds 0 1 2 3 4 --out training_curve.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(__file__), "logs"))
    ap.add_argument("--out", default="training_curve.png")
    ap.add_argument("--window", type=int, default=200)
    a = ap.parse_args()
    plt.figure(figsize=(7, 4))
    for s in a.seeds:
        df = pd.read_csv(os.path.join(a.logs, f"monitor_seed{s}.monitor.csv"), skiprows=1)
        steps = np.cumsum(df["l"].values)
        plt.plot(steps, df["r"].rolling(a.window, min_periods=20).mean(), label=f"seed {s}", alpha=0.8)
    plt.xlabel("Timesteps (scheduling rounds)")
    plt.ylabel(f"Episode reward ({a.window}-episode rolling mean)")
    plt.title("PPO training on the Lattice scheduling simulator")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
