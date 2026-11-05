"""Evaluate PPO (one model per seed) vs baselines on identical initial pools.
    python -m rl.evaluate --seeds 0 1 2 3 4 --episodes 200
Reports mean +/- std ACROSS SEEDS of the per-seed mean of each metric.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from .baselines import all_nodes, heuristic_top5_reliability
from .features import decode, encode
from .sim_env import LatticeSchedEnv


def run_episode(env, select_fn, seed):
    env.reset(seed=seed)
    while True:
        ids = select_fn(env.current_nodes(), env.current_job())
        _, _, done, _, info = env.step(env.ids_to_action(ids))
        if done:
            return info


def ppo_selector(model):
    def fn(nodes, job):
        obs = encode(nodes, job)
        bits, _ = model.predict(obs, deterministic=True)
        return [nodes[i]["node_id"] for i in decode(bits, min(len(nodes), 10))]
    return fn


def summarise(infos):
    return dict(
        completion_pct=100 * np.mean([i["completed"] for i in infos]),
        dropouts_per_100_jobs=100 * np.mean([i["dropouts"] for i in infos]),
        time_s=np.mean([i["time"] for i in infos]),
        rounds=np.mean([i["rounds"] for i in infos]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "models"))
    ap.add_argument("--out", default="rl_results.csv")
    a = ap.parse_args()

    env = LatticeSchedEnv()
    per_seed = {"ppo": [], "heuristic_top5": [], "all_nodes": []}
    for s in a.seeds:
        model = PPO.load(os.path.join(a.models, f"ppo_seed{s}"), device="cpu")
        fns = {"ppo": ppo_selector(model), "heuristic_top5": heuristic_top5_reliability,
               "all_nodes": all_nodes}
        eval_seeds = [100_000 * (s + 1) + e for e in range(a.episodes)]
        for name, fn in fns.items():
            per_seed[name].append(summarise([run_episode(env, fn, es) for es in eval_seeds]))

    metrics = ["completion_pct", "dropouts_per_100_jobs", "time_s", "rounds"]
    rows = []
    print(f"{'scheduler':16s}" + "".join(f"{m:>26s}" for m in metrics))
    for name, lst in per_seed.items():
        cells = []
        for m in metrics:
            v = np.array([d[m] for d in lst])
            cells.append(f"{v.mean():10.2f} +/- {v.std(ddof=1) if len(v) > 1 else 0:6.2f}")
            rows.append([name, m, v.mean(), v.std(ddof=1) if len(v) > 1 else 0, *v])
        print(f"{name:16s}" + "".join(f"{c:>26s}" for c in cells))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheduler", "metric", "mean", "std"] + [f"seed{s}" for s in a.seeds])
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
