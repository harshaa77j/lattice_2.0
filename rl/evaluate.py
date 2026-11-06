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

from .baselines import all_nodes, heuristic_select
from .features import decode, encode
from .sim_env import LatticeSchedEnv


def run_episode(env, select_fn, seed):
    """Run one episode using the supplied node-selection function."""
    env.reset(seed=seed)

    while True:
        ids = select_fn(env.current_nodes(), env.current_job())
        _, _, done, _, info = env.step(env.ids_to_action(ids))

        if done:
            return info


def ppo_selector(model):
    """Create a node-selection function using the trained PPO model."""

    def fn(nodes, job):
        obs = encode(nodes, job)
        bits, _ = model.predict(obs, deterministic=True)

        selected_indices = decode(bits, min(len(nodes), 10))
        return [nodes[i]["node_id"] for i in selected_indices]

    return fn


def summarise(infos):
    """Calculate average performance metrics for a set of episodes."""
    return {
        "completion_pct": 100 * np.mean([i["completed"] for i in infos]),
        "dropouts_per_100_jobs": 100 * np.mean([i["dropouts"] for i in infos]),
        "time_s": np.mean([i["time"] for i in infos]),
        "rounds": np.mean([i["rounds"] for i in infos]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO against heuristic and all-node baselines."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Training seeds whose models should be evaluated.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes per seed and scheduler.",
    )

    parser.add_argument(
        "--models",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory containing the trained PPO models.",
    )

    parser.add_argument(
        "--out",
        default="rl_results.csv",
        help="Output CSV file for evaluation results.",
    )

    args = parser.parse_args()

    env = LatticeSchedEnv()

    # Store per-seed results for each scheduler.
    per_seed = {
        "ppo": [],
        "heuristic": [],
        "all_nodes": [],
    }

    for seed in args.seeds:
        model_path = os.path.join(args.models, f"ppo_seed{seed}")

        print(f"\nEvaluating training seed {seed}...")
        print(f"Loading model: {model_path}")

        model = PPO.load(model_path, device="cpu")

        schedulers = {
            "ppo": ppo_selector(model),
            "heuristic": heuristic_select,
            "all_nodes": all_nodes,
        }

        # Each scheduler receives the same evaluation episode seeds.
        eval_seeds = [
            100_000 * (seed + 1) + episode
            for episode in range(args.episodes)
        ]

        for name, select_fn in schedulers.items():
            print(f"  Running {name}...")

            episode_infos = [
                run_episode(env, select_fn, eval_seed)
                for eval_seed in eval_seeds
            ]

            per_seed[name].append(summarise(episode_infos))

    metrics = [
        "completion_pct",
        "dropouts_per_100_jobs",
        "time_s",
        "rounds",
    ]

    rows = []

    print("\n" + "=" * 110)
    print("EVALUATION RESULTS")
    print("=" * 110)

    print(
        f"{'scheduler':16s}"
        + "".join(f"{metric:>26s}" for metric in metrics)
    )

    for name, seed_results in per_seed.items():
        cells = []

        for metric in metrics:
            values = np.array(
                [result[metric] for result in seed_results]
            )

            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0

            cells.append(f"{mean:10.2f} +/- {std:6.2f}")

            rows.append([
                name,
                metric,
                mean,
                std,
                *values,
            ])

        print(
            f"{name:16s}"
            + "".join(f"{cell:>26s}" for cell in cells)
        )

    # Save the results to CSV.
    with open(args.out, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            ["scheduler", "metric", "mean", "std"]
            + [f"seed{seed}" for seed in args.seeds]
        )

        writer.writerows(rows)

    print("=" * 110)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
