"""Train PPO on the simulator. Usage:
    python -m rl.train_ppo --seed 0 --timesteps 300000
Writes rl/models/ppo_seed{seed}.zip and rl/logs/monitor_seed{seed}.monitor.csv
"""
from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .sim_env import LatticeSchedEnv

HYPERPARAMS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
)
POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                     activation_fn=torch.nn.Tanh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__)))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = Monitor(LatticeSchedEnv(),
                  os.path.join(a.out, "logs", f"monitor_seed{a.seed}"))
    model = PPO("MlpPolicy", env, policy_kwargs=POLICY_KWARGS, seed=a.seed,
                device="cpu", verbose=0, **HYPERPARAMS)
    model.learn(total_timesteps=a.timesteps)
    path = os.path.join(a.out, "models", f"ppo_seed{a.seed}")
    model.save(path)
    print("saved", path + ".zip")


if __name__ == "__main__":
    main()
