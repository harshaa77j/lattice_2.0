"""Live RL scheduler, called by scheduler/select.py when SCHEDULER=rl.

select_nodes(nodes, job) -> list[str]   node_ids in slice order
explain(nodes, job)      -> dict[node_id, float]   P(node is selected)

Uses the SAME encode()/decode() as the simulator (rl/features.py).
The model is loaded once, on CPU. Always returns a non-empty list of
distinct node_ids taken from `nodes`.
"""
from __future__ import annotations

import os

from .features import MAX_NODES, decode, encode

MODEL_PATH = os.environ.get(
    "LATTICE_RL_MODEL", os.path.join(os.path.dirname(__file__), "policy.zip"))
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        _MODEL = PPO.load(MODEL_PATH, device="cpu")
    return _MODEL


def _window(nodes: list[dict]) -> list[dict]:
    """The policy sees at most MAX_NODES nodes; if more are alive, keep the
    MAX_NODES most reliable (the rest are not selected this round)."""
    if len(nodes) <= MAX_NODES:
        return list(nodes)
    return sorted(nodes, key=lambda n: n.get("reliability", 0.0), reverse=True)[:MAX_NODES]


def _probs(nodes: list[dict], job: dict):
    import torch
    m = _model()
    obs = encode(nodes, job)
    t, _ = m.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = m.policy.get_distribution(t)
    return dist.distribution.probs[0].cpu().numpy()


def select_nodes(nodes: list[dict], job: dict) -> list[str]:
    win = _window(nodes)
    p = _probs(win, job)
    idx = decode((p[: len(win)] > 0.5).astype(int), len(win))
    idx.sort(key=lambda i: -p[i])               # highest score gets slice 0
    return [win[i]["node_id"] for i in idx]


def explain(nodes: list[dict], job: dict) -> dict[str, float]:
    win = _window(nodes)
    p = _probs(win, job)
    scores = {n["node_id"]: 0.0 for n in nodes}
    for i, n in enumerate(win):
        scores[n["node_id"]] = float(p[i])
    return scores
