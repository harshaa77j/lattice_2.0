"""Observation encoding shared by the simulator and the live policy.

The simulator produces node dicts with EXACTLY the keys the live server
passes to select_nodes() (Handover 7.1) and calls encode() on them.
rl/policy.py calls the same encode(), so train/serve parity holds by
construction.

Per-node features (PER_NODE = 11), slot order = order given by the server:
  0  present            1 if the slot holds a node, else 0 (zero padding)
  1  cpu_count          min(cpu_count, 32) / 32
  2  device == cpu      one-hot
  3  device == mps      one-hot
  4  device == cuda     one-hot
  5  reliability        already in [0, 1]
  6  session_age_s      min(s, 3600) / 3600
  7  seconds_since_seen min(s, 20) / 20
  8  recent_drops       min(n, 10) / 10
  9  avg_round_s        min(s, 300) / 300 (0 when unknown)
  10 avg_round_known    1 if avg_round_s is not None else 0
Job features (JOB_DIM = 4):
  rounds_done / max_rounds, min(local_epochs, 5) / 5,
  target_accuracy, min(max_rounds, 50) / 50
OBS_DIM = MAX_NODES * PER_NODE + JOB_DIM = 10 * 11 + 4 = 114 (float32)
"""
from __future__ import annotations

import numpy as np

MAX_NODES = 10
PER_NODE = 11
JOB_DIM = 4
OBS_DIM = MAX_NODES * PER_NODE + JOB_DIM
DEVICES = ("cpu", "mps", "cuda")


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def encode(nodes: list[dict], job: dict) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for i, n in enumerate(nodes[:MAX_NODES]):
        b = i * PER_NODE
        obs[b] = 1.0
        obs[b + 1] = min(float(n.get("cpu_count", 1)), 32.0) / 32.0
        dev = n.get("device", "cpu")
        if dev in DEVICES:
            obs[b + 2 + DEVICES.index(dev)] = 1.0
        obs[b + 5] = _clip01(float(n.get("reliability", 1.0)))
        obs[b + 6] = min(float(n.get("session_age_s", 0.0)), 3600.0) / 3600.0
        obs[b + 7] = min(float(n.get("seconds_since_seen", 0.0)), 20.0) / 20.0
        obs[b + 8] = min(float(n.get("recent_drops", 0)), 10.0) / 10.0
        avg = n.get("avg_round_s")
        if avg is not None:
            obs[b + 9] = min(float(avg), 300.0) / 300.0
            obs[b + 10] = 1.0
    j = MAX_NODES * PER_NODE
    max_rounds = max(float(job.get("max_rounds", 1)), 1.0)
    obs[j] = _clip01(float(job.get("rounds_done", 0)) / max_rounds)
    obs[j + 1] = min(float(job.get("local_epochs", 1)), 5.0) / 5.0
    obs[j + 2] = _clip01(float(job.get("target_accuracy", 0.8)))
    obs[j + 3] = min(max_rounds, 50.0) / 50.0
    return obs


def decode(bits, n_present: int) -> list[int]:
    """Action bits -> slot indices. Bits beyond n_present are ignored.
    An empty selection is invalid, so it falls back to ALL present nodes
    (the same rule is used in training and in rl/policy.py)."""
    chosen = [i for i in range(n_present) if int(bits[i]) == 1]
    return chosen if chosen else list(range(n_present))
