"""Gymnasium simulator of the live Lattice round loop.

One episode = one federated job (6-12 rounds). One step = one round:
the agent picks which of the currently available volunteers train.
ALL numeric assumptions are the constants / distributions below; they are
modelling assumptions, NOT calibrated against real traces (see SPEC.md).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .features import DEVICES, MAX_NODES, OBS_DIM, decode, encode

# ---- round / reward constants -------------------------------------------
OVERHEAD_S = 5.0     # fixed per-round overhead (download + upload)
DETECT_S = 20.0      # server declares a silent volunteer dropped after 20 s
DEADLINE_S = 180.0   # round deadline
REF_S = 90.0         # reference round time for the speed term
A_MAX = 0.80         # accuracy ceiling of the toy convergence model
GAIN = 0.30          # fraction of remaining gap closed by a full-data round
W_DROP = 0.3         # penalty per dropout
W_IDLE = 0.1         # penalty per reliable (>=0.9) node left out
W_DONE = 0.5         # bonus when the target accuracy is reached
W_FAIL = 1.0         # penalty when max_rounds ends without reaching target
REL_HI = 0.9
REJOIN_P = 0.5       # a node that drops sits out the next round with this prob

DEV_SPEED = {"cpu": 1.0, "mps": 0.5, "cuda": 0.3}   # multiplies epoch time
DEV_PROBS = [0.6, 0.2, 0.2]
CPU_CHOICES = [2, 4, 6, 8, 12, 16]


class LatticeSchedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.MultiBinary(MAX_NODES)
        self._pool: list[dict] = []
        self._order: list[int] = []
        self._ep = {}

    # ------------------------------------------------------------------ setup
    def _make_node(self, rng, idx: int) -> dict:
        device = str(rng.choice(DEVICES, p=DEV_PROBS))
        cpu = int(rng.choice(CPU_CHOICES))
        base = 240.0 * (4.0 / cpu) ** 0.5 * DEV_SPEED[device] * rng.lognormal(0, 0.25)
        flaky = rng.random() < 0.2
        p = rng.beta(4, 6) if flaky else rng.beta(1.2, 8)
        assigned = int(rng.integers(0, 31))
        hist = deque((bool(rng.random() < p) for _ in range(min(assigned, 10))), maxlen=10)
        completed = int(rng.binomial(assigned, 1 - p)) if assigned else 0
        avg = None if assigned == 0 else float((base / 3 + OVERHEAD_S) * rng.lognormal(0, 0.2))
        return dict(id=f"n{idx}", cpu=cpu, device=device, base=base, p=float(p),
                    assigned=assigned, completed=completed, hist=hist, avg=avg,
                    age=float(rng.exponential(1800.0)), absent=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        n = int(rng.integers(3, MAX_NODES + 1))
        self._pool = [self._make_node(rng, i) for i in range(n)]
        self._job = dict(name="sim", model=str(rng.choice(["cnn_small", "cnn_bn"])),
                         target_accuracy=float(rng.uniform(0.55, 0.70)),
                         max_rounds=int(rng.integers(6, 13)), local_epochs=1,
                         rounds_done=0)
        self._acc = 0.10
        self._ep = dict(dropouts=0, time=0.0, rounds=0, completed=False)
        return self._observe(), {}

    # ---------------------------------------------------------------- observe
    def _available(self):
        av = [x for x in self._pool if x["absent"] == 0]
        if not av:                       # never leave the server with nobody
            self._pool[0]["absent"] = 0
            av = [self._pool[0]]
        return av

    def _node_dict(self, x) -> dict:
        rel = 1.0 if x["assigned"] == 0 else x["completed"] / x["assigned"]
        return dict(node_id=x["id"], cpu_count=x["cpu"], device=x["device"],
                    reliability=float(rel), session_age_s=float(x["age"]),
                    seconds_since_seen=float(self.np_random.uniform(0, 5)),
                    recent_drops=int(sum(x["hist"])), avg_round_s=x["avg"])

    def _observe(self):
        av = self._available()
        perm = self.np_random.permutation(len(av))      # server order is arbitrary
        self._offered = [av[i] for i in perm]
        self._nodes = [self._node_dict(x) for x in self._offered]
        return encode(self._nodes, self._job)

    # public helpers for scheduler-driven evaluation
    def current_nodes(self) -> list[dict]:
        return self._nodes

    def current_job(self) -> dict:
        return dict(self._job)

    def ids_to_action(self, ids: list[str]) -> np.ndarray:
        a = np.zeros(MAX_NODES, dtype=np.int8)
        for i, n in enumerate(self._nodes[:MAX_NODES]):
            if n["node_id"] in ids:
                a[i] = 1
        return a

    # ------------------------------------------------------------------- step
    def step(self, action):
        rng = self.np_random
        n_off = len(self._offered)
        sel = decode(np.asarray(action), n_off)
        chosen = [self._offered[i] for i in sel]
        k = len(chosen)
        skipped = sum(1 for i, nd in enumerate(self._nodes)
                      if i not in sel and nd["reliability"] >= REL_HI)

        times, reported, dropped = [], [], []
        close = [OVERHEAD_S]
        for x in chosen:
            t = x["base"] / k * rng.lognormal(0, 0.1) + OVERHEAD_S
            if rng.random() < x["p"]:
                dropped.append(x)
                close.append(rng.uniform(0.1, 0.9) * t + DETECT_S)
            else:
                reported.append(x)
                times.append(t)
                close.append(t)
            x["_t"] = t
        round_time = float(min(DEADLINE_S, max(close)))

        if reported:
            self._acc += (A_MAX - self._acc) * GAIN * (len(reported) / k)

        for x in chosen:
            ok = x in reported
            x["assigned"] += 1
            x["completed"] += int(ok)
            x["hist"].append(not ok)
            if ok:
                x["avg"] = x["_t"] if x["avg"] is None else 0.7 * x["avg"] + 0.3 * x["_t"]
        for x in self._pool:
            x["age"] += round_time
            if x["absent"] > 0:
                x["absent"] -= 1
        for x in dropped:
            if rng.random() < REJOIN_P:
                x["absent"] = 1

        self._job["rounds_done"] += 1
        success = self._acc >= self._job["target_accuracy"]
        fail = (not success) and self._job["rounds_done"] >= self._job["max_rounds"]

        speed = float(np.clip(1.0 - round_time / REF_S, -1.0, 1.0))
        reward = (speed - W_DROP * len(dropped) - W_IDLE * skipped
                  + W_DONE * success - W_FAIL * fail)

        self._ep["dropouts"] += len(dropped)
        self._ep["time"] += round_time
        self._ep["rounds"] += 1
        self._ep["completed"] = bool(success)

        terminated = bool(success or fail)
        obs = self._observe() if not terminated else np.zeros(OBS_DIM, np.float32)
        info = dict(self._ep) if terminated else {}
        return obs, float(reward), terminated, False, info
