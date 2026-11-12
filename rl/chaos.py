"""Chaos mode for a volunteer (7.3). Settings are read from the environment on every call.

  LATTICE_CHAOS_DROP=0.3   probability that the volunteer quits mid-round
  LATTICE_CHAOS_SLOW=2.0   factor by which its rounds are slowed (2.0 = twice as long)
  LATTICE_CHAOS_REJOIN_S=20   how long it stays away after a chaos drop (default 20 s)

Intended use inside node/client.py (Person A integrates; shared/ server/ node/ are not mine):
    from rl import chaos
    ...
    if chaos.maybe_drop():            # at the start of a training round
        chaos.go_away()               # stop heartbeats, skip the upload, sleep, rejoin
        continue
    t0 = time.time(); train_one_epoch(...); chaos.slow_down(time.time() - t0)
"""
from __future__ import annotations

import os
import random
import time


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def drop_prob() -> float:
    return min(max(_f("LATTICE_CHAOS_DROP", 0.0), 0.0), 1.0)


def slow_factor() -> float:
    return max(_f("LATTICE_CHAOS_SLOW", 1.0), 1.0)


def rejoin_seconds() -> float:
    return max(_f("LATTICE_CHAOS_REJOIN_S", 20.0), 0.0)


def maybe_drop(rng: random.Random | None = None) -> bool:
    p = drop_prob()
    return p > 0 and (rng or random).random() < p


def slow_down(elapsed_s: float) -> float:
    """Sleep so the round takes slow_factor() times as long. Returns seconds slept."""
    extra = max(elapsed_s, 0.0) * (slow_factor() - 1.0)
    if extra > 0:
        time.sleep(extra)
    return extra


def go_away() -> float:
    """Stay silent for rejoin_seconds(). The caller must stop heartbeating meanwhile."""
    s = rejoin_seconds()
    time.sleep(s)
    return s