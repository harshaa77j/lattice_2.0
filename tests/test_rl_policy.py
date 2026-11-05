import os
import random
import time

import pytest

from rl import policy
from rl.features import MAX_NODES, OBS_DIM, encode

pytestmark = pytest.mark.skipif(
    not os.path.exists(policy.MODEL_PATH), reason="rl/policy.zip not present")

JOB = dict(name="t", model="cnn_small", target_accuracy=0.8, max_rounds=10,
           local_epochs=1, rounds_done=2)


def make_nodes(n, avg_none=False):
    rnd = random.Random(n)
    return [dict(node_id=f"node{i}", cpu_count=rnd.choice([2, 4, 8]),
                 device=rnd.choice(["cpu", "mps", "cuda"]),
                 reliability=rnd.random(), session_age_s=rnd.random() * 5000,
                 seconds_since_seen=rnd.random() * 5, recent_drops=rnd.randint(0, 10),
                 avg_round_s=None if avg_none else rnd.random() * 120)
            for i in range(n)]


@pytest.mark.parametrize("n", [1, 3, 10, 14])
def test_valid_output(n):
    nodes = make_nodes(n)
    out = policy.select_nodes(nodes, JOB)
    ids = {x["node_id"] for x in nodes}
    assert out and len(out) == len(set(out)) and set(out) <= ids


def test_avg_round_none():
    out = policy.select_nodes(make_nodes(5, avg_none=True), JOB)
    assert out


def test_explain_covers_all_nodes():
    nodes = make_nodes(12)
    sc = policy.explain(nodes, JOB)
    assert set(sc) == {n["node_id"] for n in nodes}
    assert all(0.0 <= v <= 1.0 for v in sc.values())


def test_latency_under_100ms():
    nodes = make_nodes(10)
    policy.select_nodes(nodes, JOB)             # warm-up / load
    t = time.perf_counter()
    for _ in range(20):
        policy.select_nodes(nodes, JOB)
    assert (time.perf_counter() - t) / 20 < 0.1


def test_encode_shape_and_range():
    o = encode(make_nodes(10), JOB)
    assert o.shape == (OBS_DIM,) and o.min() >= 0 and o.max() <= 1
    assert MAX_NODES == 10
