"""Decision records (7.2) and round traces (7.4). No torch import at module load.

Decision record (one per round, shown in the admin 'Scheduler decisions' panel):
  {ts, job, round, scheduler, candidates: [{node_id, score, chosen, order}]}
  candidates are sorted by score (high to low); order = slice index if chosen, else null.

Trace record (one JSON line per round, saved per job for later training on real data):
  {ts, job, round, scheduler, offered: [<full node dicts as given to the scheduler>],
   chosen: [ids in slice order], reported: [ids that uploaded], dropped: [ids],
   round_time_s, accuracy}
"""
from __future__ import annotations

import json
import os
import time


def build_record(round_no: int, job: dict, nodes: list[dict], scores: dict,
                 chosen_ids: list[str], scheduler: str = "rl") -> dict:
    order = {nid: i for i, nid in enumerate(chosen_ids)}
    cands = [{"node_id": n["node_id"],
              "score": round(float(scores.get(n["node_id"], 0.0)), 4),
              "chosen": n["node_id"] in order,
              "order": order.get(n["node_id"])} for n in nodes]
    cands.sort(key=lambda c: -c["score"])
    return {"ts": time.time(), "job": job.get("name"), "round": int(round_no),
            "scheduler": scheduler, "candidates": cands}


def decide(nodes: list[dict], job: dict, round_no: int = 0):
    """Run the RL policy once and return (chosen_ids, decision_record)."""
    from . import policy                       # lazy: needs torch + policy.zip
    ids = policy.select_nodes(nodes, job)
    scores = policy.explain(nodes, job)
    return ids, build_record(round_no, job, nodes, scores, ids, "rl")


def trace_record(round_no: int, job: dict, offered: list[dict], chosen_ids: list[str],
                 reported_ids: list[str], round_time_s: float,
                 scheduler: str, accuracy: float | None = None) -> dict:
    reported = set(reported_ids)
    return {"ts": time.time(), "job": job.get("name"), "round": int(round_no),
            "scheduler": scheduler, "offered": offered, "chosen": list(chosen_ids),
            "reported": list(reported_ids),
            "dropped": [i for i in chosen_ids if i not in reported],
            "round_time_s": float(round_time_s), "accuracy": accuracy}


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")