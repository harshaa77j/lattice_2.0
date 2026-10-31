"""Node selection (DESIGN.md D10).
The server calls choose_nodes() at the start of every round. SCHEDULER=heuristic
(default) or SCHEDULER=rl picks the policy. If the RL policy is missing, raises,
or returns something invalid, the heuristic is used for that round instead, so
a misbehaving policy can never stop training.
Person B: the RL policy lives in rl/policy.py as
    select_nodes(nodes: list[dict], job: dict) -> list[str]
taking the same arguments as heuristic_select below.
"""
import logging
import os
log = logging.getLogger("lattice.scheduler")

def heuristic_select(nodes: list, job: dict) -> list:
    """Every available node, most reliable and fastest first.

    nodes: one dict per alive, available node, with node_id, cpu_count, device,
           reliability, session_age_s, seconds_since_seen, recent_drops, avg_round_s
    job:   name, model, target_accuracy, max_rounds, local_epochs, rounds_done
    """
    ranked = sorted(
        nodes,
        key=lambda n: (-n["reliability"],
                       n["avg_round_s"] if n["avg_round_s"] is not None else float("inf")),
    )
    return [n["node_id"] for n in ranked]

def choose_nodes(nodes: list, job: dict) -> tuple:
    """Returns (node_ids, "rl" or "heuristic")."""
    if os.getenv("SCHEDULER", "heuristic").strip().lower() == "rl":
        try:
            from rl.policy import select_nodes as rl_select   # Person B's module
            chosen = list(rl_select(nodes, job))
            valid = {n["node_id"] for n in nodes}
            if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= valid:
                raise ValueError(f"invalid selection {chosen!r}")
            return chosen, "rl"
        except Exception as e:
            log.warning("RL scheduler failed (%s); using the heuristic this round", e)
    return heuristic_select(nodes, job), "heuristic"