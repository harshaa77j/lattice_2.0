"""Baseline schedulers, same signature as the live select_nodes()."""
from __future__ import annotations


def heuristic_select(nodes: list[dict], job: dict) -> list[str]:
    """Match scheduler/select.py: reliability desc, avg round time asc."""
    ranked = sorted(
        nodes,
        key=lambda n: (
            -n["reliability"],
            n["avg_round_s"] if n["avg_round_s"] is not None else float("inf"),
        ),
    )
    return [n["node_id"] for n in ranked]


def all_nodes(nodes: list[dict], job: dict) -> list[str]:
    return [n["node_id"] for n in nodes]
