
"""Baseline schedulers matching the live server heuristic."""
from __future__ import annotations


def heuristic_select(nodes: list[dict], job: dict) -> list[str]:
    """Select every available node, most reliable and fastest first."""
    ranked = sorted(
        nodes,
        key=lambda n: (
            -n["reliability"],
            n["avg_round_s"]
            if n["avg_round_s"] is not None
            else float("inf"),
        ),
    )
    return [n["node_id"] for n in ranked]


def all_nodes(nodes: list[dict], job: dict) -> list[str]:
    """Select every available node in the original order."""
    return [n["node_id"] for n in nodes]
