"""Baseline schedulers, same signature as the live select_nodes()."""
from __future__ import annotations


def heuristic_top5_reliability(nodes: list[dict], job: dict) -> list[str]:
    """PLACEHOLDER for the live heuristic: sort by reliability (desc), take 5.
    Replace the body with the exact rule from scheduler/select.py before
    reporting any number (see SPEC.md item 8)."""
    ranked = sorted(nodes, key=lambda n: n["reliability"], reverse=True)
    return [n["node_id"] for n in ranked[:5]]


def all_nodes(nodes: list[dict], job: dict) -> list[str]:
    return [n["node_id"] for n in nodes]
