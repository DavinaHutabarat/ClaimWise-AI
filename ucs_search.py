"""
ClaimWise AI — Baseline Search Module
======================================

Milestone 1 deliverable: Uniform Cost Search (UCS) over the health-insurance
claim triage/verification graph.

Business framing
-----------------
Every incoming claim moves through a sequence of "checkpoints" (document
check, policy validation, risk scoring, medical review, fraud
investigation, approval, disbursement, ...). Each transition between
checkpoints has a real-world cost that combines:
    - staff handling cost (Rupiah, normalized)
    - expected processing time (days, normalized)

UCS finds the checkpoint path from "Submitted" to a goal state
("Disbursement" or "Rejected") with the LOWEST total cost, which is exactly
the "cheapest verification route" decision problem described in the
Problem Framing document.

Formal state-space model (X, A, T, G, C)
-----------------------------------------
X (States)      : claim-processing checkpoints (see GRAPH keys below)
A (Actions)     : routing decisions between checkpoints (see GRAPH edges)
T (Transition)  : deterministic — choosing an action moves the claim to
                   exactly one next checkpoint (graph is a directed graph)
G (Goal test)    : state in {"Disbursement", "Rejected"} (configurable)
C (Cost)         : edge weight = normalized (handling_cost + processing_time)

Author / Role: Jodi — AI Architect & Model Lead
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Problem definition: the claim-triage graph (X, A, C)
# ---------------------------------------------------------------------------
# GRAPH[state] = list of (next_state, cost)
# Cost unit = normalized (handling cost + processing time), higher = worse.
GRAPH: Dict[str, List[Tuple[str, float]]] = {
    "Submitted":          [("DocCheck", 1.0)],
    "DocCheck":           [("PolicyValidation", 2.0)],
    "PolicyValidation":   [("RiskScoring", 2.0)],
    "RiskScoring": [
        ("FastTrack", 3.0),          # klaim berisiko rendah
        ("StandardReview", 5.0),     # klaim berisiko sedang
        ("FraudInvestigation", 9.0), # klaim berisiko tinggi
    ],
    "FastTrack":          [("Disbursement", 1.0)],
    "StandardReview":     [("MedicalReview", 4.0)],
    "MedicalReview":      [("ApprovalOfficer", 3.0)],
    "ApprovalOfficer":    [("Disbursement", 1.0)],
    "FraudInvestigation": [
        ("ApprovalOfficer", 6.0),
        ("Rejected", 2.0),
    ],
    "Disbursement":       [],  # goal state, no outgoing edges
    "Rejected":           [],  # goal state, no outgoing edges
}


@dataclass(order=True)
class _Node:
    """Priority-queue entry. `sort_index` = path cost so heapq is a min-heap."""
    sort_index: float
    state: str = field(compare=False)
    path: List[str] = field(compare=False)


def uniform_cost_search(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: set,
) -> Optional[Tuple[List[str], float]]:
    """
    Uniform Cost Search using a binary heap (heapq).

    Returns (path, total_cost) for the cheapest path from `start` to any
    state in `goals`, or None if unreachable.

    Complexity: O(E log E) — standard UCS bound with a lazy-deletion heap.
    """
    counter = itertools.count()  # tie-breaker so heapq never compares dicts/lists
    frontier: List[Tuple[float, int, str, List[str]]] = []
    heapq.heappush(frontier, (0.0, next(counter), start, [start]))

    best_cost: Dict[str, float] = {start: 0.0}
    explored: set = set()

    while frontier:
        cost, _, state, path = heapq.heappop(frontier)

        if state in explored:
            continue
        explored.add(state)

        if state in goals:
            return path, cost

        for neighbor, edge_cost in graph.get(state, []):
            new_cost = cost + edge_cost
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                heapq.heappush(
                    frontier, (new_cost, next(counter), neighbor, path + [neighbor])
                )

    return None  # goal unreachable


def main() -> None:
    start_state = "Submitted"
    goal_states = {"Disbursement", "Rejected"}

    result = uniform_cost_search(GRAPH, start_state, goal_states)

    if result is None:
        print(f"Tidak ditemukan jalur dari '{start_state}' ke goal manapun.")
        return

    path, total_cost = result
    print("=== ClaimWise AI — Uniform Cost Search ===")
    print(f"Start : {start_state}")
    print(f"Goals : {goal_states}")
    print(f"Jalur optimal ditemukan:")
    print("  " + " -> ".join(path))
    print(f"Total biaya (handling + waktu, ternormalisasi): {total_cost:.2f}")


if __name__ == "__main__":
    main()
