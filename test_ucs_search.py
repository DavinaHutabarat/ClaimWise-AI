"""
Unit tests for ucs_search.py

Role: Davina Olivia — QA, Evaluation & Ethics Lead
Run with: uv run pytest -v
"""

from ucs_search import GRAPH, uniform_cost_search


def test_finds_a_path_to_disbursement():
    result = uniform_cost_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})
    assert result is not None
    path, cost = result
    assert path[0] == "Submitted"
    assert path[-1] in {"Disbursement", "Rejected"}
    assert cost >= 0


def test_picks_the_cheapest_route_not_just_any_route():
    # Manually computed cheapest path for this graph (see ucs_search.py docstring)
    expected_path = [
        "Submitted", "DocCheck", "PolicyValidation",
        "RiskScoring", "FastTrack", "Disbursement",
    ]
    expected_cost = 9.0
    path, cost = uniform_cost_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})
    assert path == expected_path
    assert cost == expected_cost


def test_unreachable_goal_returns_none():
    result = uniform_cost_search(GRAPH, "Submitted", {"NonExistentState"})
    assert result is None


def test_goal_state_has_no_outgoing_edges():
    # Sanity check on the graph definition itself (data integrity / QA)
    assert GRAPH["Disbursement"] == []
    assert GRAPH["Rejected"] == []
