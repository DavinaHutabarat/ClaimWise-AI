"""
ClaimWise AI — Comprehensive Test Suite (UCS, A*, Heuristic & Data Integrity)
=============================================================================
Role: Davina Olivia Yosefanny Hutabarat — QA, Evaluation & Ethics Lead
Execution: uv run pytest -v
"""

import pytest
from ucs_search import (
    CHECKPOINT_COST_DETAILS,
    GRAPH,
    HEURISTIC,
    SearchResult,
    astar_search,
    compare_searches,
    uniform_cost_search,
)


# ===========================================================================
# 1. Pengujian Fungsionalitas Baseline UCS
# ===========================================================================
def test_finds_a_path_to_disbursement():
    """Memastikan UCS berhasil menemukan rute valid dari Submitted ke Goal."""
    result = uniform_cost_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})
    assert result is not None
    path, cost = result.path, result.total_cost
    assert path[0] == "Submitted"
    assert path[-1] in {"Disbursement", "Rejected"}
    assert cost >= 0.0


def test_picks_the_cheapest_route_not_just_any_route():
    """Memastikan UCS selalu memilih rute berbiaya minimum (FastTrack = 9.0)."""
    expected_path = [
        "Submitted",
        "DocCheck",
        "PolicyValidation",
        "RiskScoring",
        "FastTrack",
        "Disbursement",
    ]
    expected_cost = 9.0
    result = uniform_cost_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})
    assert result is not None
    assert result.path == expected_path
    assert result.total_cost == expected_cost


# ===========================================================================
# 2. Pengujian Algoritma A* Search & Kesetaraan Solusi
# ===========================================================================
def test_astar_finds_identical_optimal_path():
    """Memastikan A* menemukan jalur optimal yang persis sama dengan UCS."""
    res_ucs = uniform_cost_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})
    res_astar = astar_search(GRAPH, "Submitted", {"Disbursement", "Rejected"})

    assert res_ucs is not None
    assert res_astar is not None
    assert res_astar.path == res_ucs.path
    assert res_astar.total_cost == pytest.approx(res_ucs.total_cost, abs=1e-6)


def test_compare_searches_utility():
    """Memastikan helper perbandingan algoritma mengembalikan kedua hasil."""
    res_ucs, res_astar = compare_searches(
        GRAPH, "Submitted", {"Disbursement", "Rejected"}
    )
    assert isinstance(res_ucs, SearchResult)
    assert isinstance(res_astar, SearchResult)
    assert res_ucs.total_cost == res_astar.total_cost


# ===========================================================================
# 3. Pembuktian Matematis Sifat Heuristik (Admissibility & Consistency)
# ===========================================================================
def _compute_true_optimal_costs(graph, goals):
    """Menghitung biaya optimal sejati h*(s) dari setiap state ke target goals menggunakan UCS."""
    h_star = {}
    for state in graph:
        res = uniform_cost_search(graph, state, goals)
        h_star[state] = res.total_cost if res is not None else float("inf")
    return h_star


def test_heuristic_is_admissible():
    """
    UJI ADMISIBILITAS:
    h(s) <= h*(s) untuk semua state s dalam graf.
    Heuristik tidak boleh pernah melebih-lebihkan (overestimate) biaya riil ke goal.
    """
    goals = {"Disbursement", "Rejected"}
    h_star = _compute_true_optimal_costs(GRAPH, goals)

    for state, h_val in HEURISTIC.items():
        true_min_cost = h_star[state]
        assert h_val <= true_min_cost, (
            f"Pelanggaran admisibilitas pada state '{state}': "
            f"h({state})={h_val} > h*({state})={true_min_cost}"
        )


def test_heuristic_is_consistent_monotonic():
    """
    UJI KONSISTENSI (MONOTONIK):
    h(u) <= c(u, v) + h(v) untuk setiap transisi berarah (u, v) dengan bobot c(u, v).
    Menjamin bahwa evaluasi f(n) bersifat non-decreasing sepanjang jalur.
    """
    for u, transitions in GRAPH.items():
        h_u = HEURISTIC.get(u, 0.0)
        for v, cost in transitions:
            h_v = HEURISTIC.get(v, 0.0)
            assert h_u <= cost + h_v + 1e-9, (
                f"Pelanggaran konsistensi pada transisi '{u}' -> '{v}': "
                f"h({u})={h_u} > c({cost}) + h({v})={cost + h_v}"
            )


def test_heuristic_goal_states_are_zero():
    """Nilai heuristik untuk goal states harus bernilai tepat 0.0."""
    assert HEURISTIC["Disbursement"] == 0.0
    assert HEURISTIC["Rejected"] == 0.0


# ===========================================================================
# 4. Integritas Data & Validasi Graf Bisnis
# ===========================================================================
def test_all_edge_weights_are_strictly_positive():
    """Semua bobot edge dalam graf bisnis harus non-negatif (c(u, v) > 0)."""
    for state, neighbors in GRAPH.items():
        for neighbor, weight in neighbors:
            assert weight > 0.0, f"Bobot edge ({state} -> {neighbor}) harus positif!"


def test_goal_state_has_no_outgoing_edges():
    """State terminal (Disbursement, Rejected) tidak boleh memiliki transisi keluar."""
    assert GRAPH["Disbursement"] == []
    assert GRAPH["Rejected"] == []


def test_cost_breakdown_matches_graph_weights():
    """Memastikan bobot graf sinkron dengan tabel metrik justifikasi biaya riil."""
    for (u, v), meta in CHECKPOINT_COST_DETAILS.items():
        matching_edges = [w for target, w in GRAPH[u] if target == v]
        assert len(matching_edges) == 1
        assert matching_edges[0] == pytest.approx(meta["weight"], abs=1e-6)


# ===========================================================================
# 5. Robustness, Edge Cases, & Proteksi Galat
# ===========================================================================
def test_unreachable_goal_returns_none():
    """Jika target goal tidak ada atau tidak terjangkau, fungsi mengembalikan None."""
    assert uniform_cost_search(GRAPH, "Submitted", {"NonExistentState"}) is None
    assert astar_search(GRAPH, "Submitted", {"NonExistentState"}) is None


def test_target_specific_fraud_rejection():
    """Pencarian jalur khusus yang berakhir pada penolakan klaim (Rejected)."""
    result = uniform_cost_search(GRAPH, "Submitted", {"Rejected"})
    assert result is not None
    assert result.path[-1] == "Rejected"
    assert "FraudInvestigation" in result.path
    assert result.total_cost == 16.0  # 1+2+2+9+2 = 16.0


def test_cycle_resistance_explored_set():
    """Memastikan algoritma tidak mengalami infinite loop pada graf siklik."""
    cyclic_graph = {
        "A": [("B", 1.0)],
        "B": [("A", 1.0), ("C", 5.0)],
        "C": [],
    }
    res_ucs = uniform_cost_search(cyclic_graph, "A", {"C"})
    assert res_ucs is not None
    assert res_ucs.path == ["A", "B", "C"]
    assert res_ucs.total_cost == 6.0


def test_negative_edge_weight_raises_error():
    """Memastikan algoritma menolak graf dengan bobot negatif untuk mencegah galat Dijkstra."""
    invalid_graph = {
        "A": [("B", -2.0)],
        "B": [],
    }
    with pytest.raises(ValueError, match="Bobot edge tidak boleh negatif"):
        uniform_cost_search(invalid_graph, "A", {"B"})

    with pytest.raises(ValueError, match="Bobot edge tidak boleh negatif"):
        astar_search(invalid_graph, "A", {"B"})

