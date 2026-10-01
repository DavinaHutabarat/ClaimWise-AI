"""
ClaimWise AI — Comprehensive Test Suite
========================================
Testing:
1. Risk Classification
2. Dynamic Claim Graph
3. Uniform Cost Search (UCS)
4. A* Search
5. Robustness & Edge Cases

Execution:
    uv run pytest -v
"""

import pytest

from ucs_search import (
    HEURISTIC,
    SearchResult,
    astar_search,
    build_dynamic_claim_graph,
    predict_claim_risk,
    uniform_cost_search,
)


# ===========================================================================
# 1. TEST MACHINE LEARNING / RISK CLASSIFICATION
# ===========================================================================

def test_low_risk_classification():
    """Klaim bernilai rendah tanpa anomali harus dikategorikan Low Risk."""
    claim = {
        "claim_amount_idr": 3_500_000,
        "has_billing_anomaly": False,
        "policy_active_months": 24,
        "is_frequent_claimer": False,
        "incomplete_docs": False,
    }

    result = predict_claim_risk(claim)

    assert result["risk_category"] == "Low Risk"
    assert result["risk_score"] == pytest.approx(0.10)
    assert result["has_anomaly"] is False
    assert result["incomplete_docs"] is False


def test_high_risk_classification():
    """Klaim dengan nominal tinggi dan beberapa indikator risiko harus High Risk."""
    claim = {
        "claim_amount_idr": 55_000_000,
        "has_billing_anomaly": True,
        "policy_active_months": 2,
        "is_frequent_claimer": True,
        "incomplete_docs": True,
    }

    result = predict_claim_risk(claim)

    assert result["risk_category"] == "High Risk"
    assert result["risk_score"] == pytest.approx(1.0)
    assert result["has_anomaly"] is True
    assert result["incomplete_docs"] is True
    assert result["high_value"] is True


def test_medium_risk_classification():
    """Kombinasi fitur risiko sedang harus menghasilkan Medium Risk."""
    claim = {
        "claim_amount_idr": 20_000_000,
        "has_billing_anomaly": False,
        "policy_active_months": 12,
        "is_frequent_claimer": False,
        "incomplete_docs": False,
    }

    result = predict_claim_risk(claim)

    assert result["risk_category"] == "Medium Risk"
    assert result["risk_score"] == pytest.approx(0.35)


def test_risk_score_never_exceeds_one():
    """Risk score harus dibatasi maksimum 1.0."""
    claim = {
        "claim_amount_idr": 100_000_000,
        "has_billing_anomaly": True,
        "policy_active_months": 1,
        "is_frequent_claimer": True,
        "incomplete_docs": True,
    }

    result = predict_claim_risk(claim)

    assert 0.0 <= result["risk_score"] <= 1.0


# ===========================================================================
# 2. TEST DYNAMIC GRAPH
# ===========================================================================

def make_low_risk_claim():
    return {
        "claim_amount_idr": 3_500_000,
        "has_billing_anomaly": False,
        "policy_active_months": 24,
        "is_frequent_claimer": False,
        "incomplete_docs": False,
    }


def make_medium_risk_claim():
    return {
        "claim_amount_idr": 20_000_000,
        "has_billing_anomaly": False,
        "policy_active_months": 12,
        "is_frequent_claimer": False,
        "incomplete_docs": False,
    }


def make_high_risk_claim():
    return {
        "claim_amount_idr": 55_000_000,
        "has_billing_anomaly": True,
        "policy_active_months": 2,
        "is_frequent_claimer": True,
        "incomplete_docs": True,
    }


def test_low_risk_graph_contains_fast_track():
    """Low Risk harus menyediakan jalur FastTrack."""
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    risk_routes = [target for target, _ in graph["RiskScoring"]]

    assert "FastTrack" in risk_routes
    assert "StandardReview" in risk_routes
    assert "HospitalCrossCheck" in risk_routes


def test_medium_risk_graph_contains_standard_routes():
    """Medium Risk harus memiliki beberapa alternatif verifikasi."""
    ml_result = predict_claim_risk(make_medium_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    risk_routes = [target for target, _ in graph["RiskScoring"]]

    assert "StandardReview" in risk_routes
    assert "HospitalCrossCheck" in risk_routes
    assert "MedicalReview" in risk_routes


def test_high_risk_graph_contains_fraud_investigation():
    """High Risk harus menyediakan jalur investigasi fraud."""
    ml_result = predict_claim_risk(make_high_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    risk_routes = [target for target, _ in graph["RiskScoring"]]

    assert "FraudInvestigation" in risk_routes
    assert "HospitalCrossCheck" in risk_routes
    assert "StandardReview" in risk_routes


def test_incomplete_documents_create_document_request_branch():
    """Dokumen tidak lengkap harus membuat cabang DocRequestPending."""
    ml_result = predict_claim_risk(make_high_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    doc_routes = [target for target, _ in graph["DocCheck"]]

    assert "DocRequestPending" in doc_routes
    assert "PolicyValidation" in doc_routes


def test_complete_documents_skip_document_request():
    """Dokumen lengkap tidak membutuhkan DocRequestPending."""
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    doc_routes = [target for target, _ in graph["DocCheck"]]

    assert "DocRequestPending" not in doc_routes
    assert "PolicyValidation" in doc_routes


# ===========================================================================
# 3. TEST UNIFORM COST SEARCH
# ===========================================================================

def test_ucs_finds_valid_low_risk_route():
    """UCS harus menemukan goal dari klaim Low Risk."""
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    result = uniform_cost_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
    )

    assert result is not None
    assert isinstance(result, SearchResult)
    assert result.path[0] == "Submitted"
    assert result.path[-1] in {"Disbursement", "Rejected"}
    assert result.total_cost >= 0.0


def test_ucs_selects_lowest_cost_route_for_low_risk():
    """
    Pada Low Risk, UCS seharusnya memilih jalur FastTrack
    karena jalur tersebut memiliki biaya terendah.
    """
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    result = uniform_cost_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
    )

    assert result is not None

    expected_path = [
        "Submitted",
        "DocCheck",
        "PolicyValidation",
        "RiskScoring",
        "FastTrack",
        "AutoDisbursement",
        "Disbursement",
    ]

    expected_cost = 7.2

    assert result.path == expected_path
    assert result.total_cost == pytest.approx(expected_cost)


def test_ucs_high_risk_reaches_valid_goal():
    """UCS harus mampu mencari solusi pada klaim High Risk."""
    ml_result = predict_claim_risk(make_high_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    result = uniform_cost_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
    )

    assert result is not None
    assert result.path[0] == "Submitted"
    assert result.path[-1] in {"Disbursement", "Rejected"}


# ===========================================================================
# 4. TEST A* SEARCH
# ===========================================================================

def test_astar_finds_valid_route():
    """A* harus menemukan goal yang valid."""
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    result = astar_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
    )

    assert result is not None
    assert isinstance(result, SearchResult)
    assert result.path[0] == "Submitted"
    assert result.path[-1] in {"Disbursement", "Rejected"}
    assert result.total_cost >= 0.0


def test_ucs_and_astar_have_same_optimal_cost_with_zero_heuristic():
    """
    Dengan heuristic nol, A* berubah menjadi pencarian berdasarkan g(n)
    sehingga hasil biayanya harus sama dengan UCS.
    """
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    zero_heuristic = {state: 0.0 for state in graph}

    res_ucs = uniform_cost_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
    )

    res_astar = astar_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
        zero_heuristic,
    )

    assert res_ucs is not None
    assert res_astar is not None

    assert res_astar.total_cost == pytest.approx(
        res_ucs.total_cost,
        abs=1e-6,
    )


def test_astar_with_project_heuristic_returns_valid_solution():
    """
    Heuristic bawaan proyek harus tetap menghasilkan solusi valid.
    Pengujian ini tidak memaksakan path identik dengan UCS karena
    nilai heuristic saat ini merupakan heuristic manual.
    """
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    result = astar_search(
        graph,
        "Submitted",
        {"Disbursement", "Rejected"},
        HEURISTIC,
    )

    assert result is not None
    assert result.path[0] == "Submitted"
    assert result.path[-1] in {"Disbursement", "Rejected"}
    assert result.total_cost >= 0.0


# ===========================================================================
# 5. TEST HEURISTIC BASIC INTEGRITY
# ===========================================================================

def test_heuristic_goal_states_are_zero():
    """Heuristic pada goal harus 0."""
    assert HEURISTIC["Disbursement"] == 0.0
    assert HEURISTIC["Rejected"] == 0.0


def test_heuristic_values_are_non_negative():
    """Semua nilai heuristic tidak boleh negatif."""
    for state, value in HEURISTIC.items():
        assert value >= 0.0, (
            f"Heuristic state '{state}' tidak boleh negatif."
        )


# ===========================================================================
# 6. GRAPH DATA INTEGRITY
# ===========================================================================

def test_all_graph_edges_have_positive_weights():
    """Semua bobot edge harus positif."""
    ml_result = predict_claim_risk(make_high_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    for state, neighbors in graph.items():
        for neighbor, weight in neighbors:
            assert weight > 0.0, (
                f"Bobot edge {state} -> {neighbor} harus positif."
            )


def test_goal_states_have_no_outgoing_edges():
    """Disbursement dan Rejected harus menjadi terminal state."""
    ml_result = predict_claim_risk(make_high_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    assert graph["Disbursement"] == []
    assert graph["Rejected"] == []


# ===========================================================================
# 7. ROBUSTNESS & EDGE CASES
# ===========================================================================

def test_unreachable_goal_returns_none():
    """Jika goal tidak dapat dicapai, search mengembalikan None."""
    ml_result = predict_claim_risk(make_low_risk_claim())
    graph = build_dynamic_claim_graph(ml_result)

    assert (
        uniform_cost_search(
            graph,
            "Submitted",
            {"NonExistentState"},
        )
        is None
    )

    assert (
        astar_search(
            graph,
            "Submitted",
            {"NonExistentState"},
        )
        is None
    )


def test_cycle_does_not_cause_infinite_loop():
    """UCS harus mampu menangani graph yang memiliki cycle."""
    cyclic_graph = {
        "A": [("B", 1.0)],
        "B": [
            ("A", 1.0),
            ("C", 5.0),
        ],
        "C": [],
    }

    result = uniform_cost_search(
        cyclic_graph,
        "A",
        {"C"},
    )

    assert result is not None
    assert result.path == ["A", "B", "C"]
    assert result.total_cost == pytest.approx(6.0)


def test_astar_cycle_does_not_cause_infinite_loop():
    """A* juga harus mampu menangani graph yang memiliki cycle."""
    cyclic_graph = {
        "A": [("B", 1.0)],
        "B": [
            ("A", 1.0),
            ("C", 5.0),
        ],
        "C": [],
    }

    heuristic = {
        "A": 6.0,
        "B": 5.0,
        "C": 0.0,
    }

    result = astar_search(
        cyclic_graph,
        "A",
        {"C"},
        heuristic,
    )

    assert result is not None
    assert result.path == ["A", "B", "C"]
    assert result.total_cost == pytest.approx(6.0)


def test_negative_edge_weight_raises_error():
    """Search harus menolak edge dengan bobot negatif."""
    invalid_graph = {
        "A": [("B", -2.0)],
        "B": [],
    }

    with pytest.raises(
        ValueError,
        match="Bobot edge tidak boleh negatif",
    ):
        uniform_cost_search(
            invalid_graph,
            "A",
            {"B"},
        )

    with pytest.raises(
        ValueError,
        match="Bobot edge tidak boleh negatif",
    ):
        astar_search(
            invalid_graph,
            "A",
            {"B"},
        )