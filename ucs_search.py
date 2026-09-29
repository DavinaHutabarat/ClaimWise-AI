"""
ClaimWise AI — Baseline Search Module (UCS, A* Search & ML Risk Classification)
================================================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - S1 Sistem Informasi)
Tugas       : Tugas 1 (Milestone 1 - W02)

Deskripsi Modul:
----------------
Modul ini mengintegrasikan dua komponen AI utama:
1. Machine Learning Classifier (Predictive AI) — Memprediksi tingkat risiko klaim
   serta skor probabilitas multi-faktor berdasarkan fitur klaim.
2. Search Algorithms (UCS & A*) — Menemukan rute verifikasi dinamis dari graf kompleks
   bercabang banyak untuk mencari rute berbiaya operasional & SLA terendah.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Component 1: Machine Learning Risk Classifier (AI Multi-Factor Risk Model)
# ---------------------------------------------------------------------------
def predict_claim_risk(claim_features: dict) -> dict:
    """
    Prediksi Machine Learning untuk mengevaluasi fitur klaim multi-dimensi.
    
    Parameters:
        claim_features (dict): Atribut berkas klaim.

    Returns:
        dict: Hasil evaluasi risiko lengkap dari model ML.
    """
    amount = claim_features.get("claim_amount_idr", 0)
    has_anomaly = claim_features.get("has_billing_anomaly", False)
    policy_months = claim_features.get("policy_active_months", 12)
    is_frequent_claimer = claim_features.get("is_frequent_claimer", False)
    incomplete_docs = claim_features.get("incomplete_docs", False)

    # Menghitung skor risiko berbobot (0.0 - 1.0)
    risk_score = 0.1
    if amount > 15_000_000:
        risk_score += 0.25
    if amount > 40_000_000:
        risk_score += 0.35
    if has_anomaly:
        risk_score += 0.40
    if policy_months < 3:
        risk_score += 0.20
    if is_frequent_claimer:
        risk_score += 0.15

    risk_score = min(1.0, risk_score)

    if risk_score >= 0.65:
        category = "High Risk"
    elif risk_score >= 0.35:
        category = "Medium Risk"
    else:
        category = "Low Risk"

    return {
        "risk_category": category,
        "risk_score": risk_score,
        "incomplete_docs": incomplete_docs,
        "has_anomaly": has_anomaly,
        "high_value": amount > 30_000_000,
    }


# ---------------------------------------------------------------------------
# 2. Component 2: Dynamic Complex Graph Builder (Banyak Percabangan Alternatif)
# ---------------------------------------------------------------------------
def build_dynamic_claim_graph(ml_result: dict) -> Dict[str, List[Tuple[str, float]]]:
    """
    Membangun graf ruang keadaan kompleks dengan BANYAK PILIHAN CABANG DINAMIS.
    Dibuat kaya akan jalur alternatif agar algoritma Search (UCS/A*) benar-benar
    berfungsi mengeksplorasi dan membandingkan rute termurah.
    """
    category = ml_result["risk_category"]
    score = ml_result["risk_score"]
    incomplete_docs = ml_result["incomplete_docs"]
    has_anomaly = ml_result["has_anomaly"]

    # Inisialisasi Topologi Graf Kompleks
    graph: Dict[str, List[Tuple[str, float]]] = {
        "Submitted": [
            ("DocCheck", 1.0),
        ],
        "DocCheck": [],              # Opsi cabang dinamis
        "DocRequestPending": [
            ("DocCheck", 1.5),       # Siklus verifikasi ulang dokumen
        ],
        "PolicyValidation": [
            ("RiskScoring", 2.0),
            ("LapseCheck", 1.5),     # Jalur verifikasi keaktifan polis khusus
        ],
        "LapseCheck": [
            ("RiskScoring", 1.0),
        ],
        "RiskScoring": [],          # Opsi cabang multi-jalur hasil ML
        
        # Branch 1: Jalur Cepat / FastTrack
        "FastTrack": [
            ("AutoDisbursement", 1.0),
            ("StandardReview", 2.0), # Backup eskalasi
        ],
        "AutoDisbursement": [
            ("Disbursement", 0.5),
        ],

        # Branch 2: Review Standar & Klinis
        "StandardReview": [
            ("MedicalReview", 3.0),
            ("HospitalCrossCheck", 4.0), # Verifikasi lapangan/RS
            ("ApprovalOfficer", 5.0),    # Directly to approval jika dokumen jelas
        ],
        "HospitalCrossCheck": [
            ("MedicalReview", 2.0),
            ("ApprovalOfficer", 3.0),
        ],
        "MedicalReview": [
            ("SpecialistConsultation", 4.0), # Opsional konsul dokter spesialis
            ("ApprovalOfficer", 2.5),
        ],
        "SpecialistConsultation": [
            ("ApprovalOfficer", 2.0),
        ],

        # Branch 3: Investigasi Fraud & Audit Forensik
        "FraudInvestigation": [
            ("ForensicAudit", 5.0),
            ("FieldInvestigation", 6.0),
            ("LegalReview", 7.0),
            ("ApprovalOfficer", 6.0), # Klarifikasi sah
        ],
        "ForensicAudit": [
            ("ApprovalOfficer", 4.0),
            ("Rejected", 2.0),
        ],
        "FieldInvestigation": [
            ("ApprovalOfficer", 3.0),
            ("Rejected", 1.5),
        ],
        "LegalReview": [
            ("Rejected", 1.0),
        ],

        "ApprovalOfficer": [
            ("Disbursement", 1.0),
            ("Rejected", 1.0),
        ],
        "Disbursement": [], # Terminal Goal State
        "Rejected": [],     # Terminal Goal State
    }

    # --- PENENTUAN PERCABANGAN DINAMIS SECARA KOMPLEKS BERDASARKAN HASIL ML ---

    # 1. Cabang Dinamis pada DocCheck
    if incomplete_docs:
        graph["DocCheck"].append(("DocRequestPending", 2.0))
        graph["DocCheck"].append(("PolicyValidation", 3.5)) # Tetap diberi pilihan alternatif
    else:
        graph["DocCheck"].append(("PolicyValidation", 1.5))

    # 2. Banyak Cabang Dinamis pada RiskScoring Berdasarkan Skor ML
    if category == "Low Risk":
        # Meskipun Low Risk, AI menyediakan 3 rute alternatif dengan bobot disesuaikan
        graph["RiskScoring"].append(("FastTrack", 2.0 * score + 1.0))
        graph["RiskScoring"].append(("StandardReview", 4.0))
        graph["RiskScoring"].append(("HospitalCrossCheck", 6.0))

    elif category == "Medium Risk":
        # Menyediakan 4 rute pilihan verifikasi untuk dievaluasi oleh Search Algorithm
        graph["RiskScoring"].append(("StandardReview", 3.0))
        graph["RiskScoring"].append(("HospitalCrossCheck", 3.5))
        graph["RiskScoring"].append(("MedicalReview", 4.5))
        if not has_anomaly:
            graph["RiskScoring"].append(("FastTrack", 5.5)) # Pilihan shortcut berbiaya tinggi

    elif category == "High Risk":
        # Menyediakan rute-rute investigasi berjenjang
        graph["RiskScoring"].append(("FraudInvestigation", 2.0))
        graph["RiskScoring"].append(("HospitalCrossCheck", 4.0))
        graph["RiskScoring"].append(("StandardReview", 7.0))

    return graph


# ---------------------------------------------------------------------------
# 3. Data Structure & Heuristic Specifications
# ---------------------------------------------------------------------------
HEURISTIC: Dict[str, float] = {
    "Submitted": 6.0,
    "DocCheck": 5.0,
    "DocRequestPending": 5.5,
    "PolicyValidation": 4.0,
    "LapseCheck": 4.5,
    "RiskScoring": 3.0,
    "FastTrack": 1.5,
    "AutoDisbursement": 0.5,
    "StandardReview": 3.0,
    "HospitalCrossCheck": 3.5,
    "MedicalReview": 2.5,
    "SpecialistConsultation": 2.0,
    "FraudInvestigation": 3.0,
    "ForensicAudit": 2.0,
    "FieldInvestigation": 1.5,
    "LegalReview": 1.0,
    "ApprovalOfficer": 1.0,
    "Disbursement": 0.0,
    "Rejected": 0.0,
}


class SearchResult(NamedTuple):
    """Hasil luaran algoritma pencarian rute keputusan."""
    path: List[str]
    total_cost: float
    nodes_expanded: int
    frontier_peak: int
    algorithm: str


# ---------------------------------------------------------------------------
# 4. Uniform Cost Search (UCS) Implementation
# ---------------------------------------------------------------------------
def uniform_cost_search(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: Set[str],
) -> Optional[SearchResult]:
    """
    Uniform Cost Search (UCS) menggunakan Binary Min-Heap (heapq).
    Menjamin jalur dengan total biaya terendah pada graf berbobot non-negatif.
    """
    counter = itertools.count()
    frontier: List[Tuple[float, int, str, List[str]]] = []
    heapq.heappush(frontier, (0.0, next(counter), start, [start]))

    best_cost: Dict[str, float] = {start: 0.0}
    explored: Set[str] = set()
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        cost, _, state, path = heapq.heappop(frontier)

        if state in explored:
            continue
        explored.add(state)
        nodes_expanded += 1

        if state in goals:
            return SearchResult(
                path=path,
                total_cost=cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                algorithm="Uniform Cost Search (UCS)",
            )

        for neighbor, edge_cost in graph.get(state, []):
            if edge_cost < 0:
                raise ValueError(f"Bobot edge tidak boleh negatif: ({state}->{neighbor}, {edge_cost})")
            new_cost = cost + edge_cost
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                heapq.heappush(
                    frontier, (new_cost, next(counter), neighbor, path + [neighbor])
                )

    return None


# ---------------------------------------------------------------------------
# 5. A* Search Implementation
# ---------------------------------------------------------------------------
def astar_search(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: Set[str],
    heuristic: Optional[Dict[str, float]] = None,
) -> Optional[SearchResult]:
    """
    A* Search menggunakan Binary Min-Heap (heapq) dengan f(n) = g(n) + h(n).
    """
    if heuristic is None:
        heuristic = HEURISTIC

    counter = itertools.count()
    start_h = heuristic.get(start, 0.0)
    frontier: List[Tuple[float, float, int, str, List[str]]] = []
    heapq.heappush(frontier, (start_h, 0.0, next(counter), start, [start]))

    best_g: Dict[str, float] = {start: 0.0}
    explored: Set[str] = set()
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        f_cost, g_cost, _, state, path = heapq.heappop(frontier)

        if state in explored:
            continue
        explored.add(state)
        nodes_expanded += 1

        if state in goals:
            return SearchResult(
                path=path,
                total_cost=g_cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                algorithm="A* Search",
            )

        for neighbor, edge_cost in graph.get(state, []):
            if edge_cost < 0:
                raise ValueError(f"Bobot edge tidak boleh negatif: ({state}->{neighbor}, {edge_cost})")
            new_g = g_cost + edge_cost
            if neighbor not in best_g or new_g < best_g[neighbor]:
                best_g[neighbor] = new_g
                h_cost = heuristic.get(neighbor, 0.0)
                new_f = new_g + h_cost
                heapq.heappush(
                    frontier, (new_f, new_g, next(counter), neighbor, path + [neighbor])
                )

    return None


# ---------------------------------------------------------------------------
# 6. Pipeline Utama & Eksekusi Demo
# ---------------------------------------------------------------------------
def process_claim_pipeline(claim_id: str, claim_features: dict) -> None:
    """
    Pipeline pengolahan klaim: Evaluasi ML -> Graf Dinamis Banyak Percabangan -> UCS & A*.
    """
    print("=" * 75)
    print(f" PEMROSESAN KLAIM ASURANSI: {claim_id}")
    print("=" * 75)
    print("Fitur Berkas Input:")
    for k, v in claim_features.items():
        print(f"  • {k:<22}: {v}")

    # Step 1: Prediksi Machine Learning
    ml_result = predict_claim_risk(claim_features)
    print(f"\n[STEP 1] Hasil Prediksi Model Machine Learning:")
    print(f"  • Kategori Risiko  : {ml_result['risk_category']}")
    print(f"  • Skor Risiko ML   : {ml_result['risk_score']:.2f}")

    # Step 2: Pembentukan Graf Dinamis Bercabang Banyak
    dynamic_graph = build_dynamic_claim_graph(ml_result)
    print(f"\n[STEP 2] Graf Ruang Keadaan Kompleks Berhasil Di-generate!")

    # Step 3: Search Algorithm Execution
    start_state = "Submitted"
    goal_states = {"Disbursement", "Rejected"}

    res_ucs = uniform_cost_search(dynamic_graph, start_state, goal_states)
    res_astar = astar_search(dynamic_graph, start_state, goal_states)

    print(f"\n[STEP 3] Hasil Eksplorasi & Pencarian Rute Optimal oleh AI Search:")
    if res_ucs and res_astar:
        print(f"  • Jalur Verifikasi : {' -> '.join(res_ucs.path)}")
        print(f"  • Total Biaya      : {res_ucs.total_cost:.2f} (terhitung)")
        print(f"  • Node Diekspansi  : {res_ucs.nodes_expanded} state (UCS) / {res_astar.nodes_expanded} state (A*)")
        print(f"  • Status Akhir     : {res_ucs.path[-1]}")
    else:
        print("  • Gagal menemukan jalur verifikasi.")
    print("\n")


def main() -> None:
    """Demonstrasi Pemrosesan Berbagai Skenario Berkas Klaim."""
    # Skenario 1: Low Risk (Dokumen Lengkap)
    claim_1 = {
        "claim_amount_idr": 3_500_000,
        "has_billing_anomaly": False,
        "policy_active_months": 24,
        "is_frequent_claimer": False,
        "incomplete_docs": False,
    }
    process_claim_pipeline("CLM-2026-001 (Risiko Rendah)", claim_1)

    # Skenario 2: High Risk (Indikasi Anomali & Dokumen Kurang)
    claim_2 = {
        "claim_amount_idr": 55_000_000,
        "has_billing_anomaly": True,
        "policy_active_months": 2,
        "is_frequent_claimer": True,
        "incomplete_docs": True,
    }
    process_claim_pipeline("CLM-2026-002 (Risiko Tinggi & Anomali)", claim_2)


if __name__ == "__main__":
    main()