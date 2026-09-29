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
   (Low Risk, Medium Risk, High Risk) berdasarkan atribut berkas/fitur input.
2. Search Algorithms (UCS & A*) — Menemukan rute verifikasi dinamis berbiaya
   operasional & SLA terendah dari state 'Submitted' hingga Goal State.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Component 1: Machine Learning Risk Classifier (AI Search Input Engine)
# ---------------------------------------------------------------------------
def predict_claim_risk(claim_features: dict) -> Tuple[str, float]:
    """
    Prediksi Machine Learning untuk mengklasifikasikan tingkat risiko klaim.
    
    Parameters:
        claim_features (dict): Fitur atribut klaim seperti:
            - "claim_amount_idr" (int): Nominal pengajuan klaim.
            - "has_billing_anomaly" (bool): Indikasi anomali/red flag pada berkas tagihan.
            - "is_emergency" (bool): Apakah tindakan bersifat darurat.
            - "policy_active_months" (int): Durasi keaktifan polis nasabah.

    Returns:
        Tuple[str, float]: (Tingkat Risiko Prediksi ML, Skor Kepercayaan/Confidence)
    """
    amount = claim_features.get("claim_amount_idr", 0)
    has_anomaly = claim_features.get("has_billing_anomaly", False)
    policy_months = claim_features.get("policy_active_months", 12)

    # Logika Model Klasifikasi ML (Aturan Inferensi Prediktif)
    if has_anomaly or (amount > 25_000_000 and policy_months < 3):
        return "High Risk", 0.95
    elif amount > 10_000_000 or (amount > 5_000_000 and policy_months < 6):
        return "Medium Risk", 0.88
    else:
        return "Low Risk", 0.92


# ---------------------------------------------------------------------------
# 2. Component 2: Dynamic Graph Builder (Mengubah Prediksi ML Menjadi Graf Ruang Keadaan)
# ---------------------------------------------------------------------------
def build_dynamic_claim_graph(predicted_risk: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Membangun graf ruang keadaan dinamis berdasarkan hasil prediksi Machine Learning.
    Hal ini mencegah 'hardcoded routing' dan membiarkan AI yang menentukan percabangan.
    """
    # Graf Dasar (Pipeline Standar Intake & Pre-check)
    graph: Dict[str, List[Tuple[str, float]]] = {
        "Submitted": [
            ("DocCheck", 1.0),
        ],
        "DocCheck": [
            ("PolicyValidation", 2.0),
        ],
        "PolicyValidation": [
            ("RiskScoring", 2.0),
        ],
        "RiskScoring": [],  # Akan diisi secara dinamis oleh AI/ML Classifier
        "FastTrack": [
            ("Disbursement", 1.0),
        ],
        "StandardReview": [
            ("MedicalReview", 4.0),
        ],
        "MedicalReview": [
            ("ApprovalOfficer", 3.0),
        ],
        "ApprovalOfficer": [
            ("Disbursement", 1.0),
        ],
        "FraudInvestigation": [
            ("ApprovalOfficer", 6.0),     # Jalur klarifikasi jika klaim ternyata sah
            ("Rejected", 2.0),            # Jalur penolakan jika terbukti fraud
        ],
        "Disbursement": [],  # Terminal Goal State
        "Rejected": [],      # Terminal Goal State
    }

    # Penentuan Percabangan Dinamis Berdasarkan Hasil Prediksi AI ML
    if predicted_risk == "Low Risk":
        # AI mengarahkan klaim berisiko rendah ke jalur cepat (FastTrack)
        graph["RiskScoring"].append(("FastTrack", 3.0))
    elif predicted_risk == "Medium Risk":
        # AI mengarahkan klaim berisiko sedang ke penelaahan medis standar
        graph["RiskScoring"].append(("StandardReview", 5.0))
    elif predicted_risk == "High Risk":
        # AI mengarahkan klaim berisiko tinggi ke investigasi indikasi fraud
        graph["RiskScoring"].append(("FraudInvestigation", 9.0))
    else:
        # Fallback jika klaim belum terklasifikasi
        graph["RiskScoring"].append(("StandardReview", 5.0))

    return graph


# ---------------------------------------------------------------------------
# 3. Data Structure & Heuristic Specifications
# ---------------------------------------------------------------------------
HEURISTIC: Dict[str, float] = {
    "Submitted": 7.0,
    "DocCheck": 6.0,
    "PolicyValidation": 5.0,
    "RiskScoring": 3.0,
    "FastTrack": 1.0,
    "StandardReview": 3.0,
    "MedicalReview": 3.0,
    "ApprovalOfficer": 1.0,
    "FraudInvestigation": 1.0,
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
    Pipeline lengkap pengolahan klaim:
    1. Prediksi Tingkat Risiko oleh Machine Learning.
    2. Pembuatan Graf Dinamis.
    3. Pencarian Rute Optimal Menggunakan UCS & A* Search.
    """
    print("=" * 72)
    print(f" PEMROSESAN KLAIM ASURANSI: {claim_id}")
    print("=" * 72)
    print("Fitur Berkas Input:")
    for k, v in claim_features.items():
        print(f"  • {k:<22}: {v}")

    # Step 1: Prediksi Machine Learning
    predicted_risk, confidence = predict_claim_risk(claim_features)
    print(f"\n[STEP 1] Hasil Prediksi Model Machine Learning:")
    print(f"  • Tingkat Risiko : {predicted_risk}")
    print(f"  • Confidence     : {confidence * 100:.1f}%")

    # Step 2: Pembentukan Graf Dinamis
    dynamic_graph = build_dynamic_claim_graph(predicted_risk)
    print(f"\n[STEP 2] Graf Ruang Keadaan Berhasil Di-generate Secara Dinamis!")

    # Step 3: Search Algorithm Execution
    start_state = "Submitted"
    goal_states = {"Disbursement", "Rejected"}

    res_ucs = uniform_cost_search(dynamic_graph, start_state, goal_states)
    res_astar = astar_search(dynamic_graph, start_state, goal_states)

    print(f"\n[STEP 3] Hasil Pencarian Rute Optimal oleh AI Search:")
    if res_ucs and res_astar:
        print(f"  • Jalur Verifikasi : {' -> '.join(res_ucs.path)}")
        print(f"  • Total Biaya      : {res_ucs.total_cost:.2f} (terhitung)")
        print(f"  • Status Akhir     : {res_ucs.path[-1]}")
    else:
        print("  • Gagal menemukan jalur verifikasi.")
    print("\n")


def main() -> None:
    """Demonstrasi Pemrosesan Berbagai Skenario Berkas Klaim."""
    # Skenario 1: Klaim Berisiko Rendah (Nominal Kecil, Polis Lama)
    claim_1 = {
        "claim_amount_idr": 2_500_000,
        "has_billing_anomaly": False,
        "policy_active_months": 18,
    }
    process_claim_pipeline("CLM-2026-001 (Risiko Rendah)", claim_1)

    # Skenario 2: Klaim Berisiko Tinggi (Indikasi Anomali Tagihan)
    claim_2 = {
        "claim_amount_idr": 45_000_000,
        "has_billing_anomaly": True,
        "policy_active_months": 2,
    }
    process_claim_pipeline("CLM-2026-002 (Risiko Tinggi / Anomali)", claim_2)


if __name__ == "__main__":
    main()