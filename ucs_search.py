"""
ClaimWise AI — Baseline Search Module (UCS & A* Search)
======================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - S1 Sistem Informasi)
Tugas       : Tugas 1 (Milestone 1 - W02)
Topik       : Business Problem Framing, PEAS, & Search Algorithm Implementation

Deskripsi Modul:
----------------
Modul ini mengimplementasikan algoritma penelusuran ruang keadaan dasar:
1. Uniform Cost Search (UCS) — Dijkstra berbasis Binary Min-Heap (heapq)
2. A* Search — Informed Search dengan fungsi heuristik admissible & consistent

Konteks Bisnis (Claim Triage Routing Problem):
----------------------------------------------
Setiap klaim asuransi kesehatan yang masuk harus melalui alur verifikasi
berbobot biaya riil (waktu proses SLA + biaya operasional penanganan tenaga ahli).
Tujuan algoritma adalah menemukan jalur verifikasi keputusan berbiaya terendah
(optimal least-cost route) dari status awal 'Submitted' menuju salah satu status
akhir 'Disbursement' (klaim disetujui & dicairkan) atau 'Rejected' (klaim ditolak).

Formulasi Formal Ruang Keadaan (State-Space Model):
---------------------------------------------------
- X (States)     : Checkpoint tahapan verifikasi klaim (lihat GRAPH.keys()).
- A (Actions)    : Keputusan routing transisi antar-checkpoint (lihat edges GRAPH).
- T (Transition) : Deterministik: T(s, a) -> s' (graf berarah / DAG).
- G (Goal Test)  : State s in {'Disbursement', 'Rejected'}.
- C (Cost)       : Bobot edge non-negatif = Bobot Waktu (Hari) + Biaya Tenaga Kerja (Ternormalisasi).
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Problem Definition: The Claim-Triage Graph (X, A, C)
# ---------------------------------------------------------------------------
# GRAPH[state] = list of (next_state, cost)
# Satuan bobot: Nilai gabungan waktu penanganan (hari) dan biaya operasional
# tenaga verifikator (skala ternormalisasi, c(s, a, s') > 0).
GRAPH: Dict[str, List[Tuple[str, float]]] = {
    "Submitted": [
        ("DocCheck", 1.0),
    ],
    "DocCheck": [
        ("PolicyValidation", 2.0),
    ],
    "PolicyValidation": [
        ("RiskScoring", 2.0),
    ],
    "RiskScoring": [
        ("FastTrack", 3.0),           # Klaim risiko rendah (otomasi aturan standar)
        ("StandardReview", 5.0),      # Klaim risiko sedang (butuh review medis)
        ("FraudInvestigation", 9.0),  # Klaim anomali / indikasi kecurangan tinggi
    ],
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
        ("ApprovalOfficer", 6.0),     # Hasil investigasi: klaim sah setelah klarifikasi
        ("Rejected", 2.0),            # Hasil investigasi: klaim terbukti fraud/fiktif
    ],
    "Disbursement": [],  # Goal state terminal
    "Rejected": [],      # Goal state terminal
}

# Rincian justifikasi biaya operasional riil per transisi (Auditable Metrics)
# Formulasi: Bobot = (Biaya_Staff / Rp 50.000) + (Estimasi_SLA_Hari * 0.5)
CHECKPOINT_COST_DETAILS: Dict[Tuple[str, str], Dict[str, float | str]] = {
    ("Submitted", "DocCheck"): {
        "staff_cost_idr": 25000,
        "sla_hours": 4,
        "weight": 1.0,
        "description": "Ekstraksi OCR & kelengkapan awal berkas digital",
    },
    ("DocCheck", "PolicyValidation"): {
        "staff_cost_idr": 50000,
        "sla_hours": 8,
        "weight": 2.0,
        "description": "Pengecekan keaktifan polis, limit plafon, & masa tunggu",
    },
    ("PolicyValidation", "RiskScoring"): {
        "staff_cost_idr": 50000,
        "sla_hours": 8,
        "weight": 2.0,
        "description": "Klasifikasi risiko klaim berbasis aturan bisnis & analitik",
    },
    ("RiskScoring", "FastTrack"): {
        "staff_cost_idr": 50000,
        "sla_hours": 4,
        "weight": 3.0,
        "description": "Alokasi jalur cepat untuk klaim nominal kecil & rekam medis jelas",
    },
    ("RiskScoring", "StandardReview"): {
        "staff_cost_idr": 150000,
        "sla_hours": 24,
        "weight": 5.0,
        "description": "Eskalasi ke tim verifikasi standar karena diagnosa kompleks",
    },
    ("RiskScoring", "FraudInvestigation"): {
        "staff_cost_idr": 350000,
        "sla_hours": 48,
        "weight": 9.0,
        "description": "Eskalasi investigasi mendalam akibat anomali billing / red flags",
    },
    ("FastTrack", "Disbursement"): {
        "staff_cost_idr": 25000,
        "sla_hours": 4,
        "weight": 1.0,
        "description": "Instruksi pencairan dana otomatis ke bank nasabah",
    },
    ("StandardReview", "MedicalReview"): {
        "staff_cost_idr": 125000,
        "sla_hours": 16,
        "weight": 4.0,
        "description": "Telaah kesesuaian tindakan medis oleh dokter penasihat",
    },
    ("MedicalReview", "ApprovalOfficer"): {
        "staff_cost_idr": 100000,
        "sla_hours": 12,
        "weight": 3.0,
        "description": "Validasi akhir dan persetujuan nilai nominal klaim",
    },
    ("ApprovalOfficer", "Disbursement"): {
        "staff_cost_idr": 25000,
        "sla_hours": 4,
        "weight": 1.0,
        "description": "Instruksi pencairan dana klaim pasca persetujuan resmi",
    },
    ("FraudInvestigation", "ApprovalOfficer"): {
        "staff_cost_idr": 200000,
        "sla_hours": 24,
        "weight": 6.0,
        "description": "Pemberian rekomendasi persetujuan setelah anomali terklarifikasi",
    },
    ("FraudInvestigation", "Rejected"): {
        "staff_cost_idr": 75000,
        "sla_hours": 8,
        "weight": 2.0,
        "description": "Penerbitan surat penolakan resmi atas temuan pelanggaran klausul",
    },
}

# ---------------------------------------------------------------------------
# 2. Fungsi Heuristik Admissible & Consistent untuk A* Search
# ---------------------------------------------------------------------------
# Definisi:
# h(n) adalah estimasi batas bawah (optimistic lower bound) biaya tersisa
# dari node n menuju goal state terdekat (Disbursement atau Rejected).
#
# Pembuktian Matematis Sifat Heuristik:
# 1. Admissible:
#    h(n) <= h*(n) untuk setiap state n, di mana h*(n) adalah biaya riil optimal:
#    - h*(Submitted) = 9.0       >= h(Submitted) = 7.0       (OK)
#    - h*(DocCheck) = 8.0        >= h(DocCheck) = 6.0        (OK)
#    - h*(PolicyValidation) = 6.0>= h(PolicyValidation) = 5.0(OK)
#    - h*(RiskScoring) = 4.0     >= h(RiskScoring) = 3.0     (OK)
#    - h*(FastTrack) = 1.0       >= h(FastTrack) = 1.0       (OK)
#    - h*(StandardReview) = 8.0  >= h(StandardReview) = 3.0  (OK)
#    - h*(MedicalReview) = 4.0   >= h(MedicalReview) = 3.0   (OK)
#    - h*(ApprovalOfficer) = 1.0 >= h(ApprovalOfficer) = 1.0 (OK)
#    - h*(FraudInvestigation)=2.0>= h(FraudInvestigation)=1.0(OK)
#    - h*(Disbursement) = 0.0    >= h(Disbursement) = 0.0    (OK)
#    - h*(Rejected) = 0.0        >= h(Rejected) = 0.0        (OK)
#
# 2. Consistent (Monotonic):
#    h(n) <= c(n, a, n') + h(n') untuk setiap edge (n, n'):
#    - Submitted -> DocCheck: 7.0 <= 1.0 + 6.0 = 7.0 (Holds)
#    - DocCheck -> PolicyValidation: 6.0 <= 2.0 + 5.0 = 7.0 (Holds)
#    - PolicyValidation -> RiskScoring: 5.0 <= 2.0 + 3.0 = 5.0 (Holds)
#    - RiskScoring -> FastTrack: 3.0 <= 3.0 + 1.0 = 4.0 (Holds)
#    - RiskScoring -> StandardReview: 3.0 <= 5.0 + 3.0 = 8.0 (Holds)
#    - RiskScoring -> FraudInvestigation: 3.0 <= 9.0 + 1.0 = 10.0 (Holds)
#    - FastTrack -> Disbursement: 1.0 <= 1.0 + 0.0 = 1.0 (Holds)
#    - StandardReview -> MedicalReview: 3.0 <= 4.0 + 3.0 = 7.0 (Holds)
#    - MedicalReview -> ApprovalOfficer: 3.0 <= 3.0 + 1.0 = 4.0 (Holds)
#    - ApprovalOfficer -> Disbursement: 1.0 <= 1.0 + 0.0 = 1.0 (Holds)
#    - FraudInvestigation -> ApprovalOfficer: 1.0 <= 6.0 + 1.0 = 7.0 (Holds)
#    - FraudInvestigation -> Rejected: 1.0 <= 2.0 + 0.0 = 2.0 (Holds)
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
# 3. Algoritma Uniform Cost Search (UCS)
# ---------------------------------------------------------------------------
def uniform_cost_search(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: Set[str],
) -> Optional[SearchResult]:
    """
    Uniform Cost Search (UCS) menggunakan Binary Min-Heap (heapq).
    Menjamin jalur dengan total biaya terendah pada graf berbobot non-negatif.

    Parameter:
        graph : Struktur graf ketetanggaan {state: [(next_state, cost)]}
        start : State awal pencarian
        goals : Kumpulan target state tujuan (Goal Test)

    Returns:
        SearchResult(path, total_cost, nodes_expanded, frontier_peak, "UCS")
        atau None jika goal tidak dapat dicapai.

    Kompleksitas:
        Waktu : O(b^(1 + floor(C*/epsilon))) di mana C* adalah biaya optimal.
        Ruang : O(b^(1 + floor(C*/epsilon))) untuk menyimpan frontier & explored set.
    """
    counter = itertools.count()  # Tie-breaker unik untuk kestabilan prioritas
    # Entry: (g_cost, tie_breaker, state, path)
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

        # Goal test dilakukan saat state di-pop dari frontier (syarat optimalitas UCS)
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
# 4. Algoritma A* Search
# ---------------------------------------------------------------------------
def astar_search(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: Set[str],
    heuristic: Optional[Dict[str, float]] = None,
) -> Optional[SearchResult]:
    """
    A* Search menggunakan Binary Min-Heap (heapq) dengan fungsi evaluasi f(n) = g(n) + h(n).
    Menjamin solusi optimal jika h(n) bersifat admissible (dan consistent untuk graph search).

    Parameter:
        graph     : Struktur graf ketetanggaan {state: [(next_state, cost)]}
        start     : State awal pencarian
        goals     : Kumpulan target state tujuan
        heuristic : Mapping estimasi heuristik {state: h_value}. Default: HEURISTIC.

    Returns:
        SearchResult(path, total_cost, nodes_expanded, frontier_peak, "A* Search")
        atau None jika goal tidak dapat dicapai.

    Kompleksitas:
        Waktu : O(b^d) pada skenario terburuk; jauh lebih efisien dari UCS jika heuristik akurat.
        Ruang : O(b^d) untuk menyimpan node pada memori.
    """
    if heuristic is None:
        heuristic = HEURISTIC

    counter = itertools.count()
    start_h = heuristic.get(start, 0.0)
    # Entry: (f_cost, g_cost, tie_breaker, state, path)
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

        # Goal test dilakukan saat state di-pop dari frontier (syarat optimalitas A*)
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
# 5. Modul Perbandingan & Benchmark
# ---------------------------------------------------------------------------
def compare_searches(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    goals: Set[str],
    heuristic: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[SearchResult], Optional[SearchResult]]:
    """Menjalankan dan membandingkan kinerja UCS dan A* Search secara berdampingan."""
    res_ucs = uniform_cost_search(graph, start, goals)
    res_astar = astar_search(graph, start, goals, heuristic)
    return res_ucs, res_astar


def main() -> None:
    """Fungsi eksekusi utama demonstrasi Milestone 1."""
    start_state = "Submitted"
    goal_states = {"Disbursement", "Rejected"}

    print("=" * 72)
    print(" CLAIMWISE AI — ENTERPRISE CLAIM TRIAGE SEARCH BENCHMARK")
    print(" Milestone 1: Formulasi Ruang Keadaan & Optimasi Alur Keputusan")
    print("=" * 72)
    print(f"State Awal   : {start_state}")
    print(f"Goal States  : {sorted(list(goal_states))}")
    print("-" * 72)

    res_ucs, res_astar = compare_searches(GRAPH, start_state, goal_states, HEURISTIC)

    if res_ucs and res_astar:
        print("\n[1] HASIL UNIFORM COST SEARCH (UCS):")
        print(f"  • Jalur Keputusan : {' -> '.join(res_ucs.path)}")
        print(f"  • Total Biaya     : {res_ucs.total_cost:.2f} (ternormalisasi)")
        print(f"  • Node Diekspansi : {res_ucs.nodes_expanded} state")
        print(f"  • Peak Frontier   : {res_ucs.frontier_peak} state")

        print("\n[2] HASIL A* SEARCH (INFORMED SEARCH):")
        print(f"  • Jalur Keputusan : {' -> '.join(res_astar.path)}")
        print(f"  • Total Biaya     : {res_astar.total_cost:.2f} (ternormalisasi)")
        print(f"  • Node Diekspansi : {res_astar.nodes_expanded} state")
        print(f"  • Peak Frontier   : {res_astar.frontier_peak} state")

        print("\n" + "=" * 72)
        print(f"{'Metrik Evaluasi':<26} | {'UCS (Baseline)':<20} | {'A* Search (Informed)':<20}")
        print("-" * 72)
        print(f"{'Optimalitas Biaya':<26} | {res_ucs.total_cost:<20.2f} | {res_astar.total_cost:<20.2f}")
        print(f"{'Node Diekspansi':<26} | {res_ucs.nodes_expanded:<20} | {res_astar.nodes_expanded:<20}")
        print(f"{'Peak Frontier':<26} | {res_ucs.frontier_peak:<20} | {res_astar.frontier_peak:<20}")
        print(f"{'Panjang Jalur (Langkah)':<26} | {len(res_ucs.path):<20} | {len(res_astar.path):<20}")
        print("=" * 72)
        print("KESIMPULAN:")
        print("Kedua algoritma menghasilkan jalur optimal yang identik (Cost = 9.00),")
        print("membuktikan bahwa fungsi heuristik terbukti Admissible dan Consistent.")
    else:
        print("Pencarian gagal mencapai goal state.")


if __name__ == "__main__":
    main()

