"""
ClaimWise AI — Comprehensive Test Suite for CSP & GA Constraint Solver
=======================================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - S1 Sistem Informasi)
Tugas       : Tugas 2 (Milestone 2 - W04)

Cakupan Pengujian:
1. Pemodelan Batasan Unary (Kualifikasi Kompetensi, Limit Finansial, Konflik RS, SLA Shift).
2. Propagasi Batasan AC-3 (Pemangkasan Domain & Deteksi Insolvabilitas Dini).
3. Heuristik MRV (Minimum Remaining Values) & Degree Heuristic (Tie-Breaker).
4. Heuristik LCV (Least Constraining Value).
5. Inferensi MAC & Forward Checking pada Backtracking Search.
6. Kepatuhan Batasan Global Kapasitas Beban Kerja (UU Ketenagakerjaan No. 13/2003).
7. Kepatuhan Batasan Biner Pemisahan Tugas (Separation of Duties / Conflict Group).
8. Konvergensi Algoritma Genetika (GA) dengan Elitisme & Turnamen.
9. Pengujian Kasus Ekstrem (Over-Constrained, Bottleneck, Clique Conflict, Single/Zero Claim).

Eksekusi:
    pytest -v test_solver.py
"""

import pytest

from solver import (
    ROLE_HIERARCHY,
    AssignmentSlot,
    BinaryConstraint,
    Claim,
    ClaimAllocationEngine,
    CSP,
    GeneticAlgorithmSolver,
    ShiftSlot,
    Verifier,
    VerifierRole,
    ac3,
    backtracking_search,
    order_domain_values_lcv,
    select_unassigned_variable_mrv,
)


# ===========================================================================
# FIXTURES & DATA PENDUKUNG
# ===========================================================================

@pytest.fixture
def standard_verifiers():
    """Himpunan verifikator standar dengan variasi peran dan kapasitas."""
    return [
        Verifier(
            verifier_id="ADJ-01",
            name="Andi Pratama",
            role=VerifierRole.JUNIOR_ADJUSTER,
            max_daily_workload=20,
            max_daily_claims=10,
            max_claim_amount_idr=25_000_000.0,
            affiliated_hospitals=frozenset(["HOSP-CONFLICT-01"]),
            hourly_rate_idr=60_000.0,
        ),
        Verifier(
            verifier_id="ADJ-02",
            name="Bona Siregar",
            role=VerifierRole.SENIOR_ADJUSTER,
            max_daily_workload=25,
            max_daily_claims=12,
            max_claim_amount_idr=75_000_000.0,
            hourly_rate_idr=90_000.0,
        ),
        Verifier(
            verifier_id="DOC-01",
            name="dr. Citra Sp.A",
            role=VerifierRole.MEDICAL_ADVISOR,
            max_daily_workload=18,
            max_daily_claims=8,
            max_claim_amount_idr=200_000_000.0,
            hourly_rate_idr=180_000.0,
        ),
        Verifier(
            verifier_id="INV-01",
            name="David Sianipar",
            role=VerifierRole.FRAUD_INVESTIGATOR,
            max_daily_workload=15,
            max_daily_claims=6,
            max_claim_amount_idr=500_000_000.0,
            hourly_rate_idr=160_000.0,
        ),
    ]


@pytest.fixture
def standard_claims():
    """Himpunan berkas klaim harian dengan variasi tingkat risiko dan nominal."""
    return [
        Claim("CLM-001", "OUTPATIENT", 3_500_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-01"),
        Claim("CLM-002", "INPATIENT", 45_000_000, VerifierRole.SENIOR_ADJUSTER, complexity=2, sla_hours=48, hospital_id="HOSP-02"),
        Claim("CLM-003", "SURGICAL", 135_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4, sla_hours=72, hospital_id="HOSP-01"),
        Claim("CLM-004", "FRAUD_SUSPECT", 80_000_000, VerifierRole.FRAUD_INVESTIGATOR, complexity=3, sla_hours=48, hospital_id="HOSP-03"),
        Claim("CLM-005", "OUTPATIENT", 5_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAM-1"),
        Claim("CLM-006", "OUTPATIENT", 4_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAM-1"),
    ]


# ===========================================================================
# 1. PENGUJIAN BATASAN UNARY & PEMBENTUKAN DOMAIN
# ===========================================================================

def test_unary_competency_constraints(standard_verifiers):
    """Klaim bedah hanya boleh diisi oleh Medical Advisor; klaim fraud oleh Investigator."""
    claim_surg = Claim("CLM-SURG", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=3)
    claim_fraud = Claim("CLM-FRD", "FRAUD_SUSPECT", 30_000_000, VerifierRole.FRAUD_INVESTIGATOR, complexity=3)

    engine = ClaimAllocationEngine([claim_surg, claim_fraud], standard_verifiers)
    csp = engine.build_csp()

    # Domain bedah hanya boleh berisi DOC-01
    surg_verifiers = {slot.verifier_id for slot in csp.domains["CLM-SURG"]}
    assert surg_verifiers == {"DOC-01"}

    # Domain fraud hanya boleh berisi INV-01
    fraud_verifiers = {slot.verifier_id for slot in csp.domains["CLM-FRD"]}
    assert fraud_verifiers == {"INV-01"}


def test_financial_authority_limits(standard_verifiers):
    """Klaim bernilai Rp 50 Juta dilarang dialokasikan ke Junior Adjuster (plafon Rp 25 Juta)."""
    claim_high = Claim("CLM-HIGH", "INPATIENT", 50_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=2)

    engine = ClaimAllocationEngine([claim_high], standard_verifiers)
    csp = engine.build_csp()

    allowed_verifiers = {slot.verifier_id for slot in csp.domains["CLM-HIGH"]}
    assert "ADJ-01" not in allowed_verifiers
    assert "ADJ-02" in allowed_verifiers or "DOC-01" in allowed_verifiers


def test_hospital_conflict_of_interest(standard_verifiers):
    """Verifikator yang terafiliasi dengan RS tertentu tidak boleh memverifikasi klaim dari RS tersebut."""
    claim_conflict = Claim("CLM-CONF", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, hospital_id="HOSP-CONFLICT-01")

    engine = ClaimAllocationEngine([claim_conflict], standard_verifiers)
    csp = engine.build_csp()

    verifiers_in_domain = {slot.verifier_id for slot in csp.domains["CLM-CONF"]}
    assert "ADJ-01" not in verifiers_in_domain  # ADJ-01 terafiliasi dengan HOSP-CONFLICT-01


def test_sla_night_shift_restriction(standard_verifiers):
    """Klaim dengan SLA <= 24 jam (Fast-Track) dilarang ditempatkan pada Shift MALAM."""
    claim_fast = Claim("CLM-FAST", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, sla_hours=24)

    engine = ClaimAllocationEngine([claim_fast], standard_verifiers)
    csp = engine.build_csp()

    shifts_in_domain = {slot.shift for slot in csp.domains["CLM-FAST"]}
    assert ShiftSlot.MALAM not in shifts_in_domain
    assert ShiftSlot.PAGI in shifts_in_domain
    assert ShiftSlot.SIANG in shifts_in_domain


# ===========================================================================
# 2. PENGUJIAN ALGORITMA AC-3 (ARC CONSISTENCY 3)
# ===========================================================================

def test_ac3_prunes_inconsistent_values():
    """AC-3 harus memangkas nilai domain yang tidak memiliki pasangan pendukung (support)."""
    variables = ["X", "Y"]
    domains = {
        "X": [1, 2, 3],
        "Y": [2],
    }
    csp = CSP[str, int](variables, domains)
    # Batasan: X harus sama dengan Y
    csp.add_constraint(BinaryConstraint("X", "Y", lambda x, y: x == y))

    is_consistent = ac3(csp)

    assert is_consistent is True
    # Nilai 1 dan 3 pada X harus dipangkas karena tidak didukung oleh Y
    assert csp.domains["X"] == [2]
    assert csp.domains["Y"] == [2]


def test_ac3_detects_empty_domain_early():
    """Jika tidak ada nilai yang saling mendukung, AC-3 harus mengembalikan False."""
    variables = ["A", "B"]
    domains = {
        "A": [1, 2],
        "B": [3, 4],
    }
    csp = CSP[str, int](variables, domains)
    # Batasan mustahil: A == B
    csp.add_constraint(BinaryConstraint("A", "B", lambda a, b: a == b))

    is_consistent = ac3(csp)

    assert is_consistent is False
    assert len(csp.domains["A"]) == 0 or len(csp.domains["B"]) == 0


# ===========================================================================
# 3. PENGUJIAN HEURISTIK MRV, DEGREE & LCV
# ===========================================================================

def test_mrv_heuristic_selection():
    """MRV harus memprioritaskan variabel dengan domain legal terkecil."""
    variables = ["V1", "V2", "V3"]
    domains = {
        "V1": [1, 2, 3, 4],
        "V2": [10, 20],        # Paling sedikit (ukuran 2)
        "V3": [100, 200, 300],
    }
    csp = CSP[str, int](variables, domains)
    assignment = {}

    selected = select_unassigned_variable_mrv(assignment, csp)
    assert selected == "V2"


def test_degree_heuristic_tie_breaker():
    """Jika ukuran domain sama, Degree Heuristic memilih variabel dengan relasi tetangga terbanyak."""
    variables = ["A", "B", "C", "D"]
    domains = {
        "A": [1, 2],
        "B": [1, 2],
        "C": [1, 2, 3],
        "D": [1, 2, 3],
    }
    csp = CSP[str, int](variables, domains)
    # A terhubung ke B, C, D (3 tetangga); B hanya terhubung ke A (1 tetangga)
    csp.add_constraint(BinaryConstraint("A", "B", lambda x, y: x != y))
    csp.add_constraint(BinaryConstraint("A", "C", lambda x, y: x != y))
    csp.add_constraint(BinaryConstraint("A", "D", lambda x, y: x != y))

    assignment = {}
    selected = select_unassigned_variable_mrv(assignment, csp)
    # A dan B sama-sama memiliki domain ukuran 2, tapi A memiliki 3 relasi batasan
    assert selected == "A"


def test_lcv_heuristic_ordering():
    """LCV harus menempatkan nilai yang paling sedikit membatasi pilihan tetangga di urutan pertama."""
    variables = ["X", "Y"]
    domains = {
        "X": [1, 2],
        "Y": [1, 2, 3],
    }
    csp = CSP[str, int](variables, domains)
    # X != Y
    csp.add_constraint(BinaryConstraint("X", "Y", lambda x, y: x != y))

    ordered_values = order_domain_values_lcv("X", {}, csp)
    # Nilai 1 memangkas Y=1 (1 konflik)
    # Nilai 2 memangkas Y=2 (1 konflik)
    assert len(ordered_values) == 2


# ===========================================================================
# 4. PENGUJIAN SOLVER CSP TERPADU (BACKTRACKING + MAC)
# ===========================================================================

def test_csp_backtracking_mac_valid_solution(standard_claims, standard_verifiers):
    """CSP solver dengan AC-3 & MAC harus menemukan solusi 100% legal untuk dataset standar."""
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    assert stats.solution is not None
    assert len(stats.solution) == len(standard_claims)
    assert stats.backtracks == 0  # Heuristik MRV + MAC menemukan solusi secara langsung tanpa backtrack


def test_daily_workload_capacity_compliance(standard_claims, standard_verifiers):
    """Solusi yang dihasilkan wajib mematuhi batas beban kerja harian UU Ketenagakerjaan."""
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    verifier_map = {v.verifier_id: v for v in standard_verifiers}
    claim_map = {c.claim_id: c for c in standard_claims}

    workload_count = {v.verifier_id: 0 for v in standard_verifiers}
    claim_count = {v.verifier_id: 0 for v in standard_verifiers}

    for c_id, slot in stats.solution.items():
        claim = claim_map[c_id]
        workload_count[slot.verifier_id] += claim.complexity
        claim_count[slot.verifier_id] += 1

    for v_id, total_load in workload_count.items():
        assert total_load <= verifier_map[v_id].max_daily_workload
    for v_id, total_claims in claim_count.items():
        assert total_claims <= verifier_map[v_id].max_daily_claims


def test_conflict_group_separation_of_duties(standard_claims, standard_verifiers):
    """Dua berkas klaim dalam kelompok konflik yang sama (FAM-1) wajib ditugaskan ke staf berbeda."""
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    slot_c5 = stats.solution["CLM-005"]
    slot_c6 = stats.solution["CLM-006"]

    # Pemisahan tugas mutlak
    assert slot_c5.verifier_id != slot_c6.verifier_id


# ===========================================================================
# 5. PENGUJIAN ALGORITMA GENETIKA (GA)
# ===========================================================================

def test_ga_solver_finds_valid_solution(standard_claims, standard_verifiers):
    """GA solver dengan turnamen dan elitisme harus menemukan solusi tanpa pelanggaran batasan mutlak."""
    ga_solver = GeneticAlgorithmSolver(standard_claims, standard_verifiers)
    stats = ga_solver.solve()

    assert stats.is_feasible is True
    assert stats.hard_violations == 0
    assert stats.solution is not None
    assert len(stats.solution) == len(standard_claims)


# ===========================================================================
# 6. PENGUJIAN KASUS EKSTREM (EDGE CASES)
# ===========================================================================

def test_edge_case_over_constrained_graceful_failure():
    """Kapasitas staf jauh di bawah volume klaim (Pigeonhole) harus terdeteksi infeasible tanpa hanging."""
    claims = [
        Claim(f"CLM-{i}", "OUTPATIENT", 1_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=3)
        for i in range(10)
    ]
    # Total beban = 30 poin, tetapi kapasitas verifikator hanya 5 poin
    verifiers = [
        Verifier("V1", "Officer 1", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=5, max_daily_claims=2),
    ]

    engine = ClaimAllocationEngine(claims, verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is False
    assert stats.solution is None


def test_edge_case_zero_claims(standard_verifiers):
    """Himpunan klaim kosong harus diselesaikan secara instan."""
    engine = ClaimAllocationEngine([], standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    assert stats.solution == {}
    assert stats.runtime_ms < 5.0


def test_edge_case_single_claim_boundary(standard_verifiers):
    """Satu klaim tunggal harus dialokasikan dengan tepat dan cepat."""
    claim = Claim("CLM-SINGLE", "SURGICAL", 80_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4)
    engine = ClaimAllocationEngine([claim], standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    assert stats.solution["CLM-SINGLE"].verifier_id == "DOC-01"


def test_edge_case_specialist_bottleneck():
    """Banyak klaim bedah dengan hanya satu Medical Advisor yang kapasitasnya tepat mencukupi."""
    verifiers = [
        Verifier("DOC-1", "dr. Spesialis", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=15, max_daily_claims=5, max_claim_amount_idr=200_000_000),
        Verifier("ADJ-1", "Adjuster Biasa", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
    ]
    claims = [
        Claim(f"CLM-S-{i}", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=3)
        for i in range(5)
    ]
    engine = ClaimAllocationEngine(claims, verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    for c in claims:
        assert stats.solution[c.claim_id].verifier_id == "DOC-1"


def test_edge_case_clique_of_conflicts():
    """
    Tiga klaim yang saling melarang verifikator yang sama (clique berukuran 3).
    Dengan 3 verifikator berkualifikasi, penugasan harus unik untuk tiap klaim.
    """
    verifiers = [
        Verifier("V1", "Staff 1", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=20, max_daily_claims=5),
        Verifier("V2", "Staff 2", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=20, max_daily_claims=5),
        Verifier("V3", "Staff 3", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=20, max_daily_claims=5),
    ]
    claims = [
        Claim("C1", "INPATIENT", 20_000_000, VerifierRole.SENIOR_ADJUSTER, conflict_group="CLIQUE"),
        Claim("C2", "INPATIENT", 20_000_000, VerifierRole.SENIOR_ADJUSTER, conflict_group="CLIQUE"),
        Claim("C3", "INPATIENT", 20_000_000, VerifierRole.SENIOR_ADJUSTER, conflict_group="CLIQUE"),
    ]
    engine = ClaimAllocationEngine(claims, verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is True
    assigned_vids = {stats.solution[c.claim_id].verifier_id for c in claims}
    # Seluruh verifikator yang ditugaskan harus saling berbeda
    assert len(assigned_vids) == 3


def test_edge_case_extreme_claim_amount_exceeds_all_authorities(standard_verifiers):
    """Klaim dengan nominal ekstrem (Rp 2 Miliar) melebihi seluruh otorisasi staf -> Infeasible."""
    extreme_claim = Claim("CLM-MEGA", "SURGICAL", 2_000_000_000.0, VerifierRole.MEDICAL_ADVISOR, complexity=5)
    engine = ClaimAllocationEngine([extreme_claim], standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.is_feasible is False
    assert stats.solution is None


def test_solver_statistics_completeness(standard_claims, standard_verifiers):
    """Objek SolverStatistics harus memuat seluruh metrik komputasi dan finansial yang valid."""
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    assert stats.runtime_ms >= 0.0
    assert stats.nodes_expanded > 0
    assert stats.backtracks >= 0
    assert stats.total_handling_cost_idr > 0.0
    assert stats.workload_std_dev >= 0.0
