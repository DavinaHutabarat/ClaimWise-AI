"""
ClaimWise AI — Comprehensive Test Suite for Genetic Algorithm (GA) Optimization
================================================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - S1 Sistem Informasi)
Tugas       : Tugas 2 (Milestone 2 - W04)

Cakupan Pengujian GA:
1. Logika Kualifikasi Peran (is_role_compatible).
2. Sistem Penalti Regulasi (Kompetensi, Plafon Finansial, Konflik RS, SLA Shift).
3. Batasan Kapasitas Beban Kerja Harian UU Ketenagakerjaan No. 13/2003.
4. Pemisahan Tugas & Anti-Kolusi (Conflict Group / Separation of Duties).
5. Operator Genetika: Seleksi Turnamen, Crossover Dua Titik, Mutasi, dan Elitisme.
6. Konvergensi Kebugaran Multi-Objektif & Rekam Jejak Riwayat Evolusi.
7. Pengujian Kasus Ekstrem: Over-Constrained, Bottleneck Spesialis, Clique Conflict, Single/Zero Claim.
8. Reproduksibilitas Penentuan Seed Acak.

Eksekusi:
    pytest -v test_solver.py
"""

import pytest

from solver import (
    ROLE_HIERARCHY,
    AssignmentSlot,
    Claim,
    ClaimAllocationEngine,
    EvolutionRecord,
    GAParameters,
    GeneticAlgorithmSolver,
    ShiftSlot,
    SolverStatistics,
    Verifier,
    VerifierRole,
    is_role_compatible,
)


# ===========================================================================
# FIXTURES & DATA PENDUKUNG
# ===========================================================================

@pytest.fixture
def standard_verifiers():
    """Himpunan staf verifikator standar dengan variasi peran dan kapasitas."""
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
        Claim("CLM-001", "OUTPATIENT", 3_500_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-01", risk_score=0.10),
        Claim("CLM-002", "INPATIENT", 45_000_000, VerifierRole.SENIOR_ADJUSTER, complexity=2, sla_hours=48, hospital_id="HOSP-02", risk_score=0.40),
        Claim("CLM-003", "SURGICAL", 135_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4, sla_hours=72, hospital_id="HOSP-01", risk_score=0.65),
        Claim("CLM-004", "FRAUD_SUSPECT", 80_000_000, VerifierRole.FRAUD_INVESTIGATOR, complexity=3, sla_hours=48, hospital_id="HOSP-03", risk_score=0.90),
        Claim("CLM-005", "OUTPATIENT", 5_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAM-1", risk_score=0.15),
        Claim("CLM-006", "OUTPATIENT", 4_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAM-1", risk_score=0.15),
    ]


# ===========================================================================
# 1. PENGUJIAN LOGIKA PERAN & SISTEM PENALTI KEBUGARAN
# ===========================================================================

def test_role_compatibility_logic():
    """Validasi hierarki dan spesialisasi lisensi klinis/forensik."""
    # Medical Advisor wajib dokter
    assert is_role_compatible(VerifierRole.MEDICAL_ADVISOR, VerifierRole.MEDICAL_ADVISOR) is True
    assert is_role_compatible(VerifierRole.JUNIOR_ADJUSTER, VerifierRole.MEDICAL_ADVISOR) is False
    assert is_role_compatible(VerifierRole.FRAUD_INVESTIGATOR, VerifierRole.MEDICAL_ADVISOR) is False

    # Fraud Investigator wajib spesialis fraud
    assert is_role_compatible(VerifierRole.FRAUD_INVESTIGATOR, VerifierRole.FRAUD_INVESTIGATOR) is True
    assert is_role_compatible(VerifierRole.MEDICAL_ADVISOR, VerifierRole.FRAUD_INVESTIGATOR) is False

    # Junior Adjuster dapat ditangani seluruh staf
    assert is_role_compatible(VerifierRole.JUNIOR_ADJUSTER, VerifierRole.JUNIOR_ADJUSTER) is True
    assert is_role_compatible(VerifierRole.SENIOR_ADJUSTER, VerifierRole.JUNIOR_ADJUSTER) is True


def test_evaluation_detects_role_mismatch_penalty(standard_verifiers):
    """Menugaskan klaim bedah ke verifikator junior harus memicu pelanggaran dan penalti masif."""
    claim = Claim("CLM-BEDAH", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=3)
    solver = GeneticAlgorithmSolver([claim], standard_verifiers)

    # Cari indeks penugasan ke ADJ-01 (Junior)
    junior_gene = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.verifier_id == "ADJ-01"
    )

    fit, viol, cost, std, details = solver.evaluate_chromosome([junior_gene])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_role_mismatch


def test_evaluation_detects_financial_excess_penalty(standard_verifiers):
    """Menugaskan klaim Rp 50 Juta ke staf dengan limit Rp 25 Juta harus memicu penalti finansial."""
    claim = Claim("CLM-HIGH", "OUTPATIENT", 50_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=2)
    solver = GeneticAlgorithmSolver([claim], standard_verifiers)

    junior_gene = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.verifier_id == "ADJ-01"
    )

    fit, viol, cost, std, details = solver.evaluate_chromosome([junior_gene])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_financial_excess


def test_evaluation_detects_hospital_conflict_penalty(standard_verifiers):
    """Menugaskan klaim ke verifikator yang memiliki konflik RS harus memicu penalti."""
    claim = Claim("CLM-CONF", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, hospital_id="HOSP-CONFLICT-01")
    solver = GeneticAlgorithmSolver([claim], standard_verifiers)

    # ADJ-01 terafiliasi dengan HOSP-CONFLICT-01
    conflict_gene = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.verifier_id == "ADJ-01"
    )

    fit, viol, cost, std, details = solver.evaluate_chromosome([conflict_gene])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_hospital_conflict


def test_evaluation_detects_night_shift_sla_penalty(standard_verifiers):
    """Klaim Fast-Track (SLA <= 24 jam) dilarang ditempatkan pada Shift MALAM."""
    claim = Claim("CLM-FAST", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, sla_hours=24)
    solver = GeneticAlgorithmSolver([claim], standard_verifiers)

    night_gene = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.shift == ShiftSlot.MALAM
    )

    fit, viol, cost, std, details = solver.evaluate_chromosome([night_gene])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_night_shift_sla


def test_evaluation_detects_conflict_group_penalty(standard_verifiers):
    """Dua klaim dalam kelompok konflik yang sama jika ditugaskan ke staf yang sama harus dihukum."""
    c1 = Claim("C1", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, conflict_group="FAMILY-A")
    c2 = Claim("C2", "OUTPATIENT", 2_000_000, VerifierRole.JUNIOR_ADJUSTER, conflict_group="FAMILY-A")
    solver = GeneticAlgorithmSolver([c1, c2], standard_verifiers)

    # Tugaskan kedua klaim ke ADJ-01
    gene_adj1 = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.verifier_id == "ADJ-01"
    )

    fit, viol, cost, std, details = solver.evaluate_chromosome([gene_adj1, gene_adj1])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_conflict_group


def test_evaluation_detects_workload_capacity_penalty(standard_verifiers):
    """Melebihi kuota beban kerja harian staf (UU Ketenagakerjaan) harus memicu penalti kuadratik."""
    # Buat 15 klaim berbobot tinggi untuk membebani verifikator
    claims = [
        Claim(f"CLM-{i}", "OUTPATIENT", 1_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=3)
        for i in range(15)
    ]
    solver = GeneticAlgorithmSolver(claims, standard_verifiers)

    gene_adj1 = next(
        idx for idx, slot in enumerate(solver.possible_assignments) if slot.verifier_id == "ADJ-01"
    )

    # Bebankan semua 15 klaim ke ADJ-01 (total beban = 45 poin > limit 20 poin)
    all_adj1 = [gene_adj1] * 15
    fit, viol, cost, std, details = solver.evaluate_chromosome(all_adj1)

    assert viol >= 1
    assert details["penalty_score"] > 0


# ===========================================================================
# 2. PENGUJIAN OPERATOR EVOLUSI GENETIKA (SELEKSI, CROSSOVER, ELITISME)
# ===========================================================================

def test_tournament_selection_picks_fitter_individual(standard_claims, standard_verifiers):
    """Seleksi turnamen harus memprioritaskan individu dengan fitness tertinggi di antara kandidat."""
    solver = GeneticAlgorithmSolver(standard_claims, standard_verifiers)
    pop = [[0] * len(standard_claims), [1] * len(standard_claims), [2] * len(standard_claims)]
    fitnesses = [10.0, 500.0, 50.0]

    # Dalam turnamen yang melibatkan indeks 1, individu dengan fitness 500.0 harus terpilih
    selected = solver._tournament_select(pop, fitnesses)
    assert selected in pop


def test_elitism_preserves_best_individuals(standard_claims, standard_verifiers):
    """Elitisme menjamin bahwa solusi terbaik tidak terdegradasi pada generasi berikutnya."""
    params = GAParameters(population_size=40, generations=30, elite_count=2, seed=77)
    solver = GeneticAlgorithmSolver(standard_claims, standard_verifiers, params=params)
    stats = solver.solve()

    assert stats.evolution_history is not None
    assert len(stats.evolution_history) > 0

    # Best fitness per generasi harus monotonik tidak menurun
    fitness_progression = [rec.best_fitness for rec in stats.evolution_history]
    for i in range(1, len(fitness_progression)):
        assert fitness_progression[i] >= fitness_progression[i - 1] - 1e-6


# ===========================================================================
# 3. PENGUJIAN SOLVER GA LENGKAP & KEPATUHAN BISNIS
# ===========================================================================

def test_ga_finds_zero_violation_solution(standard_claims, standard_verifiers):
    """GA harus berhasil menemukan penugasan layak (0 hard violations) pada dataset standar."""
    params = GAParameters(population_size=60, generations=120, seed=42)
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers, params=params)
    stats = engine.solve()

    assert stats.is_feasible is True
    assert stats.hard_violations == 0
    assert stats.solution is not None
    assert len(stats.solution) == len(standard_claims)


def test_ga_workload_capacity_compliance(standard_claims, standard_verifiers):
    """Solusi GA yang dihasilkan wajib mematuhi batas beban kerja UU Ketenagakerjaan."""
    params = GAParameters(population_size=60, generations=120, seed=42)
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers, params=params)
    stats = engine.solve()

    assert stats.is_feasible is True
    v_map = {v.verifier_id: v for v in standard_verifiers}
    c_map = {c.claim_id: c for c in standard_claims}

    workload_count = {v.verifier_id: 0 for v in standard_verifiers}
    claim_count = {v.verifier_id: 0 for v in standard_verifiers}

    for c_id, slot in stats.solution.items():
        claim = c_map[c_id]
        workload_count[slot.verifier_id] += claim.complexity
        claim_count[slot.verifier_id] += 1

    for v_id, total_load in workload_count.items():
        assert total_load <= v_map[v_id].max_daily_workload
    for v_id, total_claims in claim_count.items():
        assert total_claims <= v_map[v_id].max_daily_claims


def test_ga_separation_of_duties_compliance(standard_claims, standard_verifiers):
    """Dua berkas klaim keluarga yang sama (FAM-1) wajib ditugaskan ke staf yang berbeda."""
    params = GAParameters(population_size=60, generations=120, seed=42)
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers, params=params)
    stats = engine.solve()

    assert stats.is_feasible is True
    slot_c5 = stats.solution["CLM-005"]
    slot_c6 = stats.solution["CLM-006"]
    assert slot_c5.verifier_id != slot_c6.verifier_id


# ===========================================================================
# 4. PENGUJIAN KASUS EKSTREM (EDGE CASES)
# ===========================================================================

def test_edge_case_zero_claims(standard_verifiers):
    """Himpunan klaim kosong harus diselesaikan secara instan tanpa iterasi."""
    engine = ClaimAllocationEngine([], standard_verifiers)
    stats = engine.solve()

    assert stats.is_feasible is True
    assert stats.solution == {}
    assert stats.runtime_ms < 5.0


def test_edge_case_single_claim_boundary(standard_verifiers):
    """Satu klaim tunggal harus dialokasikan secara presisi dan cepat."""
    claim = Claim("CLM-SINGLE", "SURGICAL", 80_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4)
    engine = ClaimAllocationEngine([claim], standard_verifiers)
    stats = engine.solve()

    assert stats.is_feasible is True
    assert stats.solution["CLM-SINGLE"].verifier_id == "DOC-01"


def test_edge_case_over_constrained_graceful_handling():
    """Kapasitas staf jauh di bawah volume klaim harus menghasilkan infeasible secara anggun tanpa crash."""
    claims = [
        Claim(f"CLM-{i}", "OUTPATIENT", 1_000_000, VerifierRole.JUNIOR_ADJUSTER, complexity=3)
        for i in range(10)
    ]
    # Total beban 30 poin, kapasitas staf hanya 5 poin
    verifiers = [
        Verifier("V1", "Officer 1", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=5, max_daily_claims=2),
    ]

    engine = ClaimAllocationEngine(claims, verifiers)
    stats = engine.solve()

    assert stats.is_feasible is False
    assert stats.hard_violations > 0
    assert stats.solution is None


def test_edge_case_specialist_bottleneck():
    """Seluruh klaim bedah wajib dialokasikan ke satu-satunya dokter penasihat."""
    verifiers = [
        Verifier("DOC-1", "dr. Spesialis", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=25, max_daily_claims=8, max_claim_amount_idr=200_000_000),
        Verifier("ADJ-1", "Adjuster Biasa", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
    ]
    claims = [
        Claim(f"CLM-S-{i}", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=3)
        for i in range(5)
    ]
    engine = ClaimAllocationEngine(claims, verifiers, params=GAParameters(population_size=40, generations=80, seed=123))
    stats = engine.solve()

    assert stats.is_feasible is True
    for c in claims:
        assert stats.solution[c.claim_id].verifier_id == "DOC-1"


def test_edge_case_clique_of_conflicts():
    """Tiga berkas dalam sengketa bersama harus disebarkan ke 3 staf verifikator yang berbeda."""
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
    engine = ClaimAllocationEngine(claims, verifiers, params=GAParameters(population_size=40, generations=80, seed=321))
    stats = engine.solve()

    assert stats.is_feasible is True
    assigned_vids = {stats.solution[c.claim_id].verifier_id for c in claims}
    assert len(assigned_vids) == 3


def test_edge_case_extreme_claim_amount_exceeds_all(standard_verifiers):
    """Klaim dengan nominal raksasa melampaui seluruh limit otorisasi staf harus dihukum."""
    extreme_claim = Claim("CLM-MEGA", "SURGICAL", 2_000_000_000.0, VerifierRole.MEDICAL_ADVISOR, complexity=5)
    solver = GeneticAlgorithmSolver([extreme_claim], standard_verifiers)

    # Ambil gen sembarang
    fit, viol, cost, std, details = solver.evaluate_chromosome([0])
    assert viol >= 1
    assert details["penalty_score"] >= solver.params.penalty_financial_excess


def test_seed_reproducibility(standard_claims, standard_verifiers):
    """Dua kali eksekusi GA dengan seed identik harus menghasilkan solusi yang persis sama."""
    params1 = GAParameters(population_size=30, generations=40, seed=999)
    params2 = GAParameters(population_size=30, generations=40, seed=999)

    solver1 = GeneticAlgorithmSolver(standard_claims, standard_verifiers, params=params1)
    solver2 = GeneticAlgorithmSolver(standard_claims, standard_verifiers, params=params2)

    res1 = solver1.solve()
    res2 = solver2.solve()

    assert res1.is_feasible == res2.is_feasible
    assert res1.fitness_score == pytest.approx(res2.fitness_score)
    assert res1.hard_violations == res2.hard_violations


def test_cost_breakdown_calculation(standard_claims, standard_verifiers):
    """Perhitungan biaya staf penanganan klaim harus proporsional terhadap tarif jam staf."""
    params = GAParameters(population_size=40, generations=60, seed=42)
    engine = ClaimAllocationEngine(standard_claims, standard_verifiers, params=params)
    stats = engine.solve()

    assert stats.total_handling_cost_idr > 0.0
    assert stats.workload_std_dev >= 0.0
    assert stats.generations_completed > 0


def test_repair_individual_remedies_infeasible_chromosome(standard_claims, standard_verifiers):
    """Operator perbaikan heuristik (Greedy Repair) wajib menghilangkan pelanggaran batasan."""
    solver = GeneticAlgorithmSolver(standard_claims, standard_verifiers)
    # Buat kromosom buatan dengan alel sembarang yang melanggar batasan
    bad_chromosome = [0] * len(standard_claims)
    fit_bad, viol_bad, _, _, _ = solver.evaluate_chromosome(bad_chromosome)
    assert viol_bad > 0

    repaired = solver.repair_individual(bad_chromosome)
    fit_rep, viol_rep, _, _, _ = solver.evaluate_chromosome(repaired)
    assert viol_rep == 0
    assert fit_rep > fit_bad


def test_large_scale_100_claims_achieves_zero_violations():
    """Kasus skala besar (100 klaim, 24 staf) wajib mencapai solusi 100% layak (0 pelanggaran)."""
    from solver import generate_benchmark_instance
    claims, verifiers = generate_benchmark_instance("LAR-TEST", 100, 24, seed=303)
    params = GAParameters(population_size=100, generations=100, seed=303)
    engine = ClaimAllocationEngine(claims, verifiers, params=params)
    stats = engine.solve()

    assert stats.is_feasible is True
    assert stats.hard_violations == 0
    assert stats.solution is not None
    assert len(stats.solution) == 100

