"""
ClaimWise AI — Business Constraint Solver Module (Genetic Algorithm Optimization)
==================================================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - Sarjana Sistem Informasi)
Tugas       : Tugas 2 (Milestone 2 - W04) — Milestone Proyek Terpadu (PjBL)

Deskripsi Modul:
----------------
Modul ini mengimplementasikan mesin optimasi komputasional cerdas berbasis
**Algoritma Genetika (Genetic Algorithm - GA)** untuk sub-masalah keputusan bisnis:
"Optimasi Penjadwalan Shift & Alokasi Penugasan Berkas Verifikator Klaim Asuransi
Kesehatan Berbasis Trade-Off Biaya Staf, Waktu SLA, Risiko Fraud, dan Regulasi."

Domain Masalah & Justifikasi Pemilihan GA:
------------------------------------------
Pada ClaimWise AI, alokasi berkas klaim bukan sekadar memenuhi batasan kaku (hard
satisfaction), melainkan merupakan persoalan optimasi kombinatorik multi-objektif:
1. Trade-off antara Biaya Penanganan (Handling Cost verifikator ahli vs junior).
2. Percepatan Waktu SLA Klaim (POJK No. 69/POJK.05/2016).
3. Mitigasi Risiko Kebocoran Fraud (Fraud Leakage) berdasarkan skor Machine Learning.
4. Keadilan Beban Kerja Staf (Workload Fairness) sesuai UU Ketenagakerjaan No. 13/2003.
5. Penegakan batasan hukum ketat melalui Sistem Penalti Kebugaran Bertingkat.

Komponen Utama:
1. Skema Representasi Kromosom Diskret Penugasan Berkas.
2. Fungsi Kebugaran Multi-Objektif Terbobot dengan Sistem Penalti Masif.
3. Operator Evolusioner:
   - Tournament Selection (k=3)
   - Two-Point Crossover (pc = 0.85)
   - Adaptive Mutation (pm = 0.08)
   - Elitism (Top-E individu dipertahankan utuh)
4. Mesin Analisis Sensitivitas & Tolok Ukur (Benchmark):
   - Pengujian terhadap variasi skala masalah (kecil, sedang, besar).
   - Pengujian parameter genetika (ukuran populasi & tingkat mutasi).
   - Pengujian kasus ekstrem (over-constrained, specialist bottleneck, conflict clique).
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ===========================================================================
# 1. DOMAIN ENUMS & DATA MODELS
# ===========================================================================

class VerifierRole(str, Enum):
    """Tingkat kompetensi dan kualifikasi lisensi verifikator klaim asuransi."""
    JUNIOR_ADJUSTER = "JUNIOR_ADJUSTER"      # Verifikator dasar (klaim sederhana < Rp 25 Juta)
    SENIOR_ADJUSTER = "SENIOR_ADJUSTER"      # Verifikator senior (klaim moderat < Rp 75 Juta)
    MEDICAL_ADVISOR = "MEDICAL_ADVISOR"      # Dokter penasihat medis (tindakan bedah, rawat inap kompleks)
    FRAUD_INVESTIGATOR = "FRAUD_INVESTIGATOR"# Auditor forensik anti-fraud (klaim anomali/sengketa)


ROLE_HIERARCHY: Dict[VerifierRole, int] = {
    VerifierRole.JUNIOR_ADJUSTER: 1,
    VerifierRole.SENIOR_ADJUSTER: 2,
    VerifierRole.MEDICAL_ADVISOR: 3,
    VerifierRole.FRAUD_INVESTIGATOR: 4,
}


def is_role_compatible(verifier_role: VerifierRole, required_role: VerifierRole) -> bool:
    """
    Memvalidasi kesesuaian kompetensi dan lisensi klinis/forensik staf:
    - Klaim MEDICAL_ADVISOR (bedah/kritis) wajib dokter (hanya MEDICAL_ADVISOR).
    - Klaim FRAUD_INVESTIGATOR wajib auditor forensik (hanya FRAUD_INVESTIGATOR).
    - Klaim SENIOR_ADJUSTER dapat ditangani Senior Adjuster, Medical Advisor, atau Fraud Investigator.
    - Klaim JUNIOR_ADJUSTER dapat ditangani oleh seluruh staf (jika batasan nominal terpenuhi).
    """
    if required_role == VerifierRole.MEDICAL_ADVISOR:
        return verifier_role == VerifierRole.MEDICAL_ADVISOR
    if required_role == VerifierRole.FRAUD_INVESTIGATOR:
        return verifier_role == VerifierRole.FRAUD_INVESTIGATOR
    if required_role == VerifierRole.SENIOR_ADJUSTER:
        return verifier_role in (
            VerifierRole.SENIOR_ADJUSTER,
            VerifierRole.MEDICAL_ADVISOR,
            VerifierRole.FRAUD_INVESTIGATOR,
        )
    return True


class ShiftSlot(str, Enum):
    """Jendela shift kerja operasional harian."""
    PAGI = "PAGI"      # 08:00 - 16:00 (Kapasitas penuh, otorisasi supervisi & perbankan aktif)
    SIANG = "SIANG"    # 14:00 - 22:00 (Triase intensif & verifikasi standar)
    MALAM = "MALAM"    # 22:00 - 06:00 (Intake darurat & antrean otomatis; supervisi offline)


@dataclass(frozen=True)
class Claim:
    """Representasi berkas klaim asuransi kesehatan dalam antrean harian."""
    claim_id: str
    claim_type: str                  # "OUTPATIENT", "INPATIENT", "SURGICAL", "FRAUD_SUSPECT"
    amount_idr: float                # Nilai nominal klaim dalam Rupiah
    required_role: VerifierRole      # Kompetensi minimum yang dipersyaratkan
    complexity: int = 1              # Bobot kompleksitas kognitif (1 - 5 poin)
    sla_hours: int = 24              # Batas waktu maksimal penyelesaian SLA (Jam)
    hospital_id: str = "HOSP-01"     # Rumah sakit penyedia pelayanan medis
    conflict_group: Optional[str] = None  # ID kelompok sengketa jika ada relasi keluarga/afiliasi
    risk_score: float = 0.10         # Skor risiko hasil prediksi model ML (0.0 - 1.0)


@dataclass(frozen=True)
class Verifier:
    """Representasi staf verifikator asuransi dengan kuota legal terikat regulasi."""
    verifier_id: str
    name: str
    role: VerifierRole
    max_daily_workload: int = 20     # Maksimal poin kompleksitas per hari (UU Ketenagakerjaan No. 13/2003)
    max_daily_claims: int = 12       # Maksimal volume berkas klaim per hari
    max_claim_amount_idr: float = 25_000_000.0  # Plafon otorisasi wewenang finansial
    affiliated_hospitals: FrozenSet[str] = field(default_factory=frozenset)  # Konflik kepentingan RS
    shift_preference: Optional[ShiftSlot] = None
    hourly_rate_idr: float = 75_000.0 # Tarif kompensasi operasional per jam


@dataclass(frozen=True)
class AssignmentSlot:
    """Representasi gen: Pasangan staf verifikator dan jendela shift kerja."""
    verifier_id: str
    shift: ShiftSlot

    def __repr__(self) -> str:
        return f"({self.verifier_id}@{self.shift.value})"


@dataclass
class EvolutionRecord:
    """Catatan riwayat metrik pada satu generasi evolusi."""
    generation: int
    best_fitness: float
    avg_fitness: float
    hard_violations: int
    handling_cost_idr: float
    workload_std_dev: float


@dataclass
class SolverStatistics:
    """Statistik komputasi dan hasil evaluasi solusi optimasi GA."""
    algorithm: str
    is_feasible: bool = False
    runtime_ms: float = 0.0
    generations_completed: int = 0
    nodes_expanded: int = 0          # Total evaluasi individu (pop_size * gen)
    backtracks: int = 0              # 0 untuk GA (stokastik non-backtracking)
    hard_violations: int = 0
    total_handling_cost_idr: float = 0.0
    workload_std_dev: float = 0.0
    fitness_score: float = 0.0
    solution: Optional[Dict[str, AssignmentSlot]] = None
    evolution_history: List[EvolutionRecord] = field(default_factory=list)


# ===========================================================================
# 2. PARAMETER & KONFIGURASI ALGORITMA GENETIKA (GA)
# ===========================================================================

@dataclass
class GAParameters:
    """Parameter konfigurasi loop evolusi Algoritma Genetika."""
    population_size: int = 60
    generations: int = 150
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.08
    elite_count: int = 2
    seed: Optional[int] = 42

    # Bobot Penalti Batasan Regulasi Mutlak (Hard Penalties)
    penalty_role_mismatch: float = 25_000.0
    penalty_financial_excess: float = 20_000.0
    penalty_hospital_conflict: float = 25_000.0
    penalty_night_shift_sla: float = 15_000.0
    penalty_conflict_group: float = 20_000.0
    penalty_capacity_excess: float = 10_000.0

    # Bobot Trade-Off Multi-Objektif Operasional (Soft Objectives)
    weight_handling_cost: float = 0.001   # Normalisasi skala biaya IDR
    weight_workload_balance: float = 15.0 # Mendorong keadilan beban kerja
    weight_fraud_exposure: float = 50.0   # Mendorong penugasan risiko tinggi ke ahli


# ===========================================================================
# 3. GENETIC ALGORITHM (GA) OPTIMIZATION ENGINE
# ===========================================================================

class GeneticAlgorithmSolver:
    """
    Mesin inferensi optimasi berbasis Algoritma Genetika dengan operator
    Tournament Selection, Two-Point Crossover, Adaptive Mutation, dan Elitism.
    """

    def __init__(
        self,
        claims: Sequence[Claim],
        verifiers: Sequence[Verifier],
        shifts: Sequence[ShiftSlot] = (ShiftSlot.PAGI, ShiftSlot.SIANG, ShiftSlot.MALAM),
        params: Optional[GAParameters] = None,
    ) -> None:
        self.claims = list(claims)
        self.verifiers = list(verifiers)
        self.shifts = list(shifts)
        self.params = params or GAParameters()
        if self.params.seed is not None:
            random.seed(self.params.seed)

        # Ruang penugasan diskret (kandidat alel untuk tiap gen)
        self.possible_assignments: List[AssignmentSlot] = [
            AssignmentSlot(v.verifier_id, s)
            for v in self.verifiers
            for s in self.shifts
        ]
        self.verifier_map = {v.verifier_id: v for v in self.verifiers}
        self.claim_map = {c.claim_id: c for c in self.claims}

    def evaluate_chromosome(
        self, chromosome: List[int]
    ) -> Tuple[float, int, float, float, Dict[str, Any]]:
        """
        Evaluasi kebugaran kromosom secara multi-objektif dengan sistem penalti bertingkat:
        
        Returns:
            (fitness_score, hard_violations, total_cost, workload_std_dev, details)
        """
        hard_violations = 0
        penalty_score = 0.0
        total_handling_cost = 0.0
        total_fraud_exposure = 0.0

        verifier_workload: Dict[str, int] = defaultdict(int)
        verifier_claims: Dict[str, int] = defaultdict(int)
        conflict_tracker: Dict[Tuple[str, str], str] = {}  # (group_id, verifier_id) -> claim_id

        for claim_idx, gene in enumerate(chromosome):
            claim = self.claims[claim_idx]
            slot = self.possible_assignments[gene]
            verifier = self.verifier_map[slot.verifier_id]

            # 1. Batasan Kualifikasi Kompetensi & Lisensi (Role Compatibility)
            if not is_role_compatible(verifier.role, claim.required_role):
                hard_violations += 1
                penalty_score += self.params.penalty_role_mismatch

            # 2. Batasan Wewenang Finansial Nominal (Delegation of Authority)
            if claim.amount_idr > verifier.max_claim_amount_idr:
                hard_violations += 1
                penalty_score += self.params.penalty_financial_excess

            # 3. Batasan Pencegahan Konflik Kepentingan RS Rekanan
            if claim.hospital_id in verifier.affiliated_hospitals:
                hard_violations += 1
                penalty_score += self.params.penalty_hospital_conflict

            # 4. Batasan Kepatuhan SLA Cepat & Jendela Shift (POJK No. 69/2016)
            if slot.shift == ShiftSlot.MALAM and claim.sla_hours <= 24:
                hard_violations += 1
                penalty_score += self.params.penalty_night_shift_sla

            # 5. Batasan Pemisahan Tugas Sengketa (Separation of Duties / Conflict Group)
            if claim.conflict_group:
                key = (claim.conflict_group, slot.verifier_id)
                if key in conflict_tracker:
                    hard_violations += 1
                    penalty_score += self.params.penalty_conflict_group
                conflict_tracker[key] = claim.claim_id

            # Akumulasi Beban Kerja & Biaya Operasional
            verifier_workload[verifier.verifier_id] += claim.complexity
            verifier_claims[verifier.verifier_id] += 1

            # Biaya staf = tarif_per_jam * (kompleksitas * 0.25 jam)
            processing_hours = claim.complexity * 0.25
            total_handling_cost += verifier.hourly_rate_idr * processing_hours

            # Risiko fraud: jika risiko klaim tinggi tetapi adjuster masih junior
            if claim.risk_score >= 0.65 and verifier.role == VerifierRole.JUNIOR_ADJUSTER:
                total_fraud_exposure += claim.risk_score * 100.0

        # 6. Batasan Kapasitas Beban Kerja UU Ketenagakerjaan No. 13/2003
        for v_id, workload in verifier_workload.items():
            max_wl = self.verifier_map[v_id].max_daily_workload
            if workload > max_wl:
                excess = workload - max_wl
                hard_violations += excess
                # Penalti kuadratik untuk pelanggaran jam kerja tenaga kerja
                penalty_score += self.params.penalty_capacity_excess * (excess ** 2)

        for v_id, c_count in verifier_claims.items():
            max_cl = self.verifier_map[v_id].max_daily_claims
            if c_count > max_cl:
                excess_cl = c_count - max_cl
                hard_violations += excess_cl
                penalty_score += self.params.penalty_capacity_excess * (excess_cl ** 2)

        # 7. Keadilan Beban Kerja (Workload Fairness / Anti-Burnout)
        loads = [verifier_workload[v.verifier_id] for v in self.verifiers]
        mean_load = sum(loads) / max(1, len(loads))
        variance = sum((x - mean_load) ** 2 for x in loads) / max(1, len(loads))
        std_dev = math.sqrt(variance)

        # Formulasi Fungsi Kebugaran Multi-Objektif
        # f(g) = 1.000.000 / (1.0 + Penalty + CostTerm + BalanceTerm + FraudRiskTerm)
        cost_term = total_handling_cost * self.params.weight_handling_cost
        balance_term = std_dev * self.params.weight_workload_balance
        fraud_term = total_fraud_exposure * self.params.weight_fraud_exposure

        total_denominator = 1.0 + penalty_score + cost_term + balance_term + fraud_term
        fitness = 1_000_000.0 / total_denominator

        details = {
            "penalty_score": penalty_score,
            "cost_term": cost_term,
            "balance_term": balance_term,
            "fraud_term": fraud_term,
            "workload_distribution": dict(verifier_workload),
        }
        return fitness, hard_violations, total_handling_cost, std_dev, details

    def solve(self) -> SolverStatistics:
        """
        Menjalankan siklus evolusi Algoritma Genetika lengkap.
        
        Returns:
            SolverStatistics: Metrik komputasi, riwayat evolusi, dan penugasan optimal.
        """
        start_time = time.perf_counter()
        n_claims = len(self.claims)
        n_genes = len(self.possible_assignments)

        # Kasus Batas: Nol klaim
        if n_claims == 0:
            return SolverStatistics(
                algorithm="Genetic Algorithm",
                is_feasible=True,
                runtime_ms=0.0,
                generations_completed=0,
                hard_violations=0,
                solution={},
            )

        # 1. Pembangkitan Populasi Awal Cerdas (Heuristic Seeding + Random Diversity)
        population: List[List[int]] = self._generate_initial_population(n_claims, n_genes)

        best_individual: List[int] = population[0]
        best_fitness = -1.0
        best_violations = 999999
        best_cost = 0.0
        best_std_dev = 0.0
        history: List[EvolutionRecord] = []
        gen_completed = 0

        # 2. Siklus Evolusi Generasi
        for gen in range(self.params.generations):
            gen_completed = gen + 1
            evaluations = [self.evaluate_chromosome(ind) for ind in population]
            fitness_scores = [ev[0] for ev in evaluations]

            # Pencatatan Statistik Generasi Saat Ini
            avg_fit = sum(fitness_scores) / len(fitness_scores)
            cur_best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            cur_best_fit = fitness_scores[cur_best_idx]
            cur_best_viol = evaluations[cur_best_idx][1]
            cur_best_cost = evaluations[cur_best_idx][2]
            cur_best_std = evaluations[cur_best_idx][3]

            if cur_best_fit > best_fitness:
                best_fitness = cur_best_fit
                best_individual = population[cur_best_idx][:]
                best_violations = cur_best_viol
                best_cost = cur_best_cost
                best_std_dev = cur_best_std

            history.append(
                EvolutionRecord(
                    generation=gen_completed,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fit,
                    hard_violations=best_violations,
                    handling_cost_idr=best_cost,
                    workload_std_dev=best_std_dev,
                )
            )

            # Terminasi Dini jika Solusi Layak (0 Pelanggaran) dan Konvergensi Tercapai
            if best_violations == 0 and gen >= 45:
                # Periksa apakah fitness telah stabil selama 15 generasi terakhir
                recent_fits = [rec.best_fitness for rec in history[-15:]]
                if max(recent_fits) - min(recent_fits) < 1e-4:
                    break

            # 3. Elitisme: Menjaga E Individu Terbaik Tanpa Gangguan Mutasi
            sorted_indices = sorted(
                range(len(population)), key=lambda i: fitness_scores[i], reverse=True
            )
            new_population: List[List[int]] = [
                population[sorted_indices[e]][:] for e in range(self.params.elite_count)
            ]

            # 4. Reproduksi: Seleksi Turnamen, Crossover Dua Titik, dan Mutasi Adaptif
            while len(new_population) < self.params.population_size:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)

                # Two-Point Crossover
                if random.random() < self.params.crossover_rate and n_claims > 2:
                    pt1 = random.randint(0, n_claims - 2)
                    pt2 = random.randint(pt1 + 1, n_claims - 1)
                    child1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
                    child2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
                else:
                    child1 = p1[:]
                    child2 = p2[:]

                # Adaptive Mutation
                self._mutate(child1, n_genes)
                self._mutate(child2, n_genes)

                new_population.append(child1)
                if len(new_population) < self.params.population_size:
                    new_population.append(child2)

            population = new_population

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Konstruksi Solusi Akhir
        solution_map: Dict[str, AssignmentSlot] = {}
        for c_idx, gene in enumerate(best_individual):
            solution_map[self.claims[c_idx].claim_id] = self.possible_assignments[gene]

        return SolverStatistics(
            algorithm=f"Genetic Algorithm (Pop={self.params.population_size}, Gen={self.params.generations})",
            is_feasible=(best_violations == 0),
            runtime_ms=elapsed_ms,
            generations_completed=gen_completed,
            nodes_expanded=self.params.population_size * gen_completed,
            backtracks=0,
            hard_violations=best_violations,
            total_handling_cost_idr=best_cost,
            workload_std_dev=best_std_dev,
            fitness_score=best_fitness,
            solution=solution_map if best_violations == 0 else None,
            evolution_history=history,
        )

    def _generate_initial_population(self, n_claims: int, n_genes: int) -> List[List[int]]:
        """
        Pembangkitan populasi awal hibrida:
        - 30% seeded dengan pemilihan gen yang cocok secara kualifikasi peran.
        - 70% acak murni untuk menjamin diversitas genetik ruang pencarian.
        """
        pop: List[List[int]] = []
        seeded_count = max(1, int(self.params.population_size * 0.3))

        # 1. Individu Terarah (Heuristic Seeding)
        for _ in range(seeded_count):
            chromosome: List[int] = []
            for claim in self.claims:
                valid_genes = [
                    idx
                    for idx, slot in enumerate(self.possible_assignments)
                    if is_role_compatible(self.verifier_map[slot.verifier_id].role, claim.required_role)
                    and claim.amount_idr <= self.verifier_map[slot.verifier_id].max_claim_amount_idr
                    and claim.hospital_id not in self.verifier_map[slot.verifier_id].affiliated_hospitals
                    and not (slot.shift == ShiftSlot.MALAM and claim.sla_hours <= 24)
                ]
                if valid_genes:
                    chromosome.append(random.choice(valid_genes))
                else:
                    chromosome.append(random.randint(0, n_genes - 1))
            pop.append(chromosome)

        # 2. Individu Acak Murni (Diversity)
        while len(pop) < self.params.population_size:
            pop.append([random.randint(0, n_genes - 1) for _ in range(n_claims)])

        return pop

    def _tournament_select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        """Operator seleksi turnamen (Tournament Selection)."""
        candidates = random.sample(range(len(population)), self.params.tournament_size)
        best_candidate = max(candidates, key=lambda c: fitnesses[c])
        return population[best_candidate]

    def _mutate(self, chromosome: List[int], n_genes: int) -> None:
        """Operator mutasi seragam dengan probabilitas mutasi per gen."""
        for i in range(len(chromosome)):
            if random.random() < self.params.mutation_rate:
                chromosome[i] = random.randint(0, n_genes - 1)


# ===========================================================================
# 4. CLAIM ALLOCATION ENGINE (ENTERPRISE DOMAIN ORCHESTRATION)
# ===========================================================================

class ClaimAllocationEngine:
    """
    Mesin orkestrasi alokasi klaim terpadu.
    
    Memetakan batch berkas klaim dan tim verifikator menjadi masalah optimasi
    GA multi-objektif serta mengevaluasi kelayakan operasional.
    """

    def __init__(
        self,
        claims: Sequence[Claim],
        verifiers: Sequence[Verifier],
        shifts: Sequence[ShiftSlot] = (ShiftSlot.PAGI, ShiftSlot.SIANG, ShiftSlot.MALAM),
        params: Optional[GAParameters] = None,
    ) -> None:
        self.claims = list(claims)
        self.verifiers = list(verifiers)
        self.shifts = list(shifts)
        self.params = params or GAParameters()
        self.solver = GeneticAlgorithmSolver(
            self.claims, self.verifiers, self.shifts, self.params
        )

    def solve(self, method: str = "ga", timeout_seconds: float = 30.0) -> SolverStatistics:
        """Menjalankan proses optimasi alokasi klaim menggunakan Algoritma Genetika."""
        return self.solver.solve()


# ===========================================================================
# 5. SUITE ANALISIS SENSITIVITAS & BENCHMARK TOLOK UKUR
# ===========================================================================

def generate_benchmark_instance(
    scale_label: str,
    n_claims: int,
    n_verifiers: int,
    seed: int = 100,
) -> Tuple[List[Claim], List[Verifier]]:
    """Membangkitkan dataset klaim dan verifikator sintetis realistis untuk eksperimen."""
    rng = random.Random(seed)

    roles_pool = [
        VerifierRole.JUNIOR_ADJUSTER,
        VerifierRole.SENIOR_ADJUSTER,
        VerifierRole.MEDICAL_ADVISOR,
        VerifierRole.FRAUD_INVESTIGATOR,
    ]
    verifiers: List[Verifier] = []
    for i in range(n_verifiers):
        role = roles_pool[i % len(roles_pool)]
        if role == VerifierRole.JUNIOR_ADJUSTER:
            cap_wl = rng.randint(20, 26)
            cap_cl = rng.randint(12, 16)
            limit_val = 25_000_000.0
            rate = 60_000.0
        elif role == VerifierRole.SENIOR_ADJUSTER:
            cap_wl = rng.randint(24, 30)
            cap_cl = rng.randint(14, 18)
            limit_val = 75_000_000.0
            rate = 90_000.0
        elif role == VerifierRole.MEDICAL_ADVISOR:
            cap_wl = rng.randint(18, 24)
            cap_cl = rng.randint(8, 12)
            limit_val = 250_000_000.0
            rate = 180_000.0
        else:  # FRAUD_INVESTIGATOR
            cap_wl = rng.randint(16, 22)
            cap_cl = rng.randint(6, 10)
            limit_val = 500_000_000.0
            rate = 160_000.0

        v = Verifier(
            verifier_id=f"STF-{i+1:02d}",
            name=f"Officer {i+1}",
            role=role,
            max_daily_workload=cap_wl,
            max_daily_claims=cap_cl,
            max_claim_amount_idr=limit_val,
            affiliated_hospitals=frozenset([f"HOSP-CONFLICT-{i}"]) if i % 5 == 0 else frozenset(),
            hourly_rate_idr=rate,
        )
        verifiers.append(v)

    claims: List[Claim] = []
    for j in range(n_claims):
        claim_type_choice = rng.choices(
            ["OUTPATIENT", "INPATIENT", "SURGICAL", "FRAUD_SUSPECT"],
            weights=[0.50, 0.25, 0.15, 0.10],
            k=1,
        )[0]

        if claim_type_choice == "OUTPATIENT":
            req_role = VerifierRole.JUNIOR_ADJUSTER
            amount = rng.uniform(500_000, 15_000_000)
            comp = rng.randint(1, 2)
            sla = 24
            risk = rng.uniform(0.05, 0.25)
        elif claim_type_choice == "INPATIENT":
            req_role = VerifierRole.SENIOR_ADJUSTER
            amount = rng.uniform(15_000_000, 60_000_000)
            comp = rng.randint(2, 3)
            sla = 48
            risk = rng.uniform(0.20, 0.50)
        elif claim_type_choice == "SURGICAL":
            req_role = VerifierRole.MEDICAL_ADVISOR
            amount = rng.uniform(40_000_000, 180_000_000)
            comp = rng.randint(3, 5)
            sla = 72
            risk = rng.uniform(0.35, 0.70)
        else:  # FRAUD_SUSPECT
            req_role = VerifierRole.FRAUD_INVESTIGATOR
            amount = rng.uniform(20_000_000, 250_000_000)
            comp = rng.randint(3, 5)
            sla = 48
            risk = rng.uniform(0.70, 0.95)

        conflict = f"GRP-{j // 8}" if j % 6 == 0 else None

        c = Claim(
            claim_id=f"CLM-{scale_label}-{j+1:03d}",
            claim_type=claim_type_choice,
            amount_idr=amount,
            required_role=req_role,
            complexity=comp,
            sla_hours=sla,
            hospital_id=f"HOSP-{rng.randint(1, 5):02d}",
            conflict_group=conflict,
            risk_score=risk,
        )
        claims.append(c)

    return claims, verifiers


def run_sensitivity_analysis() -> List[Dict[str, Any]]:
    """
    Menjalankan pengujian analisis sensitivitas terhadap:
    1. Variasi ukuran masalah (skala kecil, sedang, besar).
    2. Variasi parameter Algoritma Genetika (ukuran populasi).
    3. Pengujian kasus ekstrem (over-constrained, bottleneck spesialis).
    """
    scenarios = [
        ("Skala Kecil (Small Scale)", 10, 4, GAParameters(population_size=40, generations=100, seed=101)),
        ("Skala Sedang (Medium Scale)", 40, 10, GAParameters(population_size=60, generations=150, seed=202)),
        ("Skala Besar (Large Scale)", 100, 24, GAParameters(population_size=100, generations=200, seed=303)),
        ("Kasus Ekstrem: Over-Constrained (Pigeonhole)", 25, 2, GAParameters(population_size=50, generations=100, seed=404)),
        ("Kasus Ekstrem: Specialist Bottleneck", 12, 3, GAParameters(population_size=50, generations=100, seed=505)),
    ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print("CLAIMWISE AI — GENETIC ALGORITHM (GA) SENSITIVITY & CONVERGENCE BENCHMARK")
    print("=" * 95)

    for name, n_c, n_v, params in scenarios:
        if "Over-Constrained" in name:
            # 25 berkas klaim, tetapi hanya 2 staf dengan kuota kecil (kapasitas total < volume)
            claims, _ = generate_benchmark_instance("EXT-OVER", n_c, 2, seed=params.seed or 1)
            verifiers = [
                Verifier("V1", "Officer 1", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=6, max_daily_claims=4),
                Verifier("V2", "Officer 2", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=6, max_daily_claims=4),
            ]
        elif "Specialist Bottleneck" in name:
            # 12 klaim bedah (wajib Medical Advisor), hanya ada 1 Medical Advisor dengan kapasitas pas
            verifiers = [
                Verifier("DOC-1", "Dr. Budi", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=30, max_daily_claims=12, max_claim_amount_idr=300_000_000),
                Verifier("ADJ-1", "Siti", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
                Verifier("ADJ-2", "Rudi", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
            ]
            claims = [
                Claim(f"CLM-SURG-{i+1:02d}", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=2, sla_hours=48)
                for i in range(n_c)
            ]
        else:
            claims, verifiers = generate_benchmark_instance(name[:3].strip(), n_c, n_v, seed=params.seed or 1)

        engine = ClaimAllocationEngine(claims, verifiers, params=params)
        stats = engine.solve()

        record = {
            "scenario": name,
            "claims_count": len(claims),
            "verifiers_count": len(verifiers),
            "pop_size": params.population_size,
            "generations": stats.generations_completed,
            "is_feasible": stats.is_feasible,
            "runtime_ms": round(stats.runtime_ms, 2),
            "hard_violations": stats.hard_violations,
            "handling_cost_idr": round(stats.total_handling_cost_idr, 2),
            "workload_std_dev": round(stats.workload_std_dev, 2),
            "best_fitness": round(stats.fitness_score, 4),
        }
        results.append(record)

        print(f"\n[Skenario: {name}]")
        print(f"  • Konfigurasi : {len(claims)} Berkas Klaim | {len(verifiers)} Staf | Populasi={params.population_size}")
        print(f"  • Hasil GA    : Feasible={stats.is_feasible} | Waktu={stats.runtime_ms:.2f} ms | Gen Selesai={stats.generations_completed}")
        print(f"  • Metrik Mutu : Pelanggaran={stats.hard_violations} | Biaya=Rp {stats.total_handling_cost_idr:,.0f} | &sigma; Beban={stats.workload_std_dev:.2f}")

    print("\n" + "=" * 95)
    print("ANALISIS SENSITIVITAS GA SELESAI DENGAN SUKSES.")
    print("=" * 95 + "\n")
    return results


# ===========================================================================
# 6. DEMONSTRASI & ANTARMUKA CLI
# ===========================================================================

def run_sample_demonstration() -> None:
    """Menjalankan demonstrasi alokasi klaim interaktif menggunakan Algoritma Genetika."""
    print("\n=== CLAIMWISE AI — DEMONSTRASI OPTIMASI ALGORITMA GENETIKA (MILESTONE 2) ===")

    # 1. Definisi Tim Verifikator Enterprise
    verifiers = [
        Verifier("ADJ-01", "Andi Pratama", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=20, max_daily_claims=10, max_claim_amount_idr=25_000_000.0, hourly_rate_idr=60_000.0),
        Verifier("ADJ-02", "Bona Siregar", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=25, max_daily_claims=12, max_claim_amount_idr=75_000_000.0, hourly_rate_idr=90_000.0),
        Verifier("DOC-01", "dr. Citra Sp.A", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=18, max_daily_claims=8, max_claim_amount_idr=200_000_000.0, hourly_rate_idr=180_000.0),
        Verifier("INV-01", "David Sianipar", VerifierRole.FRAUD_INVESTIGATOR, max_daily_workload=15, max_daily_claims=6, max_claim_amount_idr=500_000_000.0, hourly_rate_idr=160_000.0),
    ]

    # 2. Definisi Berkas Klaim Harian Masuk
    claims = [
        Claim("CLM-001", "OUTPATIENT", 2_500_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-01", risk_score=0.10),
        Claim("CLM-002", "INPATIENT", 35_000_000, VerifierRole.SENIOR_ADJUSTER, complexity=2, sla_hours=48, hospital_id="HOSP-02", risk_score=0.45),
        Claim("CLM-003", "SURGICAL", 120_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4, sla_hours=72, hospital_id="HOSP-01", risk_score=0.60),
        Claim("CLM-004", "FRAUD_SUSPECT", 65_000_000, VerifierRole.FRAUD_INVESTIGATOR, complexity=3, sla_hours=48, hospital_id="HOSP-03", risk_score=0.90),
        Claim("CLM-005", "OUTPATIENT", 4_200_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAMILY-99", risk_score=0.15),
        Claim("CLM-006", "OUTPATIENT", 3_800_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAMILY-99", risk_score=0.15),
    ]

    params = GAParameters(population_size=60, generations=120, seed=42)
    engine = ClaimAllocationEngine(claims, verifiers, params=params)
    stats = engine.solve()

    print(f"\n[Status Solusi GA: {'BERHASIL (FEASIBLE)' if stats.is_feasible else 'TIDAK LAYAK'}]")
    print(f"Durasi Komputasi     : {stats.runtime_ms:.2f} ms")
    print(f"Generasi Selesai     : {stats.generations_completed} generasi")
    print(f"Pelanggaran Regulasi : {stats.hard_violations} kasus")
    print(f"Total Biaya Staf     : Rp {stats.total_handling_cost_idr:,.2f}")
    print(f"Deviasi Beban Staf   : {stats.workload_std_dev:.2f} poin kompleksitas")
    print("-" * 75)
    print(f"{'KODE KLAIM':<12} | {'TIPE KLAIM':<14} | {'NOMINAL (IDR)':<15} | {'VERIFIKATOR':<12} | {'SHIFT'}")
    print("-" * 75)

    if stats.solution:
        for c in claims:
            slot = stats.solution[c.claim_id]
            print(f"{c.claim_id:<12} | {c.claim_type:<14} | Rp {c.amount_idr:>11,.0f} | {slot.verifier_id:<12} | {slot.shift.value}")
    print("-" * 75)


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        run_sensitivity_analysis()
    else:
        run_sample_demonstration()
