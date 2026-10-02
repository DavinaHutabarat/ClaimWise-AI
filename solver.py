"""
ClaimWise AI — Business Constraint Solver Module (CSP & GA Optimization)
=========================================================================
Mata Kuliah : 10S3001 - Kecerdasan Buatan (+P) / Artificial Intelligence
Institusi   : Institut Teknologi Del (FITE - Sarjana Sistem Informasi)
Tugas       : Tugas 2 (Milestone 2 - W04) — Milestone Proyek Terpadu (PjBL)

Deskripsi Modul:
----------------
Modul ini mengimplementasikan mesin penalaran dan pemecahan batasan keputusan
bisnis (*Constraint Satisfaction Problem* & *Genetic Algorithm Optimization*)
untuk sub-masalah operasional kritis pada ClaimWise AI:
"Optimasi Penjadwalan Shift & Alokasi Penugasan Berkas Verifikator Klaim Asuransi
Kesehatan Terikat Batasan Regulasi & Ketenagakerjaan."

Komponen Utama:
1. CSP Engine Formal:
   - Abstraksi Variabel, Domain, Batasan Unary, Binary, dan Global.
   - Algoritma Arc Consistency 3 (AC-3) untuk propagasi dan pemangkasan domain.
   - Backtracking Search dengan heuristik MRV (Minimum Remaining Values),
     Degree Heuristic (tie-breaker), dan LCV (Least Constraining Value).
   - Inferensi MAC (Maintaining Arc Consistency) & Forward Checking.
2. Genetic Algorithm (GA) Optimization Engine:
   - Representasi kromosom diskret.
   - Evaluasi kebugaran dengan fungsi penalti batasan mutlak & optimasi biaya/beban.
   - Seleksi turnamen (Tournament Selection), Two-point crossover, dan Elitisme.
3. Domain Model Enterprise (ClaimWise AI):
   - Pemodelan wewenang finansial & kompetensi klinis/fraud staf.
   - Batasan batas beban kerja harian regulasi ketenagakerjaan (UU No. 13/2003).
   - Pemisahan tugas (*Separation of Duties*) & pencegahan konflik kepentingan RS.
   - Kepatuhan SLA OJK (POJK No. 69/POJK.05/2016).
4. Mesin Analisis Sensitivitas & Tolok Ukur (Benchmark):
   - Pengujian terhadap variasi skala masalah (kecil, sedang, besar).
   - Pengujian kasus ekstrem (over-constrained, bottleneck, conflict clique).
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)


# ===========================================================================
# 1. DOMAIN ENUMS & DATA MODELS
# ===========================================================================

class VerifierRole(str, Enum):
    """Tingkat kompetensi dan wewenang verifikator klaim asuransi."""
    JUNIOR_ADJUSTER = "JUNIOR_ADJUSTER"      # Verifikator dasar (klaim sederhana < Rp 25 Juta)
    SENIOR_ADJUSTER = "SENIOR_ADJUSTER"      # Verifikator senior (klaim moderat < Rp 75 Juta)
    MEDICAL_ADVISOR = "MEDICAL_ADVISOR"      # Dokter penasihat (tindakan bedah, rawat inap kompleks)
    FRAUD_INVESTIGATOR = "FRAUD_INVESTIGATOR"# Investigator forensik (indikasi anomali/fraud)


ROLE_HIERARCHY: Dict[VerifierRole, int] = {
    VerifierRole.JUNIOR_ADJUSTER: 1,
    VerifierRole.SENIOR_ADJUSTER: 2,
    VerifierRole.MEDICAL_ADVISOR: 3,
    VerifierRole.FRAUD_INVESTIGATOR: 4,
}


def is_role_compatible(verifier_role: VerifierRole, required_role: VerifierRole) -> bool:
    """
    Memvalidasi kesesuaian kompetensi dan lisensi verifikator terhadap jenis klaim:
    - MEDICAL_ADVISOR wajib dokter (hanya MEDICAL_ADVISOR yang sah).
    - FRAUD_INVESTIGATOR wajib auditor forensik (hanya FRAUD_INVESTIGATOR yang sah).
    - SENIOR_ADJUSTER dapat ditangani oleh Senior Adjuster, Medical Advisor, atau Fraud Investigator.
    - JUNIOR_ADJUSTER dapat ditangani oleh seluruh staf (jika batasan nominal dipenuhi).
    """
    if required_role == VerifierRole.MEDICAL_ADVISOR:
        return verifier_role == VerifierRole.MEDICAL_ADVISOR
    if required_role == VerifierRole.FRAUD_INVESTIGATOR:
        return verifier_role == VerifierRole.FRAUD_INVESTIGATOR
    if required_role == VerifierRole.SENIOR_ADJUSTER:
        return verifier_role in (VerifierRole.SENIOR_ADJUSTER, VerifierRole.MEDICAL_ADVISOR, VerifierRole.FRAUD_INVESTIGATOR)
    return True


class ShiftSlot(str, Enum):
    """Jendela shift kerja operasional harian."""
    PAGI = "PAGI"      # 08:00 - 16:00 (Kapasitas penuh, verifikasi & otorisasi supervisi)
    SIANG = "SIANG"    # 14:00 - 22:00 (Triase intensif & verifikasi standar)
    MALAM = "MALAM"    # 22:00 - 06:00 (Intake darurat & monitoring antrean otomatis)


@dataclass(frozen=True)
class Claim:
    """Representasi berkas klaim asuransi yang membutuhkan verifikasi."""
    claim_id: str
    claim_type: str                  # "OUTPATIENT", "INPATIENT", "SURGICAL", "FRAUD_SUSPECT"
    amount_idr: float                # Nilai nominal klaim dalam Rupiah
    required_role: VerifierRole      # Kompetensi minimum yang dipersyaratkan
    complexity: int = 1              # Bobot kompleksitas kognitif (1 - 5 poin)
    sla_hours: int = 24              # Batas waktu maksimal penyelesaian SLA (Jam)
    hospital_id: str = "HOSP-01"     # Rumah sakit penyedia pelayanan medis
    conflict_group: Optional[str] = None  # ID kelompok jika ada relasi/sengketa terafiliasi


@dataclass(frozen=True)
class Verifier:
    """Representasi staf verifikator asuransi dengan kapasitas terikat regulasi."""
    verifier_id: str
    name: str
    role: VerifierRole
    max_daily_workload: int = 20     # Maksimal poin kompleksitas per hari (UU Ketenagakerjaan)
    max_daily_claims: int = 12       # Maksimal volume berkas klaim per hari
    max_claim_amount_idr: float = 25_000_000.0  # Limit otorisasi wewenang finansial
    affiliated_hospitals: FrozenSet[str] = field(default_factory=frozenset)  # Konflik kepentingan
    shift_preference: Optional[ShiftSlot] = None
    hourly_rate_idr: float = 75_000.0


@dataclass(frozen=True)
class AssignmentSlot:
    """Nilai domain: Pasangan verifikator yang ditugaskan dan slot shift."""
    verifier_id: str
    shift: ShiftSlot

    def __repr__(self) -> str:
        return f"({self.verifier_id}@{self.shift.value})"


@dataclass
class SolverStatistics:
    """Statistik komputasi dan metrik inferensi penelusuran batasan."""
    algorithm: str
    is_feasible: bool = False
    runtime_ms: float = 0.0
    nodes_expanded: int = 0
    backtracks: int = 0
    ac3_prunes: int = 0
    hard_violations: int = 0
    total_handling_cost_idr: float = 0.0
    workload_std_dev: float = 0.0
    solution: Optional[Dict[str, AssignmentSlot]] = None


# ===========================================================================
# 2. GENERIC CONSTRAINT SATISFACTION PROBLEM (CSP) FRAMEWORK
# ===========================================================================

V = TypeVar("V")  # Variable type
D = TypeVar("D")  # Domain value type


class Constraint(Generic[V, D]):
    """Kelas abstrak dasar untuk batasan pada CSP."""

    def __init__(self, variables: Sequence[V]) -> None:
        self.variables = list(variables)

    def is_satisfied(self, assignment: Dict[V, D]) -> bool:
        """Memeriksa apakah penugasan parsial/lengkap memenuhi batasan ini."""
        raise NotImplementedError


class BinaryConstraint(Constraint[V, D]):
    """Batasan biner yang mengikat dua variabel (X, Y)."""

    def __init__(
        self,
        var1: V,
        var2: V,
        predicate: Callable[[D, D], bool],
        name: str = "BinaryConstraint",
    ) -> None:
        super().__init__([var1, var2])
        self.var1 = var1
        self.var2 = var2
        self.predicate = predicate
        self.name = name

    def is_satisfied(self, assignment: Dict[V, D]) -> bool:
        if self.var1 not in assignment or self.var2 not in assignment:
            return True
        return self.predicate(assignment[self.var1], assignment[self.var2])

    def test(self, val1: D, val2: D) -> bool:
        return self.predicate(val1, val2)


class CSP(Generic[V, D]):
    """
    Representasi formal model CSP: P = <X, D, C>.
    
    Menyimpan himpunan variabel X, pemetaan domain D, dan batasan C.
    Mendukung pembangunan graf tetangga (neighbor graph) untuk AC-3 dan MAC.
    """

    def __init__(
        self,
        variables: Sequence[V],
        domains: Dict[V, List[D]],
    ) -> None:
        self.variables: List[V] = list(variables)
        self.domains: Dict[V, List[D]] = {v: list(domains[v]) for v in variables}
        self.constraints: Dict[V, List[Constraint[V, D]]] = {v: [] for v in variables}
        self.binary_constraints: Dict[Tuple[V, V], List[BinaryConstraint[V, D]]] = defaultdict(list)
        self.neighbors: Dict[V, Set[V]] = {v: set() for v in variables}
        self.global_constraints: List[Constraint[V, D]] = []

    def add_constraint(self, constraint: Constraint[V, D]) -> None:
        """Menambahkan batasan ke CSP dan memperbarui relasi tetangga."""
        for var in constraint.variables:
            if var not in self.variables:
                raise ValueError(f"Variabel {var} tidak terdaftar dalam CSP.")
            self.constraints[var].append(constraint)

        if isinstance(constraint, BinaryConstraint):
            v1, v2 = constraint.var1, constraint.var2
            self.binary_constraints[(v1, v2)].append(constraint)
            # Simetris untuk penelusuran dua arah
            reverse_c = BinaryConstraint(v2, v1, lambda y, x: constraint.predicate(x, y), name=constraint.name)
            self.binary_constraints[(v2, v1)].append(reverse_c)
            self.neighbors[v1].add(v2)
            self.neighbors[v2].add(v1)
        elif len(constraint.variables) > 2:
            self.global_constraints.append(constraint)

    def is_consistent(self, var: V, val: D, assignment: Dict[V, D]) -> bool:
        """
        Memeriksa apakah penugasan var=val konsisten dengan seluruh batasan
        terhadap variabel yang telah memiliki nilai penugasan.
        """
        temp_assignment = assignment.copy()
        temp_assignment[var] = val

        for constraint in self.constraints[var]:
            if not constraint.is_satisfied(temp_assignment):
                return False

        for g_constraint in self.global_constraints:
            if var in g_constraint.variables:
                if not g_constraint.is_satisfied(temp_assignment):
                    return False

        return True


# ===========================================================================
# 3. ALGORITMA AC-3 (ARC CONSISTENCY 3)
# ===========================================================================

def ac3(
    csp: CSP[V, D],
    queue: Optional[Sequence[Tuple[V, V]]] = None,
    stats: Optional[SolverStatistics] = None,
) -> bool:
    """
    Algoritma AC-3 (Arc Consistency 3) karya Mackworth (1977).
    """
    if queue is None:
        arc_queue: deque[Tuple[V, V]] = deque()
        for v1 in csp.variables:
            for v2 in csp.neighbors[v1]:
                arc_queue.append((v1, v2))
    else:
        arc_queue = deque(queue)

    while arc_queue:
        xi, xj = arc_queue.popleft()
        revised, pruned_count = _revise(csp, xi, xj)
        if revised:
            if stats:
                stats.ac3_prunes += pruned_count
            if len(csp.domains[xi]) == 0:
                return False
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    arc_queue.append((xk, xi))
    return True


def _revise(csp: CSP[V, D], xi: V, xj: V) -> Tuple[bool, int]:
    """
    Fungsi Revise: Menghapus nilai x dari D(Xi) jika tidak ada y di D(Xj) yang konsisten.
    """
    revised = False
    pruned_count = 0
    constraints = csp.binary_constraints.get((xi, xj), [])
    if not constraints:
        return False, 0

    new_domain: List[D] = []
    for x in csp.domains[xi]:
        has_support = False
        for y in csp.domains[xj]:
            if all(c.test(x, y) for c in constraints):
                has_support = True
                break
        if has_support:
            new_domain.append(x)
        else:
            revised = True
            pruned_count += 1

    if revised:
        csp.domains[xi] = new_domain

    return revised, pruned_count


# ===========================================================================
# 4. BACKTRACKING SEARCH DENGAN MRV, DEGREE & LCV HEURISTICS
# ===========================================================================

def select_unassigned_variable_mrv(
    assignment: Dict[V, D],
    csp: CSP[V, D],
) -> V:
    """
    Heuristik Minimum Remaining Values (MRV / Most Constrained Variable).
    Tie breaker dengan Degree Heuristic.
    """
    unassigned = [v for v in csp.variables if v not in assignment]

    def key_func(v: V) -> Tuple[int, int]:
        domain_size = len(csp.domains[v])
        unassigned_neighbors = sum(1 for neighbor in csp.neighbors[v] if neighbor not in assignment)
        return (domain_size, -unassigned_neighbors)

    return min(unassigned, key=key_func)


def order_domain_values_lcv(
    var: V,
    assignment: Dict[V, D],
    csp: CSP[V, D],
) -> List[D]:
    """
    Heuristik Least Constraining Value (LCV).
    """
    if len(csp.domains[var]) <= 1:
        return list(csp.domains[var])

    unassigned_neighbors = [n for n in csp.neighbors[var] if n not in assignment]
    if not unassigned_neighbors:
        return list(csp.domains[var])

    def count_conflicts(val: D) -> int:
        conflicts = 0
        for neighbor in unassigned_neighbors:
            constraints = csp.binary_constraints.get((var, neighbor), [])
            for neighbor_val in csp.domains[neighbor]:
                if any(not c.test(val, neighbor_val) for c in constraints):
                    conflicts += 1
        return conflicts

    return sorted(csp.domains[var], key=count_conflicts)


def forward_check(
    var: V,
    val: D,
    assignment: Dict[V, D],
    csp: CSP[V, D],
) -> Tuple[bool, Dict[V, List[D]]]:
    """Inferensi Forward Checking."""
    pruned: Dict[V, List[D]] = defaultdict(list)
    unassigned_neighbors = [n for n in csp.neighbors[var] if n not in assignment]

    for neighbor in unassigned_neighbors:
        constraints = csp.binary_constraints.get((var, neighbor), [])
        valid_neighbor_vals: List[D] = []
        for n_val in csp.domains[neighbor]:
            if all(c.test(val, n_val) for c in constraints):
                valid_neighbor_vals.append(n_val)
            else:
                pruned[neighbor].append(n_val)

        if not valid_neighbor_vals:
            return False, pruned

        csp.domains[neighbor] = valid_neighbor_vals

    return True, pruned


def mac_inference(
    var: V,
    val: D,
    assignment: Dict[V, D],
    csp: CSP[V, D],
    stats: Optional[SolverStatistics] = None,
) -> Tuple[bool, Dict[V, List[D]]]:
    """Inferensi Maintaining Arc Consistency (MAC)."""
    saved_domains = {v: list(csp.domains[v]) for v in csp.variables if v not in assignment and v != var}
    csp.domains[var] = [val]

    queue = [(neighbor, var) for neighbor in csp.neighbors[var] if neighbor not in assignment]
    consistent = ac3(csp, queue=queue, stats=stats)

    if not consistent:
        for v, dom in saved_domains.items():
            csp.domains[v] = dom
        return False, {}

    pruned_dict: Dict[V, List[D]] = {}
    for v, old_dom in saved_domains.items():
        pruned_vals = [x for x in old_dom if x not in csp.domains[v]]
        if pruned_vals:
            pruned_dict[v] = pruned_vals

    return True, pruned_dict


def backtracking_search(
    csp: CSP[V, D],
    use_ac3_preprocess: bool = True,
    use_mrv: bool = True,
    use_lcv: bool = True,
    inference: str = "mac",
    timeout_seconds: float = 30.0,
) -> SolverStatistics:
    """Eksekusi penelusuran Backtracking lengkap."""
    stats = SolverStatistics(algorithm=f"Backtracking (MRV={use_mrv}, LCV={use_lcv}, Infer={inference})")
    start_time = time.perf_counter()

    csp_copy = copy.deepcopy(csp)

    if use_ac3_preprocess:
        pre_consistent = ac3(csp_copy, stats=stats)
        if not pre_consistent:
            stats.runtime_ms = (time.perf_counter() - start_time) * 1000
            stats.is_feasible = False
            return stats

    assignment: Dict[V, D] = {}

    def backtrack() -> Optional[Dict[V, D]]:
        if (time.perf_counter() - start_time) > timeout_seconds:
            return None

        if len(assignment) == len(csp_copy.variables):
            return assignment

        stats.nodes_expanded += 1

        if use_mrv:
            var = select_unassigned_variable_mrv(assignment, csp_copy)
        else:
            unassigned = [v for v in csp_copy.variables if v not in assignment]
            var = unassigned[0]

        if use_lcv:
            values = order_domain_values_lcv(var, assignment, csp_copy)
        else:
            values = list(csp_copy.domains[var])

        for val in values:
            if csp_copy.is_consistent(var, val, assignment):
                assignment[var] = val
                saved_domains = {v: list(csp_copy.domains[v]) for v in csp_copy.variables}

                infe_ok = True
                if inference == "mac":
                    infe_ok, _ = mac_inference(var, val, assignment, csp_copy, stats=stats)
                elif inference == "fc":
                    infe_ok, _ = forward_check(var, val, assignment, csp_copy)

                if infe_ok:
                    result = backtrack()
                    if result is not None:
                        return result

                del assignment[var]
                for v, dom in saved_domains.items():
                    csp_copy.domains[v] = dom
                stats.backtracks += 1

        return None

    solution = backtrack()
    stats.runtime_ms = (time.perf_counter() - start_time) * 1000

    if solution is not None:
        stats.is_feasible = True
        stats.solution = solution
    else:
        stats.is_feasible = False

    return stats


# ===========================================================================
# 5. GENETIC ALGORITHM (GA) OPTIMIZATION ENGINE
# ===========================================================================

@dataclass
class GAParameters:
    """Konfigurasi parameter evolusi Algoritma Genetika."""
    population_size: int = 60
    generations: int = 150
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.08
    elite_count: int = 2
    seed: Optional[int] = 42


class GeneticAlgorithmSolver:
    """Solver GA dengan operator turnamen dan elitisme."""

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

        self.possible_assignments: List[AssignmentSlot] = [
            AssignmentSlot(v.verifier_id, s)
            for v in self.verifiers
            for s in self.shifts
        ]
        self.verifier_map = {v.verifier_id: v for v in self.verifiers}

    def _evaluate_individual(self, chromosome: List[int]) -> Tuple[float, int, float, float]:
        """Menghitung fitness individu kromosom."""
        hard_violations = 0
        total_cost = 0.0
        verifier_workload: Dict[str, int] = defaultdict(int)
        verifier_claims: Dict[str, int] = defaultdict(int)
        conflict_tracker: Dict[Tuple[str, str], str] = {}

        for claim_idx, gene in enumerate(chromosome):
            claim = self.claims[claim_idx]
            slot = self.possible_assignments[gene]
            verifier = self.verifier_map[slot.verifier_id]

            # 1. Batasan Kualifikasi Kompetensi / Role Compatibility
            if not is_role_compatible(verifier.role, claim.required_role):
                hard_violations += 1

            # 2. Batasan Wewenang Finansial Nominal
            if claim.amount_idr > verifier.max_claim_amount_idr:
                hard_violations += 1

            # 3. Batasan Konflik Kepentingan RS
            if claim.hospital_id in verifier.affiliated_hospitals:
                hard_violations += 1

            # 4. Batasan Shift Malam & SLA Cepat
            if slot.shift == ShiftSlot.MALAM and claim.sla_hours <= 24:
                hard_violations += 1

            # 5. Batasan Pemisahan Tugas (Conflict Group)
            if claim.conflict_group:
                key = (claim.conflict_group, slot.verifier_id)
                if key in conflict_tracker:
                    hard_violations += 1
                conflict_tracker[key] = claim.claim_id

            verifier_workload[verifier.verifier_id] += claim.complexity
            verifier_claims[verifier.verifier_id] += 1
            total_cost += verifier.hourly_rate_idr * (claim.complexity * 0.25)

        # 6. Batasan Kapasitas Beban Kerja Harian (UU Ketenagakerjaan)
        for v_id, workload in verifier_workload.items():
            limit = self.verifier_map[v_id].max_daily_workload
            if workload > limit:
                hard_violations += (workload - limit)

        for v_id, c_count in verifier_claims.items():
            limit = self.verifier_map[v_id].max_daily_claims
            if c_count > limit:
                hard_violations += (c_count - limit)

        loads = [verifier_workload[v.verifier_id] for v in self.verifiers]
        mean_load = sum(loads) / max(1, len(loads))
        variance = sum((x - mean_load) ** 2 for x in loads) / max(1, len(loads))
        std_dev = math.sqrt(variance)

        penalty = (hard_violations * 20_000.0) + (std_dev * 15.0) + (total_cost / 1_000.0)
        fitness = 1_000_000.0 / (1.0 + penalty)

        return fitness, hard_violations, total_cost, std_dev

    def solve(self) -> SolverStatistics:
        start_time = time.perf_counter()
        n_claims = len(self.claims)
        n_genes = len(self.possible_assignments)

        if n_claims == 0:
            return SolverStatistics(
                algorithm="Genetic Algorithm",
                is_feasible=True,
                runtime_ms=0.0,
                solution={},
            )

        population: List[List[int]] = [
            [random.randint(0, n_genes - 1) for _ in range(n_claims)]
            for _ in range(self.params.population_size)
        ]

        best_individual: List[int] = population[0]
        best_fitness = -1.0
        best_violations = 999999
        best_cost = 0.0
        best_std_dev = 0.0
        generation = 0

        for generation in range(self.params.generations):
            evaluations = [self._evaluate_individual(ind) for ind in population]
            fitness_scores = [ev[0] for ev in evaluations]

            for idx, (fit, viol, cost, std_dev) in enumerate(evaluations):
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = population[idx][:]
                    best_violations = viol
                    best_cost = cost
                    best_std_dev = std_dev

            if best_violations == 0 and generation >= 40:
                break

            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
            new_population: List[List[int]] = [
                population[sorted_indices[e]][:] for e in range(self.params.elite_count)
            ]

            while len(new_population) < self.params.population_size:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)

                if random.random() < self.params.crossover_rate and n_claims > 2:
                    pt1 = random.randint(0, n_claims - 2)
                    pt2 = random.randint(pt1 + 1, n_claims - 1)
                    child1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
                    child2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
                else:
                    child1 = p1[:]
                    child2 = p2[:]

                self._mutate(child1, n_genes)
                self._mutate(child2, n_genes)

                new_population.append(child1)
                if len(new_population) < self.params.population_size:
                    new_population.append(child2)

            population = new_population

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        sol_map: Dict[str, AssignmentSlot] = {}
        for c_idx, gene in enumerate(best_individual):
            sol_map[self.claims[c_idx].claim_id] = self.possible_assignments[gene]

        return SolverStatistics(
            algorithm=f"Genetic Algorithm (Pop={self.params.population_size}, Gen={self.params.generations})",
            is_feasible=(best_violations == 0),
            runtime_ms=elapsed_ms,
            nodes_expanded=self.params.population_size * (generation + 1),
            backtracks=0,
            ac3_prunes=0,
            hard_violations=best_violations,
            total_handling_cost_idr=best_cost,
            workload_std_dev=best_std_dev,
            solution=sol_map if best_violations == 0 else None,
        )

    def _tournament_select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        candidates = random.sample(range(len(population)), self.params.tournament_size)
        best_candidate = max(candidates, key=lambda c: fitnesses[c])
        return population[best_candidate]

    def _mutate(self, chromosome: List[int], n_genes: int) -> None:
        for i in range(len(chromosome)):
            if random.random() < self.params.mutation_rate:
                chromosome[i] = random.randint(0, n_genes - 1)


# ===========================================================================
# 6. CLAIM ALLOCATION ENGINE (ENTERPRISE DOMAIN MAPPING)
# ===========================================================================

class ClaimAllocationEngine:
    """Mesin orkestrasi alokasi klaim terpadu."""

    def __init__(
        self,
        claims: Sequence[Claim],
        verifiers: Sequence[Verifier],
        shifts: Sequence[ShiftSlot] = (ShiftSlot.PAGI, ShiftSlot.SIANG, ShiftSlot.MALAM),
    ) -> None:
        self.claims = list(claims)
        self.verifiers = list(verifiers)
        self.shifts = list(shifts)
        self.verifier_map = {v.verifier_id: v for v in self.verifiers}
        self.claim_map = {c.claim_id: c for c in self.claims}

    def build_csp(self) -> CSP[str, AssignmentSlot]:
        """Membangun formulasi CSP formal."""
        variables = [c.claim_id for c in self.claims]
        domains: Dict[str, List[AssignmentSlot]] = {}

        for claim in self.claims:
            legal_slots: List[AssignmentSlot] = []
            for verifier in self.verifiers:
                if not is_role_compatible(verifier.role, claim.required_role):
                    continue
                if claim.amount_idr > verifier.max_claim_amount_idr:
                    continue
                if claim.hospital_id in verifier.affiliated_hospitals:
                    continue

                for shift in self.shifts:
                    if shift == ShiftSlot.MALAM and claim.sla_hours <= 24:
                        continue
                    legal_slots.append(AssignmentSlot(verifier.verifier_id, shift))

            domains[claim.claim_id] = legal_slots

        csp = CSP[str, AssignmentSlot](variables, domains)

        conflict_groups: Dict[str, List[str]] = defaultdict(list)
        for claim in self.claims:
            if claim.conflict_group:
                conflict_groups[claim.conflict_group].append(claim.claim_id)

        for group_id, member_claims in conflict_groups.items():
            for i in range(len(member_claims)):
                for j in range(i + 1, len(member_claims)):
                    c1_id, c2_id = member_claims[i], member_claims[j]
                    c_bin = BinaryConstraint[str, AssignmentSlot](
                        c1_id,
                        c2_id,
                        lambda slot1, slot2: slot1.verifier_id != slot2.verifier_id,
                        name=f"ConflictGroup_{group_id}",
                    )
                    csp.add_constraint(c_bin)

        class WorkloadCapacityConstraint(Constraint[str, AssignmentSlot]):
            def __init__(
                c_self,
                variables: Sequence[str],
                engine: ClaimAllocationEngine,
            ) -> None:
                super().__init__(variables)
                c_self.engine = engine

            def is_satisfied(c_self, assignment: Dict[str, AssignmentSlot]) -> bool:
                workload_accum: Dict[str, int] = defaultdict(int)
                claim_count_accum: Dict[str, int] = defaultdict(int)

                for c_id, slot in assignment.items():
                    claim_obj = c_self.engine.claim_map[c_id]
                    workload_accum[slot.verifier_id] += claim_obj.complexity
                    claim_count_accum[slot.verifier_id] += 1

                    v_obj = c_self.engine.verifier_map[slot.verifier_id]
                    if workload_accum[slot.verifier_id] > v_obj.max_daily_workload:
                        return False
                    if claim_count_accum[slot.verifier_id] > v_obj.max_daily_claims:
                        return False

                return True

        if len(variables) > 0:
            csp.add_constraint(WorkloadCapacityConstraint(variables, self))

        return csp

    def solve(
        self,
        method: str = "csp_ac3_mrv",
        timeout_seconds: float = 30.0,
    ) -> SolverStatistics:
        if method.startswith("csp"):
            csp = self.build_csp()
            inference_type = "mac" if "mac" in method or "mrv" in method else "fc"
            stats = backtracking_search(
                csp,
                use_ac3_preprocess=True,
                use_mrv=True,
                use_lcv=True,
                inference=inference_type,
                timeout_seconds=timeout_seconds,
            )
            if stats.is_feasible and stats.solution:
                self._calculate_solution_metrics(stats)
            return stats
        elif method == "ga":
            ga_solver = GeneticAlgorithmSolver(self.claims, self.verifiers, self.shifts)
            stats = ga_solver.solve()
            return stats
        else:
            raise ValueError(f"Metode solver '{method}' tidak dikenali.")

    def _calculate_solution_metrics(self, stats: SolverStatistics) -> None:
        assert stats.solution is not None
        total_cost = 0.0
        workloads: Dict[str, int] = defaultdict(int)

        for c_id, slot in stats.solution.items():
            claim = self.claim_map[c_id]
            verifier = self.verifier_map[slot.verifier_id]
            workloads[slot.verifier_id] += claim.complexity
            total_cost += verifier.hourly_rate_idr * (claim.complexity * 0.25)

        loads = [workloads[v.verifier_id] for v in self.verifiers]
        mean_load = sum(loads) / max(1, len(loads))
        variance = sum((x - mean_load) ** 2 for x in loads) / max(1, len(loads))

        stats.total_handling_cost_idr = total_cost
        stats.workload_std_dev = math.sqrt(variance)


# ===========================================================================
# 7. SUITE ANALISIS SENSITIVITAS & PENGUJIAN TOLOK UKUR (BENCHMARK)
# ===========================================================================

def generate_benchmark_instance(
    scale_label: str,
    n_claims: int,
    n_verifiers: int,
    seed: int = 100,
) -> Tuple[List[Claim], List[Verifier]]:
    """Membuat instance dataset sintetis realistis."""
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
            cap_wl = rng.randint(18, 24)
            cap_cl = rng.randint(10, 15)
            limit_val = 25_000_000.0
            rate = 60_000.0
        elif role == VerifierRole.SENIOR_ADJUSTER:
            cap_wl = rng.randint(22, 28)
            cap_cl = rng.randint(12, 18)
            limit_val = 75_000_000.0
            rate = 90_000.0
        elif role == VerifierRole.MEDICAL_ADVISOR:
            cap_wl = rng.randint(14, 20)
            cap_cl = rng.randint(6, 10)
            limit_val = 250_000_000.0
            rate = 175_000.0
        else:
            cap_wl = rng.randint(12, 18)
            cap_cl = rng.randint(5, 8)
            limit_val = 500_000_000.0
            rate = 150_000.0

        v = Verifier(
            verifier_id=f"STF-{i+1:02d}",
            name=f"Officer {i+1}",
            role=role,
            max_daily_workload=cap_wl,
            max_daily_claims=cap_cl,
            max_claim_amount_idr=limit_val,
            affiliated_hospitals=frozenset([f"HOSP-CONFLICT-{i}"]) if i % 4 == 0 else frozenset(),
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
        elif claim_type_choice == "INPATIENT":
            req_role = VerifierRole.SENIOR_ADJUSTER
            amount = rng.uniform(15_000_000, 60_000_000)
            comp = rng.randint(2, 3)
            sla = 48
        elif claim_type_choice == "SURGICAL":
            req_role = VerifierRole.MEDICAL_ADVISOR
            amount = rng.uniform(40_000_000, 180_000_000)
            comp = rng.randint(3, 5)
            sla = 72
        else:
            req_role = VerifierRole.FRAUD_INVESTIGATOR
            amount = rng.uniform(20_000_000, 250_000_000)
            comp = rng.randint(3, 5)
            sla = 48

        conflict = f"GRP-{j // 10}" if j % 8 == 0 else None

        c = Claim(
            claim_id=f"CLM-{scale_label}-{j+1:03d}",
            claim_type=claim_type_choice,
            amount_idr=amount,
            required_role=req_role,
            complexity=comp,
            sla_hours=sla,
            hospital_id=f"HOSP-{rng.randint(1, 5):02d}",
            conflict_group=conflict,
        )
        claims.append(c)

    return claims, verifiers


def run_sensitivity_analysis() -> List[Dict[str, Any]]:
    """Menjalankan pengujian analisis sensitivitas."""
    scenarios = [
        ("Skala Kecil (Small Scale)", 10, 4, 101),
        ("Skala Sedang (Medium Scale)", 40, 10, 202),
        ("Skala Besar (Large Scale)", 100, 24, 303),
        ("Kasus Ekstrem: Over-Constrained (Pigeonhole)", 30, 2, 404),
        ("Kasus Ekstrem: Single Specialist Bottleneck", 15, 4, 505),
    ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("CLAIMWISE AI — SENSITIVITY & CONVERGENCE ANALYSIS BENCHMARK")
    print("=" * 90)

    for name, n_c, n_v, s_seed in scenarios:
        if "Over-Constrained" in name:
            claims, _ = generate_benchmark_instance("EXT-OVER", n_c, 2, seed=s_seed)
            verifiers = [
                Verifier("V1", "Officer 1", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=5, max_daily_claims=3),
                Verifier("V2", "Officer 2", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=5, max_daily_claims=3),
            ]
        elif "Specialist Bottleneck" in name:
            verifiers = [
                Verifier("DOC-1", "Dr. Budi", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=35, max_daily_claims=15, max_claim_amount_idr=300_000_000),
                Verifier("ADJ-1", "Siti", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
                Verifier("ADJ-2", "Rudi", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=30, max_daily_claims=15),
            ]
            claims = [
                Claim(f"CLM-SURG-{i+1:02d}", "SURGICAL", 50_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=2, sla_hours=48)
                for i in range(n_c)
            ]
        else:
            claims, verifiers = generate_benchmark_instance(name[:3].strip(), n_c, n_v, seed=s_seed)

        engine = ClaimAllocationEngine(claims, verifiers)

        csp_stats = engine.solve(method="csp_ac3_mrv", timeout_seconds=15.0)
        ga_stats = engine.solve(method="ga", timeout_seconds=15.0)

        record = {
            "scenario": name,
            "claims_count": len(claims),
            "verifiers_count": len(verifiers),
            "csp_feasible": csp_stats.is_feasible,
            "csp_runtime_ms": round(csp_stats.runtime_ms, 2),
            "csp_nodes": csp_stats.nodes_expanded,
            "csp_backtracks": csp_stats.backtracks,
            "ga_feasible": ga_stats.is_feasible,
            "ga_runtime_ms": round(ga_stats.runtime_ms, 2),
            "ga_violations": ga_stats.hard_violations,
        }
        results.append(record)

        print(f"\n[Skenario: {name}]")
        print(f"  • Konfigurasi: {len(claims)} Berkas Klaim | {len(verifiers)} Verifikator Staf")
        print(f"  • CSP Engine : Feasible={csp_stats.is_feasible} | Waktu={csp_stats.runtime_ms:.2f} ms | Nodes={csp_stats.nodes_expanded} | Backtracks={csp_stats.backtracks}")
        print(f"  • GA Engine  : Feasible={ga_stats.is_feasible} | Waktu={ga_stats.runtime_ms:.2f} ms | Violations={ga_stats.hard_violations}")

    print("\n" + "=" * 90)
    print("ANALISIS SELESAI DENGAN SUKSES.")
    print("=" * 90 + "\n")
    return results


# ===========================================================================
# 8. DEMONSTRASI & ANTARMUKA CLI
# ===========================================================================

def run_sample_demonstration() -> None:
    """Menjalankan demonstrasi alokasi klaim."""
    print("\n=== CLAIMWISE AI — DEMONSTRASI PEMECAHAN BATASAN BISNIS (MILESTONE 2) ===")

    verifiers = [
        Verifier("ADJ-01", "Andi Pratama", VerifierRole.JUNIOR_ADJUSTER, max_daily_workload=20, max_daily_claims=10, max_claim_amount_idr=25_000_000.0, hourly_rate_idr=60_000.0),
        Verifier("ADJ-02", "Bona Siregar", VerifierRole.SENIOR_ADJUSTER, max_daily_workload=25, max_daily_claims=12, max_claim_amount_idr=75_000_000.0, hourly_rate_idr=90_000.0),
        Verifier("DOC-01", "dr. Citra Sp.A", VerifierRole.MEDICAL_ADVISOR, max_daily_workload=18, max_daily_claims=8, max_claim_amount_idr=200_000_000.0, hourly_rate_idr=180_000.0),
        Verifier("INV-01", "David Sianipar", VerifierRole.FRAUD_INVESTIGATOR, max_daily_workload=15, max_daily_claims=6, max_claim_amount_idr=500_000_000.0, hourly_rate_idr=160_000.0),
    ]

    claims = [
        Claim("CLM-001", "OUTPATIENT", 2_500_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-01"),
        Claim("CLM-002", "INPATIENT", 35_000_000, VerifierRole.SENIOR_ADJUSTER, complexity=2, sla_hours=48, hospital_id="HOSP-02"),
        Claim("CLM-003", "SURGICAL", 120_000_000, VerifierRole.MEDICAL_ADVISOR, complexity=4, sla_hours=72, hospital_id="HOSP-01"),
        Claim("CLM-004", "FRAUD_SUSPECT", 65_000_000, VerifierRole.FRAUD_INVESTIGATOR, complexity=3, sla_hours=48, hospital_id="HOSP-03"),
        Claim("CLM-005", "OUTPATIENT", 4_200_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAMILY-99"),
        Claim("CLM-006", "OUTPATIENT", 3_800_000, VerifierRole.JUNIOR_ADJUSTER, complexity=1, sla_hours=24, hospital_id="HOSP-02", conflict_group="FAMILY-99"),
    ]

    engine = ClaimAllocationEngine(claims, verifiers)
    stats = engine.solve(method="csp_ac3_mrv")

    print(f"\n[Status Solusi CSP: {'BERHASIL (FEASIBLE)' if stats.is_feasible else 'TIDAK LAYAK'}]")
    print(f"Durasi Komputasi : {stats.runtime_ms:.2f} ms")
    print(f"Nodes Diekspansi : {stats.nodes_expanded} state")
    print(f"Jumlah Backtrack : {stats.backtracks} kali")
    print(f"Total Biaya Staf : Rp {stats.total_handling_cost_idr:,.2f}")
    print(f"Deviasi Beban    : {stats.workload_std_dev:.2f} poin kompleksitas")
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
