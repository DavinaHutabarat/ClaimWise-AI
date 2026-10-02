"""
ClaimWise AI - CSP Solver
Milestone 2

Solusi Constraint Satisfaction Problem (CSP)
untuk penugasan claim kepada reviewer.
"""

# ============================================================
# DATA REVIEWER
# ============================================================

REVIEWERS = {
    "R01": {
        "name": "General Reviewer",
        "medical": False,
        "fraud": False,
        "capacity": 3
    },
    "R02": {
        "name": "Senior Reviewer",
        "medical": True,
        "fraud": False,
        "capacity": 2
    },
    "R03": {
        "name": "Medical Reviewer",
        "medical": True,
        "fraud": False,
        "capacity": 2
    },
    "R04": {
        "name": "Fraud Specialist",
        "medical": False,
        "fraud": True,
        "capacity": 1
    }
}


# ============================================================
# DATA CLAIM
# ============================================================

CLAIMS = {
    "CLM001": {
        "risk": "High",
        "medical": False,
        "duration": 2
    },
    "CLM002": {
        "risk": "Medium",
        "medical": True,
        "duration": 3
    },
    "CLM003": {
        "risk": "Low",
        "medical": False,
        "duration": 1
    },
    "CLM004": {
        "risk": "High",
        "medical": True,
        "duration": 2
    },
    "CLM005": {
        "risk": "Medium",
        "medical": False,
        "duration": 2
    }
}


# ============================================================
# INITIAL DOMAIN
# ============================================================

DOMAINS = {
    "CLM001": {"R02", "R04"},
    "CLM002": {"R01", "R02", "R03"},
    "CLM003": {"R01"},
    "CLM004": {"R02"},
    "CLM005": {"R01", "R02"}
}


def revise(domains, claim_x, claim_y):
    """
    Menghapus nilai dari domain claim_x jika
    tidak memiliki pasangan nilai yang valid
    pada domain claim_y.

    Constraint konflik hanya berlaku untuk
    pasangan claim yang memang saling konflik.
    """

    conflicting_pairs = {
        frozenset(("CLM001", "CLM004")),
        frozenset(("CLM002", "CLM005"))
    }

    pair = frozenset((claim_x, claim_y))

    # Jika kedua claim tidak saling konflik,
    # tidak perlu melakukan revisi domain.
    if pair not in conflicting_pairs:
        return False

    revised = False

    for reviewer_x in list(domains[claim_x]):

        supported = False

        for reviewer_y in domains[claim_y]:

            # Claim yang konflik tidak boleh
            # menggunakan reviewer yang sama.
            if reviewer_x != reviewer_y:
                supported = True
                break

        if not supported:
            domains[claim_x].remove(reviewer_x)
            revised = True

    return revised


def ac3(domains):
    """
    Algoritma AC-3 untuk melakukan constraint propagation
    pada domain setiap claim.
    """

    # Membuat queue berisi semua pasangan claim
    queue = []

    claims = list(domains.keys())

    for claim_x in claims:
        for claim_y in claims:
            if claim_x != claim_y:
                queue.append((claim_x, claim_y))

    # Proses queue
    while queue:

        claim_x, claim_y = queue.pop(0)

        if revise(domains, claim_x, claim_y):

            # Jika domain kosong, CSP tidak memiliki solusi
            if len(domains[claim_x]) == 0:
                return False

            # Jika domain berubah, masukkan kembali
            # pasangan yang berkaitan dengan claim_x
            for claim_z in claims:

                if claim_z != claim_x and claim_z != claim_y:
                    queue.append((claim_z, claim_x))

    return True

def print_domains(domains):
    """
    Menampilkan domain setiap claim.
    """

    for claim_id, domain in domains.items():
        print(
            f"{claim_id} -> {sorted(domain)}"
        )

 

if __name__ == "__main__":
    print("ClaimWise AI - CSP Solver")
    print("=" * 30)

    print("\nReviewers:")
    for reviewer, data in REVIEWERS.items():
        print(reviewer, "-", data["name"])

    print("\nClaims:")
    for claim, data in CLAIMS.items():
        print(claim, "-", data["risk"])

    print("\nInitial Domains:")
    for claim, domain in DOMAINS.items():
        print(claim, "->", sorted(domain))
        
  # ============================================================
  # CONSTRAINT FUNCTIONS
  # ============================================================

def check_risk_compatibility(claim_id, reviewer_id):
    """
    Mengecek apakah reviewer sesuai dengan tingkat risiko claim.
    """

    risk = CLAIMS[claim_id]["risk"]

    if risk == "High":
        return reviewer_id in {"R02", "R04"}

    elif risk == "Medium":
        return reviewer_id in {"R01", "R02"}

    elif risk == "Low":
        return reviewer_id == "R01"

    return False

def check_medical_capability(claim_id, reviewer_id):
    """
    Mengecek apakah reviewer memiliki kemampuan
    untuk menangani claim medis.
    """

    is_medical_claim = CLAIMS[claim_id]["medical"]

    if not is_medical_claim:
        return True

    return REVIEWERS[reviewer_id]["medical"]


def check_capacity(assignment):
    """
    Mengecek apakah jumlah claim yang diberikan
    kepada setiap reviewer tidak melebihi kapasitasnya.
    """

    workload = {}

    for reviewer_id in REVIEWERS:
        workload[reviewer_id] = 0

    for reviewer_id in assignment.values():
        workload[reviewer_id] += 1

    for reviewer_id, count in workload.items():
        capacity = REVIEWERS[reviewer_id]["capacity"]

        if count > capacity:
            return False

    return True


def check_assignment_conflict(assignment):
    """
    Mengecek apakah terdapat konflik assignment.
    
    Untuk tahap awal, dua claim dianggap konflik
    apabila diberikan kepada reviewer yang sama
    dan keduanya sedang diproses bersamaan.
    """

    # Untuk dataset awal, conflict akan ditangani
    # berdasarkan pasangan claim yang sudah ditentukan.
    
    conflicting_pairs = {
        ("CLM001", "CLM004"),
        ("CLM002", "CLM005")
    }

    for claim_a, reviewer_a in assignment.items():
        for claim_b, reviewer_b in assignment.items():

            if claim_a >= claim_b:
                continue

            pair = (claim_a, claim_b)

            if pair in conflicting_pairs:
                if reviewer_a == reviewer_b:
                    return False

    return True

def is_value_consistent(claim_id, reviewer_id):
    """
    Mengecek apakah reviewer memenuhi constraint
    dasar untuk sebuah claim.
    """

    # Constraint 1: Risk Compatibility
    if not check_risk_compatibility(claim_id, reviewer_id):
        return False

    # Constraint 2: Medical Capability
    if not check_medical_capability(claim_id, reviewer_id):
        return False

    return True


if __name__ == "__main__":
    print("ClaimWise AI - CSP Solver")
    print("=" * 30)

    print("\nTesting Risk Compatibility:")

    tests = [
        ("CLM001", "R02"),
        ("CLM001", "R01"),
        ("CLM003", "R01"),
        ("CLM003", "R02")
    ]

    for claim_id, reviewer_id in tests:
        result = check_risk_compatibility(claim_id, reviewer_id)

        print(
            f"{claim_id} -> {reviewer_id}: {result}"
        )

    print("\nTesting Medical Capability:")

    tests_medical = [
        ("CLM002", "R02"),
        ("CLM002", "R03"),
        ("CLM002", "R01"),
        ("CLM003", "R01")
    ]

    for claim_id, reviewer_id in tests_medical:
        result = check_medical_capability(
            claim_id,
            reviewer_id
        )

        print(
            f"{claim_id} -> {reviewer_id}: {result}"
        )

    print("\nTesting Capacity:")

    assignment = {
        "CLM001": "R02",
        "CLM004": "R02"
    }

    print(
        "Assignment:",
        assignment
    )

    print(
        "Capacity valid:",
        check_capacity(assignment)
    )

    print("\nTesting Assignment Conflict:")

    print(
        "Conflict valid:",
        check_assignment_conflict(assignment)
    )


    print("\nTesting AC-3:")

    test_domains = {
        claim_id: set(domain)
        for claim_id, domain in DOMAINS.items()
    }

    print("\nDomains Before AC-3:")
    print_domains(test_domains)

    result = ac3(test_domains)

    print("\nAC-3 Result:", result)

    print("\nDomains After AC-3:")
    print_domains(test_domains)