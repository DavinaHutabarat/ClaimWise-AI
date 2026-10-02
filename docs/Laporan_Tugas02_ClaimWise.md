# Laporan Tugas 2 — ClaimWise AI

## 1. Problem Definition

Pada Milestone 2, ClaimWise AI dikembangkan dengan menambahkan modul
pemecahan keputusan berbasis Constraint Satisfaction Problem (CSP).
Modul ini digunakan untuk menentukan reviewer yang sesuai untuk setiap
klaim dengan mempertimbangkan berbagai batasan bisnis.

Pada proses sebelumnya, ClaimWise AI telah melakukan klasifikasi risiko
klaim dan menentukan jalur verifikasi menggunakan metode pencarian.
Pada Milestone 2, hasil tersebut digunakan sebagai informasi awal untuk
menentukan assignment reviewer.

Masalah yang ingin diselesaikan adalah bagaimana menetapkan setiap klaim
kepada reviewer yang sesuai sehingga seluruh batasan bisnis dapat
dipenuhi secara bersamaan.

Batasan yang dipertimbangkan dalam pemodelan meliputi kesesuaian tingkat
risiko klaim dengan kemampuan reviewer, kebutuhan kemampuan medis,
kapasitas reviewer, dan konflik assignment.

Solver yang dikembangkan menggunakan pendekatan CSP dengan algoritma
AC-3 untuk melakukan constraint propagation dan Backtracking dengan
heuristik Minimum Remaining Values (MRV) untuk mencari assignment yang
valid.

## 2. CSP Modeling

Model Constraint Satisfaction Problem (CSP) pada ClaimWise AI terdiri
dari tiga komponen utama, yaitu variables, domains, dan constraints.
Model ini digunakan untuk merepresentasikan proses penentuan reviewer
yang memenuhi aturan bisnis.

### 2.1 Variables

Setiap klaim yang akan diproses menjadi sebuah variable dalam CSP.
Variable tersebut merepresentasikan reviewer yang akan ditugaskan untuk
menangani klaim.

Secara formal, variable untuk klaim ke-i dinyatakan sebagai:

X_i = reviewer yang ditugaskan untuk claim i

Sebagai contoh, untuk lima klaim yang digunakan pada pengujian awal,
variables direpresentasikan sebagai:

X = {X_CLM001, X_CLM002, X_CLM003, X_CLM004, X_CLM005}

Dengan demikian:

- X_CLM001 merepresentasikan reviewer untuk CLM001.
- X_CLM002 merepresentasikan reviewer untuk CLM002.
- X_CLM003 merepresentasikan reviewer untuk CLM003.
- X_CLM004 merepresentasikan reviewer untuk CLM004.
- X_CLM005 merepresentasikan reviewer untuk CLM005.

Nilai dari setiap variable nantinya dipilih dari domain reviewer yang
memenuhi seluruh constraint yang berlaku.

### 2.2 Domains

Domain merupakan himpunan nilai yang dapat diberikan kepada setiap
variable. Pada ClaimWise AI, domain berisi daftar reviewer yang secara
awal dapat menangani suatu klaim.

Reviewer yang digunakan pada pengujian awal terdiri dari:

- R01: General Reviewer, kapasitas 3 klaim.
- R02: Senior Reviewer, memiliki kemampuan medis, kapasitas 2 klaim.
- R03: Medical Reviewer, memiliki kemampuan medis, kapasitas 2 klaim.
- R04: Fraud Specialist, memiliki kemampuan fraud, kapasitas 1 klaim.

Atribut dan kapasitas reviewer tersebut merupakan asumsi desain untuk
simulasi ClaimWise AI pada Milestone 2.

Berdasarkan karakteristik klaim dan kemampuan reviewer, domain awal
untuk lima klaim adalah:

D(X_CLM001) = {R02, R04}
D(X_CLM002) = {R01, R02, R03}
D(X_CLM003) = {R01}
D(X_CLM004) = {R02}
D(X_CLM005) = {R01, R02}

Domain tersebut merupakan domain awal sebelum proses constraint
propagation. Algoritma AC-3 nantinya dapat mengurangi nilai dalam domain
apabila terdapat nilai yang tidak konsisten dengan constraint.

### 2.3 Constraints

Constraints merupakan aturan yang membatasi nilai yang dapat diberikan
kepada setiap variable. Pada ClaimWise AI, constraint digunakan untuk
memastikan bahwa reviewer yang dipilih sesuai dengan karakteristik klaim
dan kapasitas sistem.

Constraint yang digunakan dalam pemodelan awal terdiri dari empat jenis.

#### 2.3.1 Risk Compatibility

Reviewer harus memiliki tipe yang sesuai dengan tingkat risiko klaim.

Aturan yang digunakan adalah:

- High Risk hanya dapat ditangani oleh Senior Reviewer atau Fraud
  Specialist.
- Medium Risk dapat ditangani oleh General Reviewer atau Senior Reviewer.
- Low Risk dapat ditangani oleh General Reviewer.

Secara sederhana:

High Risk → {R02, R04}

Medium Risk → {R01, R02}

Low Risk → {R01}

Constraint ini digunakan untuk mengurangi domain reviewer yang tidak
sesuai dengan tingkat risiko klaim.

#### 2.3.2 Medical Capability

Klaim yang membutuhkan penanganan medis hanya dapat diberikan kepada
reviewer yang memiliki kemampuan medis.

Reviewer yang memiliki kemampuan medis adalah R02 dan R03.

Dengan demikian, apabila:

medical(X_i) = True

maka:

X_i ∈ {R02, R03}

Constraint ini digunakan untuk memastikan bahwa klaim dengan kebutuhan
medis tidak diberikan kepada reviewer yang tidak memiliki kemampuan
tersebut.

#### 2.3.3 Reviewer Capacity

Setiap reviewer memiliki batas jumlah klaim yang dapat ditangani.

Kapasitas yang digunakan pada pengujian awal adalah:

- R01 maksimal 3 klaim.
- R02 maksimal 2 klaim.
- R03 maksimal 2 klaim.
- R04 maksimal 1 klaim.

Secara formal, untuk setiap reviewer r:

jumlah klaim yang ditugaskan kepada r ≤ kapasitas(r)

Constraint ini diperiksa berdasarkan keseluruhan assignment. Jika suatu
reviewer telah mencapai kapasitasnya, reviewer tersebut tidak dapat
menerima klaim tambahan.

#### 2.3.4 Assignment Conflict

Satu reviewer tidak boleh menangani dua klaim yang memiliki waktu
penanganan yang saling bertabrakan.

Jika dua klaim i dan j memiliki waktu penanganan yang overlap, maka:

X_i ≠ X_j

Constraint ini memastikan bahwa reviewer tidak memperoleh assignment
yang konflik pada waktu yang sama.

## 3. Dataset Pengujian Awal

Untuk pengujian awal solver CSP, digunakan lima data klaim dengan
karakteristik risiko dan kebutuhan medis yang berbeda. Dataset kecil ini
digunakan untuk memvalidasi apakah constraint dapat diterapkan dengan
benar sebelum dilakukan pengujian pada jumlah klaim yang lebih besar.

### 3.1 Data Klaim

| Claim ID | Risk | Medical | Durasi Penanganan |
|---|---|---|---:|
| CLM001 | High | Tidak | 2 |
| CLM002 | Medium | Ya | 3 |
| CLM003 | Low | Tidak | 1 |
| CLM004 | High | Ya | 2 |
| CLM005 | Medium | Tidak | 2 |

Keterangan:

- CLM001 merupakan klaim berisiko tinggi tanpa kebutuhan medis.
- CLM002 merupakan klaim berisiko sedang dengan kebutuhan medis.
- CLM003 merupakan klaim berisiko rendah tanpa kebutuhan medis.
- CLM004 merupakan klaim berisiko tinggi dengan kebutuhan medis.
- CLM005 merupakan klaim berisiko sedang tanpa kebutuhan medis.

### 3.2 Data Reviewer

| Reviewer ID | Tipe Reviewer | Medical | Fraud | Kapasitas |
|---|---|---|---|---:|
| R01 | General Reviewer | Tidak | Tidak | 3 |
| R02 | Senior Reviewer | Ya | Tidak | 2 |
| R03 | Medical Reviewer | Ya | Tidak | 2 |
| R04 | Fraud Specialist | Tidak | Ya | 1 |

Atribut reviewer dan kapasitas pada dataset ini merupakan asumsi desain
untuk kebutuhan simulasi dan pengujian solver ClaimWise AI pada
Milestone 2.

### 3.3 Domain Awal

Berdasarkan karakteristik klaim dan constraint yang telah ditentukan,
domain awal setiap variable adalah:

| Variable | Claim | Domain Awal |
|---|---|---|
| X_CLM001 | CLM001 | {R02, R04} |
| X_CLM002 | CLM002 | {R01, R02, R03} |
| X_CLM003 | CLM003 | {R01} |
| X_CLM004 | CLM004 | {R02} |
| X_CLM005 | CLM005 | {R01, R02} |

Domain tersebut menjadi kondisi awal sebelum solver melakukan proses
constraint propagation dan pencarian solusi.