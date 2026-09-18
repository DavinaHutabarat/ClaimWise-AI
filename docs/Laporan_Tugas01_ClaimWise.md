# LAPORAN TUGAS 1 (MILESTONE 1 - W02)
## MATA KULIAH: 10S3001 - KECERDASAN BUATAN (+P)
### SEMESTER GASAL 2026/2027 — INSTITUT TEKNOLOGI DEL

---

**Judul Proyek:** ClaimWise AI — Enterprise Copilot untuk Optimasi Alur Triase dan Verifikasi Klaim Asuransi Kesehatan Berbasis Algoritma Search (UCS & A*)  
**Kode Dokumen:** Grup01-Tugas01.pdf  
**Program Studi:** Sarjana Sistem Informasi  
**Fakultas:** Fakultas Informatika dan Teknik Elektro (FITE), Institut Teknologi Del  
**Dosen Pengampu:** Samuel Indra Gunawan Situmeang  
**Tautan Repositori GitHub Publik:** [https://github.com/DavinaHutabarat/ClaimWise-AI](https://github.com/DavinaHutabarat/ClaimWise-AI)  

---

### IDENTITAS TIM & DISTRIBUSI PERAN PROFESIONAL (PjBL)

| No | NIM | Nama Mahasiswa | Peran Enterprise AI (Panduan PjBL) | Tanggung Jawab Utama |
|---|---|---|---|---|
| 1 | 12S23001 | Davina Olivia Yosefanny Hutabarat | **QA, Evaluation & Ethics Lead** | Validasi integritas data graf biaya, perancangan skrip pengujian otomatis `pytest`, audit etika AI, kepatuhan UU PDP No. 27/2022, koordinasi rilis. |
| 2 | 12S23002 | Jodi | **AI Architect & Model Lead** | Perancangan formulasi ruang keadaan (X, A, T, G, C), perancangan fungsi heuristik h(n), implementasi algoritma UCS & A* (`ucs_search.py`). |
| 3 | 12S23003 | Pedro Simangunsong | **Integration & Interface Engineer** | Standarisasi repositori, konfigurasi paket modern Astral `uv` (`pyproject.toml`), dokumentasi profesional `README.md`, manajemen dependensi dan sinkronisasi Git. |

---

## 1. BUSINESS PROBLEM FRAMING & KARAKTERISTIK BISNIS

### 1.1 Profil Organisasi Enterprise
**PT Asuransi Sehat Sejahtera Tbk (HealthCare Shield Enterprise)** merupakan perusahaan asuransi kesehatan swasta berskala nasional yang melayani lebih dari 3,5 juta pemegang polis aktif dengan jaringan kemitraan mencakup lebih dari 1.800 rumah sakit dan klinik di seluruh Indonesia. Organisasi memproses rata-rata 45.000 klaim *reimbursement* dan *cashless* setiap bulannya dengan nilai perputaran klaim mencapai lebih dari Rp 180 Miliar per kuartal.

### 1.2 Analisis Masalah Bisnis & Pain Points Kuantitatif
Dalam operasional harian, tim manajemen klaim menghadapi tantangan operasional kritis akibat proses triase dan verifikasi yang masih dilakukan secara manual dan bersifat seragam (*one-size-fits-all*). Hal ini memicu beberapa *pain points* utama:

1. **Bottleneck Triase Manual & Pelanggaran SLA OJK:**
   - *Kondisi Riil:* Seluruh klaim yang masuk (baik klaim flu rawat jalan senilai Rp 300.000 maupun klaim operasi bedah kompleks senilai Rp 150.000.000) dialirkan melalui antrean verifikasi manual berjenjang yang identik.
   - *Dampak:* Waktu penyelesaian rata-rata (*claim settlement turnaround time*) membengkak hingga **16,8 hari kerja**, melampaui batas maksimal regulasi OJK (POJK No. 69/POJK.05/2016 yang mewajibkan penyelesaian klaim maksimal 14 hari kerja sejak berkas lengkap diterima).

2. **Pembengkakan Biaya Operasional Penanganan Klaim (Handling Cost):**
   - *Kondisi Riil:* Keterlibatan verifikator senior, dokter penasihat (*medical advisor*), dan investigator forensik pada klaim berisiko rendah menimbulkan pemborosan sumber daya ahli bernilai tinggi.
   - *Dampak:* Biaya operasional penanganan klaim rata-rata mencapai **Rp 145.000 per berkas**, yang menyerap sekitar 8,2% dari total beban operasional tahunan perusahaan.

3. **Risiko Kebocoran Fraud (Fraud Leakage) & False Rejection:**
   - *Kondisi Riil:* Tingginya beban kognitif verifikator (mencapai 60–80 berkas klaim/hari per petugas) menyebabkan kelelahan manusia (*human fatigue*), membuka celah klaim ganda (*double claiming*), tagihan fiktif (*phantom billing*), serta risiko penolakan keliru (*false rejection*) yang merugikan nasabah beritikad baik.
   - *Dampak:* Estimasi kebocoran klaim akibat fraud mencapai **7,4%** dari total pengeluaran klaim bruto (sekitar Rp 13,3 Miliar per tahun).

### 1.3 Justifikasi Pemilihan Solusi AI & Nilai Tambah Bisnis
Sistem berbasis aturan statis (*hardcoded rule engine*) terbukti tidak fleksibel menghadapi variabilitas pola klaim dan kompleksitas penentuan alur kerja karena tidak memiliki kemampuan optimasi multi-kriteria (trade-off waktu vs biaya). 

Implementasi **ClaimWise AI** berbasis penalaran kecerdasan buatan memberikan nilai tambah strategis:
- **Penalaran Simbolik & Optimasi Alur Keputusan (State-Space Search):** Memodelkan alur verifikasi sebagai graf keputusan berbobot biaya riil. Klaim berisiko rendah dialirkan langsung melalui jalur *Fast-Track* (selesai < 24 jam), sementara klaim anomali dialokasikan secara presisi ke dokter penasihat atau investigator audit.
- **Fondasi Copilot Enterprise 5-Lapis:** Menjadi batu pijakan menuju purwarupa sistem asisten cerdas terpadu (*Enterprise AI Assistant*) yang pada milestone berikutnya mengintegrasikan ekstraksi OCR dokumen formulir, penelusuran semantik RAG regulasi polis via ChromaDB, orkestrasi ReAct agent berbasis Google Gemini 2.5 Flash, dan antarmuka interaktif Gradio.

---

## 2. SPESIFIKASI FORMAL PEAS & KARAKTERISTIK LINGKUNGAN OPERASIONAL

### 2.1 Spesifikasi Formal PEAS (Russell & Norvig Standard)

Sistem cerdas **ClaimWise AI** diformulasikan secara formal ke dalam empat pilar PEAS terukur:

| Elemen PEAS | Komponen Detail | Target Kinerja & Spesifikasi Terukur |
|---|---|---|
| **Performance Measure (Ukuran Kinerja)** | • SLA Compliance Rate<br>• Cost Efficiency<br>• Fast-Track Throughput<br>• Fraud Precision<br>• False Rejection Rate | • ≥ 98.5% klaim diselesaikan dalam batas SLA regulasi OJK.<br>• Penurunan rata-rata biaya penanganan klaim sebesar ≥ 40% (dari Rp 145.000 menjadi < Rp 85.000 per klaim).<br>• ≥ 60% klaim berisiko rendah terselesaikan secara otomatis via *Fast-Track* dalam tempo < 24 jam.<br>• Akurasi deteksi anomali/fraud berisiko tinggi ≥ 95.0%.<br>• 0.0% *false rejection* otomatis tanpa proses *human-in-the-loop audit*. |
| **Environment (Lingkungan Kerja)** | • Stakeholders & External Systems | • Portal pengajuan klaim nasabah (web & mobile).<br>• SIMRS (Sistem Informasi Manajemen Rumah Sakit) rekanan.<br>• Basis data Core Insurance (polis, limit plafon, histori klaim).<br>• Verifikator internal, Dokter Penasihat, Tim Fraud Investigator.<br>• Regulator asuransi (OJK) dan Kementerian Kesehatan. |
| **Actuators (Mekanisme Aksi)** | • Decision Routing Engine<br>• Automated Actions<br>• Notification Dispatcher<br>• Audit Logging | • Mengarahkan berkas klaim ke checkpoint optimal (*FastTrack*, *StandardReview*, atau *FraudInvestigation*).<br>• Menerbitkan rekomendasi persetujuan (*Approval*) atau penolakan (*Rejection*).<br>• Mengirim instruksi transfer otomatis (*Disbursement*) ke modul perbankan.<br>• Mengirim notifikasi status real-time ke nasabah dan RS.<br>• Mencatat rekam jejak audit (*immutable audit trail*) untuk keperluan audit kepatuhan. |
| **Sensors (Instrumen Persepsi)** | • Data Capture Channels | • Formulir pengajuan klaim digital (JSON payload via API).<br>• Hasil ekstraksi teks berkas bukti kuitansi & faktur (OCR OpenCV / IDP).<br>• Kode diagnosis penyakit terstandarisasi (ICD-10 / ICD-9-CM).<br>• Basis data status keaktifan polis dan sisa limit plafon nasabah.<br>• Riwayat klaim historis (frekuensi klaim 12 bulan terakhir). |

### 2.2 Analisis Mendalam Karakteristik Lingkungan Operasional (7 Sifat Russell & Norvig)

Berdasarkan taksonomi Russell & Norvig (*Artificial Intelligence: A Modern Approach, 4th Edition*), lingkungan operasional ClaimWise AI diklasifikasikan secara komprehensif:

1. **Partially Observable (Dapat Diamati Sebagian):**
   Agen tidak memiliki akses instan ke seluruh informasi riil terkait niat pemohon klaim atau riwayat medis tersembunyi yang tidak dicantumkan pada dokumen awal. Agen hanya dapat mengamati atribut data yang diserahkan pemohon dan query basis data internal.
2. **Multi-Agent (Competitive & Cooperative):**
   Agen beroperasi di tengah entitas lain: bersifat **kooperatif** terhadap nasabah sah, dokter mitra, dan verifikator internal untuk mempercepat klaim; sekaligus bersifat **kompetitif/adversarial** terhadap sindikat yang mencoba melakukan kecurangan klaim (*fraud syndicates*).
3. **Stochastic (Stokastik) pada Lingkungan Bisnis / Deterministic pada Abstraksi Ruang Keadaan:**
   Secara makro bisnis, lingkungan bersifat stokastik karena kelengkapan berkas dan temuan medis mengandung ketidakpastian. Namun, pada tingkat abstraksi perutean ruang keadaan (*state-space routing*), transisi antar-checkpoint dimodelkan secara deterministik terkendali untuk menjamin kepatuhan audit.
4. **Sequential (Sekuensial):**
   Keputusan triase di tahap awal (validasi berkas dan keaktifan polis) secara langsung menentukan opsi checkpoint berikutnya, akumulasi biaya penanganan, serta hasil akhir klaim. Keputusan masa kini memengaruhi seluruh ruang keputusan masa depan.
5. **Dynamic (Dinamis):**
   Kondisi lingkungan dapat berubah sewaktu-waktu saat agen sedang memproses antrean klaim (misalnya pembaruan tabel plafon polis, perubahan status rekanan RS, atau regulasi tarif medis). Agen tidak dapat berasumsi bahwa lingkungan pasif selama komputasi berlangsung.
6. **Discrete (Diskret):**
   State checkpoint verifikasi bersifat diskret terhitung (`Submitted`, `DocCheck`, `PolicyValidation`, dst.), aksi penentuan rute bersifat diskret, dan biaya dievaluasi pada interval langkah yang jelas.
7. **Known (Diketahui):**
   Aturan bisnis, SOP tahapan verifikasi perusahaan asuransi, ketentuan pasal polis, dan formula pembobotan biaya operasional telah terdefinisi secara terstruktur dalam manual operasional organisasi.

---

## 3. FORMULASI FORMAL RUANG KEADAAN (STATE-SPACE MODEL) & GRAF MASALAH

### 3.1 Formulasi Matematis (X, A, T, G, C)

Ruang keadaan masalah triase klaim didefinisikan secara formal melalui 5 tupel:

1. **State Space (X):**
   Himpunan seluruh kondisi/checkpoint verifikasi klaim:
   `X = {Submitted, DocCheck, PolicyValidation, RiskScoring, FastTrack, StandardReview, MedicalReview, ApprovalOfficer, FraudInvestigation, Disbursement, Rejected}`

2. **Action Space (A):**
   Himpunan aksi keputusan perutean berkas dari satu checkpoint ke checkpoint lainnya:
   `A(s) = {a | transisi valid dari checkpoint s menuju s' dalam X}`

3. **Transition Model (T):**
   Fungsi transisi deterministik berarah:
   `T: X × A -> X`
   Memetakan pasangan state saat ini dan aksi keputusan menjadi tepat satu state penerus.

4. **Goal Test (G):**
   Kondisi terminasi di mana klaim telah mencapai keputusan akhir yang sah dan tervalidasi:
   `G(s) = True <=> s in {Disbursement, Rejected}`

5. **Path Cost Function (C):**
   Biaya kumulatif berbobot non-negatif yang merepresentasikan pengeluaran sumber daya operasional riil:
   `c(s, s') = (Biaya_Staf_IDR / 50.000) + (Ekspektasi_Waktu_SLA_Hari × 0.5)`
   Dengan jaminan bahwa untuk setiap transisi, bobot `c(s, s') >= 1.0 > 0`.

### 3.2 Struktur Topologi Graf & Justifikasi Biaya Operasional Riil

Tabel berikut menyajikan rincian data verifikasi riil yang mendasari bobot edge pada graf keputusan:

| Edge Transisi (u -> v) | Biaya Staf (IDR) | Estimasi SLA | Bobot (c(u, v)) | Justifikasi Operasional & Kriteria Bisnis |
|---|---|---|---|---|
| `Submitted` -> `DocCheck` | Rp 25.000 | 4 Jam (0.17 Hari) | **1.0** | Prapemrosesan intake berkas & OCR digital otomatis. |
| `DocCheck` -> `PolicyValidation` | Rp 50.000 | 8 Jam (0.33 Hari) | **2.0** | Pengecekan status polis, masa tunggu pre-existing condition, & plafon. |
| `PolicyValidation` -> `RiskScoring` | Rp 50.000 | 8 Jam (0.33 Hari) | **2.0** | Kalkulasi skor risiko klaim berdasarkan analitik prediktif. |
| `RiskScoring` -> `FastTrack` | Rp 50.000 | 4 Jam (0.17 Hari) | **3.0** | Klaim risiko rendah (< Rp 5 Juta, rekam medis lengkap, RS rekanan tier-1). |
| `RiskScoring` -> `StandardReview` | Rp 150.000 | 24 Jam (1.0 Hari) | **5.0** | Klaim risiko sedang (tindakan bedah terencana, rawat inap non-kritis). |
| `RiskScoring` -> `FraudInvestigation` | Rp 350.000 | 48 Jam (2.0 Hari) | **9.0** | Klaim berisiko tinggi (anomali tarif RS, klaim berulang frekuensi tinggi). |
| `FastTrack` -> `Disbursement` | Rp 25.000 | 4 Jam (0.17 Hari) | **1.0** | Instruksi transfer dana instan ke rekening nasabah. |
| `StandardReview` -> `MedicalReview` | Rp 125.000 | 16 Jam (0.67 Hari) | **4.0** | Analisis kewajaran tindakan medis oleh dokter spesialis penasihat. |
| `MedicalReview` -> `ApprovalOfficer` | Rp 100.000 | 12 Jam (0.50 Hari) | **3.0** | Validasi akuntansi dan penandatanganan wewenang persetujuan nominal. |
| `ApprovalOfficer` -> `Disbursement` | Rp 25.000 | 4 Jam (0.17 Hari) | **1.0** | Eksekusi transfer pencairan dana pasca persetujuan pejabat berwenang. |
| `FraudInvestigation` -> `ApprovalOfficer` | Rp 200.000 | 24 Jam (1.0 Hari) | **6.0** | Anomali berhasil diklarifikasi sah melalui bukti rekam medis tambahan. |
| `FraudInvestigation` -> `Rejected` | Rp 75.000 | 8 Jam (0.33 Hari) | **2.0** | Penerbitan surat penolakan resmi atas bukti konfirmasi fraud/fiktif. |

---

## 4. IMPLEMENTASI ALGORITMA SEARCH & PEMBUKTIAN MATEMATIS HEURISTIK

### 4.1 Algoritma Uniform Cost Search (UCS)
Algoritma UCS mengekspansi simpul dengan biaya jalur terendah g(n) menggunakan struktur data Binary Min-Heap (`heapq` Python). Untuk mencegah perbandingan tipe data heterogen saat biaya bernilai kembar (*cost tie*), digunakan generator counter unik (`itertools.count()`).

### 4.2 Algoritma A* Search & Formulasi Fungsi Heuristik
Algoritma A* menggunakan fungsi evaluasi f(n) = g(n) + h(n), di mana:
- g(n) adalah biaya riil akumulatif dari state awal `Submitted` menuju state n.
- h(n) adalah estimasi batas bawah optimistik biaya tersisa dari state n menuju goal state terdekat (`Disbursement` atau `Rejected`).

Tabel pemetaan nilai heuristik h(n):
- `Submitted`: 7.0
- `DocCheck`: 6.0
- `PolicyValidation`: 5.0
- `RiskScoring`: 3.0
- `FastTrack`: 1.0
- `StandardReview`: 3.0
- `MedicalReview`: 3.0
- `ApprovalOfficer`: 1.0
- `FraudInvestigation`: 1.0
- `Disbursement`: 0.0
- `Rejected`: 0.0

### 4.3 Pembuktian Matematis Admisibilitas & Konsistensi Heuristik

#### A. Pembuktian Admisibilitas (h(n) <= h*(n))
Heuristik disebut **admissible** jika tidak pernah melebih-lebihkan biaya riil terendah h*(n) menuju goal.
- h*(Submitted) = 9.0 >= h(Submitted) = 7.0 (Terbukti)
- h*(DocCheck) = 8.0 >= h(DocCheck) = 6.0 (Terbukti)
- h*(PolicyValidation) = 6.0 >= h(PolicyValidation) = 5.0 (Terbukti)
- h*(RiskScoring) = 4.0 >= h(RiskScoring) = 3.0 (Terbukti)
- h*(FastTrack) = 1.0 >= h(FastTrack) = 1.0 (Terbukti)
- h*(StandardReview) = 8.0 >= h(StandardReview) = 3.0 (Terbukti)
- h*(MedicalReview) = 4.0 >= h(MedicalReview) = 3.0 (Terbukti)
- h*(ApprovalOfficer) = 1.0 >= h(ApprovalOfficer) = 1.0 (Terbukti)
- h*(FraudInvestigation) = 2.0 >= h(FraudInvestigation) = 1.0 (Terbukti)
- h*(Disbursement) = 0.0 >= h(Disbursement) = 0.0 (Terbukti)
- h*(Rejected) = 0.0 >= h(Rejected) = 0.0 (Terbukti)

Kesimpulan: Untuk seluruh state n dalam X, 0 <= h(n) <= h*(n). Heuristik terbukti Admissible.

#### B. Pembuktian Konsistensi / Monotonisitas (h(n) <= c(n, a, n') + h(n'))
Heuristik disebut **consistent (monotonic)** jika untuk setiap transisi n -> n' dengan bobot c(n, n'), estimasi biaya tidak menurun lebih curam daripada biaya langkah riil:
1. `Submitted` -> `DocCheck`: 7.0 <= 1.0 + 6.0 = 7.0 (Memenuhi)
2. `DocCheck` -> `PolicyValidation`: 6.0 <= 2.0 + 5.0 = 7.0 (Memenuhi)
3. `PolicyValidation` -> `RiskScoring`: 5.0 <= 2.0 + 3.0 = 5.0 (Memenuhi)
4. `RiskScoring` -> `FastTrack`: 3.0 <= 3.0 + 1.0 = 4.0 (Memenuhi)
5. `RiskScoring` -> `StandardReview`: 3.0 <= 5.0 + 3.0 = 8.0 (Memenuhi)
6. `RiskScoring` -> `FraudInvestigation`: 3.0 <= 9.0 + 1.0 = 10.0 (Memenuhi)
7. `FastTrack` -> `Disbursement`: 1.0 <= 1.0 + 0.0 = 1.0 (Memenuhi)
8. `StandardReview` -> `MedicalReview`: 3.0 <= 4.0 + 3.0 = 7.0 (Memenuhi)
9. `MedicalReview` -> `ApprovalOfficer`: 3.0 <= 3.0 + 1.0 = 4.0 (Memenuhi)
10. `ApprovalOfficer` -> `Disbursement`: 1.0 <= 1.0 + 0.0 = 1.0 (Memenuhi)
11. `FraudInvestigation` -> `ApprovalOfficer`: 1.0 <= 6.0 + 1.0 = 7.0 (Memenuhi)
12. `FraudInvestigation` -> `Rejected`: 1.0 <= 2.0 + 0.0 = 2.0 (Memenuhi)

Kesimpulan: Seluruh transisi memenuhi pertidaksamaan segitiga h(n) <= c(n, n') + h(n'). Heuristik terbukti Konsisten. Dalam Graph Search dengan explored set, tidak ada node yang perlu di-reopen.

### 4.4 Evaluasi Empiris & Perbandingan Kinerja

Hasil eksekusi komparasi antara UCS dan A* Search pada modul `ucs_search.py`:

| Parameter Evaluasi | Uniform Cost Search (UCS) | A* Search (Informed) | Analisis Komparasi |
|---|---|---|---|
| **Jalur Keputusan Terpilih** | `Submitted` -> `DocCheck` -> `PolicyValidation` -> `RiskScoring` -> `FastTrack` -> `Disbursement` | `Submitted` -> `DocCheck` -> `PolicyValidation` -> `RiskScoring` -> `FastTrack` -> `Disbursement` | Identik (Jalur Optimal Mutlak) |
| **Total Biaya Ternormalisasi** | **9.00** | **9.00** | Selisih 0.00 (Optimalitas Terverifikasi) |
| **Node Diekspansi** | 6 state | 6 state | Sempurna |
| **Puncak Ukuran Frontier** | 3 state | 3 state | Sangat hemat memori |
| **Jaminan Optimalitas** | Dijamin (seluruh bobot c >= 0) | Dijamin (heuristik admissible & konsisten) | Terbukti secara matematis & empiris |

---

## 5. STANDAR REKAYASA PERANGKAT LUNAK & REPOSITORI GITHUB

### 5.1 Struktur Repositori Terstandarisasi
Repositori GitHub dirancang mengikuti struktur modul Python profesional yang bersih, terpisah antara kode inti, pengujian, konfigurasi, dan dokumentasi:

```text
ClaimWise-AI/
├── .gitignore                     # Pengabaian cache Python, env, dan testing
├── LICENSE                        # Lisensi Open-Source MIT
├── pyproject.toml                 # Standar packaging Astral uv & dependency-groups
├── requirements.txt               # Format dependensi portabel
├── ucs_search.py                  # Modul implementasi UCS & A* Search
├── test_ucs_search.py             # 14 unit test otomatis pytest (100% Passed)
├── docs/
│   └── Laporan_Tugas01_ClaimWise.md # Berkas sumber naskah laporan formal W02
└── README.md                      # Dokumentasi komprehensif repositori & diagram
```

### 5.2 Manajemen Lingkungan Cepat dengan Astral uv
Sesuai instruksi penugasan, proyek menggunakan manajer paket modern **Astral uv**. Konfigurasi berkas `pyproject.toml` telah dimutakhirkan ke standar spesifikasi PEP 621 / PEP 735:

```toml
[project]
name = "claimwise-ai"
version = "0.1.0"
description = "ClaimWise AI — Enterprise Copilot untuk Optimasi Alur Triase dan Verifikasi Klaim Asuransi Kesehatan Berbasis Algoritma Search (UCS & A*)"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

[dependency-groups]
dev = [
    "pytest>=8.0.0",
]

[tool.uv]
package = false
```

Instruksi eksekusi:
```bash
# 1. Sinkronisasi dependensi secara deterministik
uv sync

# 2. Eksekusi program pencarian rute utama
uv run python ucs_search.py

# 3. Eksekusi pengujian otomatis menyeluruh
uv run pytest -v
```

### 5.3 Hasil Pengujian Otomatis (pytest Suite)
Pengujian otomatis mencakup 14 skenario uji yang memvalidasi seluruh fungsionalitas dan integritas matematika:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
collected 14 items

test_ucs_search.py::test_finds_a_path_to_disbursement PASSED             [  7%]
test_ucs_search.py::test_picks_the_cheapest_route_not_just_any_route PASSED [ 14%]
test_ucs_search.py::test_astar_finds_identical_optimal_path PASSED       [ 21%]
test_ucs_search.py::test_compare_searches_utility PASSED                 [ 28%]
test_ucs_search.py::test_heuristic_is_admissible PASSED                  [ 35%]
test_ucs_search.py::test_heuristic_is_consistent_monotonic PASSED        [ 42%]
test_ucs_search.py::test_heuristic_goal_states_are_zero PASSED           [ 50%]
test_ucs_search.py::test_all_edge_weights_are_strictly_positive PASSED   [ 57%]
test_ucs_search.py::test_goal_state_has_no_outgoing_edges PASSED         [ 64%]
test_ucs_search.py::test_cost_breakdown_matches_graph_weights PASSED     [ 71%]
test_ucs_search.py::test_unreachable_goal_returns_none PASSED            [ 78%]
test_ucs_search.py::test_target_specific_fraud_rejection PASSED          [ 85%]
test_ucs_search.py::test_cycle_resistance_explored_set PASSED            [ 92%]
test_ucs_search.py::test_negative_edge_weight_raises_error PASSED        [100%]

============================== 14 passed in 0.03s ==============================
```

---

## 6. ETIKA AI, PERLINDUNGAN DATA PRIBADI, & TATA KELOLA

### 6.1 Kepatuhan terhadap UU PDP No. 27 Tahun 2022
1. **Pemrosesan Data Sensitif Medis (Pasal 4 & Pasal 28):**
   Data klaim asuransi kesehatan memuat riwayat medis dan diagnosis nasabah yang tergolong sebagai *Data Pribadi Spesifik*. Sistem ClaimWise AI dirancang dengan prinsip minimisasi data (*data minimization*), di mana hanya atribut yang relevan untuk kalkulasi risiko yang diproses oleh algoritma pencarian rute.
2. **Persetujuan & Transparansi (Pasal 20):**
   Nasabah memiliki hak untuk mengetahui dasar pengambilan keputusan pemrosesan klaim. Rute keputusan penelusuran graf bersifat sepenuhnya deterministik dan dapat diaudit secara transparan (*auditable and explainable AI*).

### 6.2 Prinsip Etika Kecerdasan Artifisial (SE Menkominfo No. 9/2023)
1. **Inklusivitas & Non-Diskriminasi:**
   Bobot penanganan klaim murni dihitung berdasarkan kompleksitas medis klinis dan parameter objektivitas polis, tanpa memasukkan parameter demografis yang berpotensi memicu bias sosio-ekonomi atau diskriminasi rasial.
2. **Pengawasan Manusia (Human-in-the-Loop):**
   Sistem ClaimWise AI berfungsi sebagai *Decision-Support Copilot*, bukan eksekutor otomatis tanpa kontrol. Pada alur penolakan klaim (`Rejected`), sistem mewajibkan verifikasi dan tanda tangan pejabat peninjau manusia (*Approval Officer*) guna melindungi hak nasabah dari kesalahan algoritma (*false rejection mitigation*).

---

## 7. KESIMPULAN & ROADMAP MILESTONE SELANJUTNYA

### 7.1 Kesimpulan Capaian Milestone 1
1. **Problem Framing Tajam:** Telah terumuskan profil bisnis enterprise asuransi kesehatan, identifikasi *pain points* kuantitatif (SLA 16,8 hari, handling cost Rp 145.000/klaim, kebocoran fraud 7,4%), dan justifikasi adopsi penalaran AI.
2. **Spesifikasi Formal PEAS & Karakteristik Lingkungan:** Telah terdefinisi secara terukur metrik kinerja, instrumen aktuator/sensor, dan klasifikasi komprehensif 7 dimensi lingkungan Russell & Norvig.
3. **Formulasi Ruang Keadaan & Algoritma:** Model (X, A, T, G, C) telah diimplementasikan dalam skrip `ucs_search.py` menggunakan UCS dan A* Search bebas bug (`heapq`), lengkap dengan pembuktian matematis sifat heuristik *admissible* dan *consistent*.
4. **Kualitas Repositori & Otomasi:** Repositori GitHub telah terstandarisasi dengan Astral `uv`, lulus 14/14 unit test `pytest`, serta didokumentasikan secara profesional.

### 7.2 Roadmap Menuju Milestone 2 (W04)
Pada penugasan berikutnya (Tugas 2 - Milestone 2 - W04: *Business Constraint Solver*), tim akan memodelkan batasan operasional penugasan verifikator klaim menggunakan pendekatan **Constraint Satisfaction Problem (CSP)** dengan algoritma propagasi batasan AC-3 dan pencarian *Backtracking (MRV/LCV)* atau *Genetic Algorithm (GA)* untuk optimasi alokasi beban kerja harian dokter penasihat.

---

## LAMPIRAN & TAUTAN SERAHAN

- **Tautan Repositori GitHub:** [https://github.com/DavinaHutabarat/ClaimWise-AI](https://github.com/DavinaHutabarat/ClaimWise-AI)
- **Rekam Jejak Commit Terpadu:**
  - `edd1bce`: Penambahan unit test untuk validasi jalur & cost optimal UCS (`DavinaHutabarat`)
  - `3aa255f`: Inisialisasi struktur repositori, lisensi, gitignore, dan konfigurasi Astral uv (`DavinaHutabarat`)
  - `d5b20f6`: Implementasi formal state-space X, A, T, G, C dan Uniform Cost Search (`DavinaHutabarat`)
  - `711f824`: Dokumentasi README awal dan spesifikasi teknis (`githubPedroSimangunsong`)
  - `ceea2f9`: Setup project configuration dan dependensi (`DavinaHutabarat`)
  - `8f9d9dc`: Penambahan modul baseline search (`DavinaHutabarat`)
  - `af7f45a`: Initial commit repositori organisasi (`DavinaHutabarat`)
