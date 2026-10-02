# ClaimWise AI

**Enterprise AI Copilot untuk Optimasi Triase Klaim & Alokasi Penjadwalan Verifikator Berbasis Algoritma Search (UCS/A*) dan Algoritma Genetika Multi-Objektif (GA Optimization)**

> **Proyek Terpadu (PjBL) — Mata Kuliah: 10S3001 Kecerdasan Buatan (+P)**  
> Semester Gasal 2026/2027 &bull; Program Studi Sarjana Sistem Informasi  
> Fakultas Informatika dan Teknik Elektro (FITE), Institut Teknologi Del  
> **Dosen Pengampu:** Samuel Indra Gunawan Situmeang  
> **Identitas Tim Pengembang (Grup 01):**  
> 1. **Davina Olivia Yosefanny Hutabarat / 12S23001** — *QA, Evaluation & Ethics Lead*  
> 2. **Jodi / 12S23002** — *AI Architect & Model Lead*  
> 3. **Pedro Simangunsong / 12S23003** — *Integration & Interface Engineer*  
>
> **Milestones:**  
> &bull; [Milestone 1 (W02): Baseline Search Triase Klaim (UCS & A*)](docs/Laporan_Tugas01_ClaimWise.md)  
> &bull; [Milestone 2 (W04): Modul Pemecahan Batasan Bisnis (Algoritma Genetika)](docs/Grup01-Tugas02.md) &bull; **PDF Laporan:** [Grup01-Tugas02.pdf](docs/Grup01-Tugas02.pdf) &bull; **Rilis:** `v0.2-milestone2`

---

## 1. Ringkasan Eksekutif & Latar Belakang Bisnis

Perusahaan asuransi kesehatan swasta berskala nasional (**PT Asuransi Sehat Sejahtera Tbk / Inhealth**) memproses puluhan ribu klaim *reimbursement* dan *cashless* setiap bulannya dari lebih dari 1.800 rumah sakit rekanan. Proses verifikasi manual yang bersifat seragam (*one-size-fits-all*) menimbulkan inefisiensi sistemik:
- **Bottleneck Triase Manual:** Waktu penyelesaian rata-rata mencapai **16,8 hari kerja**, melanggar batas maksimal regulasi OJK (POJK No. 69/POJK.05/2016 maksimal 14 hari kerja).
- **Pembengkakan Biaya Operasional (Handling Cost):** Melibatkan verifikator ahli pada semua klaim tanpa stratifikasi risiko menelan biaya operasional rata-rata **Rp 145.000 per klaim**.
- **Fraud Leakage & Human Fatigue:** Beban kerja verifikator yang mencapai 60–80 berkas/hari memicu kejenuhan kognitif, memicu potensi klaim ganda, tagihan fiktif, hingga kebocoran biaya klaim sebesar **7,4%** (Rp 13,3 Miliar/tahun).

**ClaimWise AI** memecahkan masalah ini melalui pendekatan kecerdasan buatan komputasional:
1. **Milestone 1:** Memodelkan alur verifikasi sebagai **graf keputusan ruang keadaan berbobot biaya riil**. Menggunakan algoritma **Uniform Cost Search (UCS)** dan **A* Search**, sistem menentukan rute verifikasi termurah dan tercepat untuk setiap profil risiko klaim.
2. **Milestone 2:** Mengembangkan mesin optimasi kombinatorik berbasis **Algoritma Genetika (Genetic Algorithm - GA)** untuk memecahkan trade-off multi-objektif antara biaya operasional staf, durasi SLA, risiko kebocoran fraud, dan keadilan beban kerja harian staf dengan penegakan batasan hukum ketat (**UU Ketenagakerjaan No. 13/2003** dan regulasi OJK).

---

## 2. Arsitektur Sistem 5-Lapis (Enterprise AI Copilot)

Sesuai dengan cetak biru **Enterprise AI Assistant / Copilot** Institut Teknologi Del, ClaimWise AI dirancang memenuhi arsitektur modular 5-lapis:

```mermaid
flowchart TD
    subgraph Layer5["5. Observability & Guardrails Layer"]
        L5_1["LLM-as-a-Judge Evaluation"]
        L5_2["Strict Delimiter Guardrails"]
        L5_3["Audit Trail Logging & Latency Tracing"]
    end

    subgraph Layer4["4. Interface Layer (Gradio Web Dashboard)"]
        L4_1["Streaming Token Generator"]
        L4_2["Accordion Thought Trace"]
        L4_3["Grounding Citations Panel"]
    end

    subgraph Layer3["3. Agent, Search & GA Optimization Layer"]
        L3_1["ReAct Agent Loop (Google Gemini 2.5 Flash)"]
        L3_2["FastMCP Tool Server (JSON-RPC)"]
        L3_3["Baseline Search Engine (UCS & A* Search)"]
        L3_4["Genetic Algorithm Optimization Engine (Multi-Objective Fitness & Elitism)"]
    end

    subgraph Layer2["2. Knowledge & RAG Layer"]
        L2_1["Word-aware Chunking with Overlap"]
        L2_2["Sentence-Transformers Embeddings"]
        L2_3["ChromaDB Persistent Vector Store"]
    end

    subgraph Layer1["1. Data & Document Layer"]
        L1_1["Digital Claim Form & Invoices (PDF/Image)"]
        L1_2["Optical Character Recognition (OCR OpenCV)"]
        L1_3["Insurance Policy SOP Corpus"]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer5 -. Memantau & Mengamankan .-> Layer3
    Layer5 -. Evaluasi Mutu .-> Layer4

    classDef l1 fill:#E1F5FE,stroke:#0288D1,stroke-width:2px;
    classDef l2 fill:#E8F5E9,stroke:#388E3C,stroke-width:2px;
    classDef l3 fill:#FFF3E0,stroke:#F57C00,stroke-width:2px;
    classDef l4 fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px;
    classDef l5 fill:#FFEBEE,stroke:#D32F2F,stroke-width:2px;

    class Layer1,L1_1,L1_2,L1_3 l1;
    class Layer2,L2_1,L2_2,L2_3 l2;
    class Layer3,L3_1,L3_2,L3_3,L3_4 l3;
    class Layer4,L4_1,L4_2,L4_3 l4;
    class Layer5,L5_1,L5_2,L5_3 l5;
```

---

## 3. Spesifikasi Formal PEAS

| Komponen PEAS | Elemen Spesifik | Kriteria Terukur |
|---|---|---|
| **Performance Measure** | • SLA Compliance Rate<br>• Cost Efficiency<br>• Fast-Track Throughput<br>• Fraud Precision<br>• Zero Hard Violation | • **≥ 98.5%** klaim tuntas dalam batas waktu SLA OJK.<br>• Reduksi rata-rata biaya penanganan klaim **≥ 40%** (dari Rp 145.000 menjadi &lt; Rp 85.000).<br>• **≥ 60%** klaim risiko rendah selesai via *Fast-Track* (&lt; 24 jam).<br>• Presisi deteksi indikasi kecurangan **≥ 95.0%**.<br>• **100% kepatuhan batasan hukum** tanpa pelanggaran batas jam kerja tenaga kerja. |
| **Environment** | • Stakeholders & Modul Terhubung | • Portal klaim nasabah (web/mobile).<br>• SIMRS Rumah Sakit Rekanan.<br>• Basis data Core Insurance (polis, plafon, riwayat).<br>• Verifikator internal, Dokter Penasihat, Investigator Forensik.<br>• Regulasi OJK & Kementerian Ketenagakerjaan. |
| **Actuators** | • Mekanisme Output & Tindakan | • Dispatcher rute triase (*FastTrack* / *StandardReview* / *FraudInvestigation*).<br>• Engine solver alokasi penugasan berkas klaim dan shift kerja staf berbasis GA.<br>• Rekomendasi status keputusan (*Approve* / *Reject* / *Pending Info*).<br>• Pemicu pencairan dana otomatis (*Disbursement trigger*).<br>• Pencatat jejak audit (*Audit Logging System*). |
| **Sensors** | • Instrumen Persepsi Data | • Input formulir klaim terstruktur (JSON payload API).<br>• Ekstraksi berkas tagihan/kuitansi medis (OCR OpenCV).<br>• Kode diagnosis medis terstandarisasi (ICD-10 / ICD-9-CM).<br>• Status keaktifan polis, sisa plafon, dan status beban kerja harian staf verifikator. |

---

## 4. Milestone 2: Modul Optimasi Algoritma Genetika (`solver.py`)

Pada Milestone 2 (W04), sistem mengimplementasikan mesin optimasi komputasional cerdas berbasis **Algoritma Genetika (Genetic Algorithm - GA)** untuk memecahkan trade-off alokasi penugasan klaim asuransi.

### 4.1 Pemodelan Matematis Formal GA

1. **Skema Representasi Kromosom:**  
   Setiap individu merepresentasikan satu solusi alokasi harian ke dalam vektor integer:
   $$\mathbf{g} = \langle g_1, g_2, \dots, g_N \rangle, \quad g_i \in \{0, 1, \dots, K-1\}$$
   Setiap gen $g_i$ memetakan klaim $c_i$ ke salah satu pasangan $(\text{verifier}_j, \text{shift}_k)$ dari total $K$ kombinasi staf dan shift (PAGI, SIANG, MALAM).

2. **Formulasi Fungsi Kebugaran Multi-Objektif & Sistem Penalti Bertingkat:**
   $$f(\mathbf{g}) = \frac{10^6}{1.0 + \text{Penalty}_{\text{total}}(\mathbf{g}) + \text{Term}_{\text{cost}}(\mathbf{g}) + \text{Term}_{\text{balance}}(\mathbf{g}) + \text{Term}_{\text{fraud}}(\mathbf{g})}$$

   - **Sistem Penalti Regulasi Mutlak ($\text{Penalty}_{\text{total}}$):**
     * **Kualifikasi Dokter/Fraud ($\lambda = 25.000$):** Klaim bedah wajib oleh dokter penasihat (*Medical Advisor*); klaim anomali wajib oleh investigator forensik (*Fraud Investigator*).
     * **Limit Wewenang Finansial ($\lambda = 20.000$):** Klaim nominal besar dilarang dialokasikan ke staf junior yang melebihi plafon otorisasi.
     * **Konflik Kepentingan RS ($\lambda = 25.000$):** Staf dilarang memeriksa berkas dari RS terafiliasi secara personal.
     * **Jendela Shift SLA POJK ($\lambda = 15.000$):** Klaim prioritas tinggi (*Fast-Track* $\le 24$ jam) dilarang ditempatkan pada shift malam.
     * **Pemisahan Tugas / Anti-Kolusi ($\lambda = 20.000$):** Berkas dalam kelompok sengketa keluarga yang sama wajib diperiksa oleh verifikator berbeda.
     * **Kapasitas UU Ketenagakerjaan No. 13/2003 ($\lambda = 10.000$):** Penalti kuadratik atas kelebihan poin beban harian staf.
   - **Trade-Off Multi-Objektif Bisnis:**
     * Minimasi Biaya Staf: $\text{Term}_{\text{cost}} = \text{TotalCost}(\mathbf{g}) \times 0.001$.
     * Keadilan Beban Kerja (Anti-Burnout): $\text{Term}_{\text{balance}} = \sigma_{\text{load}}(\mathbf{g}) \times 15.0$.
     * Mitigasi Kerentanan Fraud: $\text{Term}_{\text{fraud}} = \text{FraudExposure}(\mathbf{g}) \times 50.0$.

3. **Operator Evolusioner:**
   - **Inisialisasi Populasi Hibrida:** 30% *heuristic seeding* (kesesuaian kualifikasi awal) + 70% acak untuk diversitas genetik.
   - **Tournament Selection ($k=3$):** Memilih induk kompetitif berdasarkan kebugaran.
   - **Two-Point Crossover ($p_c = 0.85$):** Rekombinasi blok penugasan berkas klaim.
   - **Adaptive Mutation ($p_m = 0.08$):** Mutasi seragam per gen untuk menghindari konvergensi prematur.
   - **Elitism ($E = 2$):** Mempertahankan 2 individu unggul tanpa mutasi langsung ke generasi penerus.

---

## 5. Hasil Analisis Sensitivitas & Tolok Ukur Kinerja GA

Hasil eksperimen pengujian terhadap variasi skala masalah dan kasus ekstrem disajikan di bawah ini:

| Skenario Masalah | Volume Klaim | Verifikator | Populasi | Generasi Selesai | Status Kelayakan | Waktu Komputasi (ms) | Pelanggaran Regulasi | Total Biaya Staf | &sigma; Beban Staf |
|---|---|---|---|---|---|---|---|---|---|
| **Skala Kecil (Small Scale)** | 10 Berkas | 4 Staf | 40 | 46 | **FEASIBLE** | **21,81 ms** | **0** | Rp 422.500 | 4,85 |
| **Skala Sedang (Medium Scale)** | 40 Berkas | 10 Staf | 60 | 106 | **FEASIBLE** | **206,02 ms** | **0** | Rp 3.210.000 | 4,82 |
| **Skala Besar (Large Scale)** | 100 Berkas | 24 Staf | 100 | 200 | *99% FEASIBLE* | **1.535,99 ms** | 1 | Rp 9.780.000 | 6,76 |
| **Kasus Ekstrem: Over-Constrained** | 25 Berkas | 2 Staf | 50 | 100 | **INFEASIBLE** | **100,30 ms** | 68 | Rp 975.000 | 0,00 |
| **Kasus Ekstrem: Bottleneck Spesialis**| 12 Berkas | 3 Staf | 50 | 46 | **FEASIBLE** | **26,05 ms** | **0** | Rp 450.000 | 11,31 |

### Wawasan Kritis Analisis Sensitivitas:
1. **Konvergensi Efisien pada Skala Operasional:** Pada beban 10–40 klaim harian, GA mencapai solusi optimal 100% legal (0 pelanggaran) dalam waktu rata-rata di bawah 210 milidetik.
2. **Ketahanan Kasus Ekstrem Over-Constrained:** Pada kondisi kapasitas staf melampaui kuota legal harian, GA menghentikan evolusi secara anggun (*graceful termination*) tanpa kebuntuan atau *infinite loop*, serta melaporkan pelanggaran kapasitas secara transparan.
3. **Bottleneck Spesialis:** 100% berkas tindakan bedah berhasil dialokasikan secara presisi ke satu-satunya dokter spesialis penasihat dalam tempo 26,05 ms.

---

## 6. Struktur Repositori Terintegrasi

```text
ClaimWise-AI/
├── .gitignore                       # Pengabaian cache Python, env, dan testing
├── LICENSE                          # Lisensi resmi proyek (MIT)
├── pyproject.toml                   # Standar dependensi modern Astral uv (v0.2.0)
├── requirements.txt                 # Dependensi cadangan kompatibilitas pip
├── ucs_search.py                    # Implementasi modul pencarian rute triase (Milestone 1)
├── test_ucs_search.py               # 23 unit test otomatis pencarian jalur (Milestone 1)
├── solver.py                        # Modul solver optimasi GA multi-objektif (Milestone 2)
├── test_solver.py                   # 20 unit test otomatis inferensi batasan & edge cases GA (Milestone 2)
├── docs/
│   ├── Laporan_Tugas01_ClaimWise.md # Dokumen laporan resmi Milestone 1 (W02)
│   ├── Laporan_Tugas01_ClaimWise.html
│   ├── Grup01-Tugas02.md            # Dokumen laporan resmi akademik Milestone 2 (W04)
│   ├── Grup01-Tugas02.html          # Versi cetak web dokumen resmi Milestone 2
│   └── Grup01-Tugas02.pdf           # Dokumen serahan final PDF ECourse Del
└── README.md                        # Dokumentasi komprehensif repositori terintegrasi
```

---

## 7. Instalasi & Panduan Reproduksi (Astral uv)

Proyek ini sepenuhnya terisolasi dan distandarisasi menggunakan manajer paket modern **[Astral uv](https://github.com/astral-sh/uv)**.

### Prasyarat
- Python `>=3.10`
- Astral `uv` terpasang (`curl -LsSf https://astral.sh/uv/install.sh | sh` atau via PowerShell / Winget)

### Langkah-Langkah Eksekusi
```bash
# 1. Kloning repositori dan checkout tag rilis resmi
git clone https://github.com/DavinaHutabarat/ClaimWise-AI.git
cd ClaimWise-AI
git checkout v0.2-milestone2

# 2. Sinkronisasi dependensi secara otomatis
uv sync

# 3. Jalankan demonstrasi modul alokasi klaim berbasis GA
uv run python solver.py

# 4. Jalankan tolok ukur analisis sensitivitas GA (skala kecil, sedang, besar & kasus ekstrem)
uv run python solver.py --benchmark

# 5. Jalankan modul pencarian triase Milestone 1 (UCS & A*)
uv run python ucs_search.py

# 6. Jalankan seluruh rangkaian pengujian otomatis (Milestone 1 + Milestone 2)
uv run pytest -v
```

---

## 8. Hasil Pengujian Otomatis (pytest: 43/43 Passed - 100%)

Seluruh logika triase, heuristik search, representasi kromosom GA, sistem penalti, operator evolusi, kepatuhan batas beban kerja, serta kasus ekstrem diuji oleh **43 unit test otomatis**:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\User\Documents\GitHub\ClaimWise-AI
configfile: pyproject.toml
collected 43 items

test_solver.py::test_role_compatibility_logic PASSED                     [  2%]
test_solver.py::test_evaluation_detects_role_mismatch_penalty PASSED     [  4%]
test_solver.py::test_evaluation_detects_financial_excess_penalty PASSED  [  6%]
test_solver.py::test_evaluation_detects_hospital_conflict_penalty PASSED [  9%]
test_solver.py::test_evaluation_detects_night_shift_sla_penalty PASSED   [ 11%]
test_solver.py::test_evaluation_detects_conflict_group_penalty PASSED    [ 13%]
test_solver.py::test_evaluation_detects_workload_capacity_penalty PASSED [ 16%]
test_solver.py::test_tournament_selection_picks_fitter_individual PASSED [ 18%]
test_solver.py::test_elitism_preserves_best_individuals PASSED           [ 20%]
test_solver.py::test_ga_finds_zero_violation_solution PASSED             [ 23%]
test_solver.py::test_ga_workload_capacity_compliance PASSED              [ 25%]
test_solver.py::test_ga_separation_of_duties_compliance PASSED           [ 27%]
test_solver.py::test_edge_case_zero_claims PASSED                        [ 30%]
test_solver.py::test_edge_case_single_claim_boundary PASSED              [ 32%]
test_solver.py::test_edge_case_over_constrained_graceful_handling PASSED [ 34%]
test_solver.py::test_edge_case_specialist_bottleneck PASSED              [ 37%]
test_solver.py::test_edge_case_clique_of_conflicts PASSED                [ 39%]
test_solver.py::test_edge_case_extreme_claim_amount_exceeds_all PASSED   [ 41%]
test_solver.py::test_seed_reproducibility PASSED                         [ 44%]
test_solver.py::test_cost_breakdown_calculation PASSED                   [ 46%]
test_ucs_search.py::test_low_risk_classification PASSED                  [ 48%]
... (22 unit test Milestone 1 lainnya) ...
test_ucs_search.py::test_negative_edge_weight_raises_error PASSED        [100%]

============================= 43 passed in 0.35s ==============================
```

---

## 9. Distribusi Peran Tim (PjBL)

| Nama Mahasiswa | Peran Enterprise AI | Tanggung Jawab & Kontribusi Teknis |
|---|---|---|
| **Davina Olivia Yosefanny Hutabarat** | **QA, Evaluation & Ethics Lead** | • Perancangan dan validasi suite pengujian otomatis `test_solver.py` (20 unit test).<br>• Validasi kepatuhan hukum batasan UU Ketenagakerjaan No. 13/2003.<br>• Audit etika AI bebas konflik kepentingan dan kepatuhan UU PDP No. 27/2022.<br>• Koordinasi penyusunan analisis sensitivitas GA dan rilis dokumen resmi. |
| **Jodi** | **AI Architect & Model Lead** | • Formulasi formal skema kromosom dan fungsi kebugaran multi-objektif GA.<br>• Implementasi algoritma GA dengan turnamen, crossover dua titik, mutasi adaptif, dan elitisme (`solver.py`).<br>• Kalibrasi sistem penalti bertingkat untuk batasan hukum dan SOP OJK.<br>• Evaluasi konvergensi kebugaran antar generasi dan ketahanan kasus ekstrem. |
| **Pedro Simangunsong** | **Integration & Interface Engineer** | • Konfigurasi lingkungan terpadu Astral `uv` (`pyproject.toml` v0.2.0).<br>• Perancangan antarmuka eksekusi CLI (`--benchmark`) dan integrasi modul.<br>• Dokumentasi teknis komprehensif `README.md`, laporan web, dan rilis tag Git `v0.2-milestone2`. |

---

## 10. Lisensi

Proyek ini dilisensikan di bawah ketentuan lisensi terbuka **[MIT License](./LICENSE)**.
