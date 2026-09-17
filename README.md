# ClaimWise AI

**Enterprise Copilot untuk Optimasi Alur Triase dan Verifikasi Klaim Asuransi Kesehatan Berbasis Uniform Cost Search (UCS)**

> Tugas Kelompok — Milestone 1
> Kelompok: 

---

## 1. Latar Belakang

Perusahaan asuransi kesehatan swasta berskala nasional memproses puluhan ribu klaim reimbursement setiap bulan. Proses verifikasi manual yang bersifat *one-size-fits-all* menyebabkan:

- **Penumpukan berkas** — klaim kecil diverifikasi dengan prosedur yang sama panjangnya dengan klaim besar.
- **Biaya penanganan membengkak** — tenaga ahli (dokter penasihat, investigator forensik) dilibatkan pada semua jenis klaim tanpa prioritas.
- **Risiko human error & fraud** — beban kerja verifikator manual yang tinggi membuka celah *double claiming* dan *false rejection*.
- **Pelanggaran SLA** — keterlambatan verifikasi memicu komplain nasabah dan sanksi regulasi.

**ClaimWise AI** menyelesaikan ini dengan memodelkan proses verifikasi sebagai **graf pencarian rute keputusan** (*decision routing graph*), di mana setiap klaim diarahkan melalui kombinasi *checkpoint* verifikasi termurah dan tercepat menggunakan algoritma **Uniform Cost Search (UCS)**.

## 2. Struktur Masalah Sebagai Ruang Keadaan (State Space)

| Komponen | Definisi |
|---|---|
| **X** (States) | Setiap *checkpoint* / tahap verifikasi yang telah dilalui suatu klaim (mis. `Intake`, `DocScan`, `PolicyCheck`, `MedicalReview`, `FraudInvestigation`, `Approved`, `Rejected`) |
| **A** (Actions) | Perpindahan dari satu checkpoint ke checkpoint lain (mis. lanjut ke fast-track, atau eskalasi ke investigasi mendalam) |
| **T** (Transition) | Fungsi deterministik yang menentukan checkpoint selanjutnya berdasarkan aksi yang dipilih |
| **G** (Goal Test) | Klaim mencapai status akhir: `Approved` atau `Rejected` dengan seluruh syarat audit terpenuhi |
| **C** (Cost) | Biaya gabungan waktu proses + biaya tenaga ahli pada tiap checkpoint (bobot non-negatif, sesuai prasyarat UCS) |

## 3. Mengapa Uniform Cost Search (UCS)?

- Setiap edge pada graf checkpoint memiliki **bobot biaya berbeda** (bukan uniform seperti BFS), sehingga UCS lebih tepat dibanding BFS/DFS biasa.
- UCS menjamin solusi **optimal** (rute termurah) selama seluruh bobot ≥ 0 — sesuai karakteristik biaya operasional riil (waktu & tenaga kerja tidak pernah negatif).
- Dibandingkan A*, UCS dipilih pada milestone ini karena belum ada *heuristic* domain yang divalidasi; UCS menjadi baseline yang aman dan mudah diaudit (penting untuk aspek etika/*explainability* pada sistem yang memengaruhi keputusan finansial nasabah).

## 4. Verifikasi (dijalankan oleh Integration Engineer)

Kode `ucs_search.py` sudah diverifikasi berjalan dengan benar:

```
=== ClaimWise AI — Uniform Cost Search ===
Start : Submitted
Goals : {'Disbursement', 'Rejected'}
Jalur optimal ditemukan:
  Submitted -> DocCheck -> PolicyValidation -> RiskScoring -> FastTrack -> Disbursement
Total biaya (handling + waktu, ternormalisasi): 9.00
```

Jalur `FastTrack` (total biaya 9.00) benar merupakan rute termurah dibanding alternatif via `StandardReview` (18.00) atau `FraudInvestigation` (16.00–21.00), sesuai jaminan optimalitas UCS.

## 5. Struktur Repository

```
claimwise-ai/
├── README.md                  # Dokumen ini
├── LICENSE
├── .gitignore
├── pyproject.toml             # Dependency management (uv)
├── ucs_search.py              # Implementasi UCS + data graf checkpoint 
├── test_ucs_search.py         # Unit test 
└── docs/
    └── Laporan_Milestone1_ClaimWise.pdf
```

> **Catatan integrasi:** Data graf checkpoint (`GRAPH`) saat ini didefinisikan langsung sebagai dictionary di dalam `ucs_search.py` (lihat bagian *"Problem definition"*), bukan file terpisah. Validasi nilai biaya/waktu tiap checkpoint tetap menjadi tanggung jawab **Davina (Data & Knowledge Engineer)** — dilakukan lewat review pull request pada dictionary `GRAPH`, bukan lewat file JSON tersendiri.

## 6. Instalasi & Menjalankan

Project ini menggunakan [`uv`](https://github.com/astral-sh/uv) sebagai package manager.

```bash
# 1. Clone repo
git clone <url-repo-ini>
cd claimwise-ai

# 2. Install dependencies
uv sync

# 3. Jalankan program utama
uv run python ucs_search.py

# 4. Jalankan unit test
uv run pytest -v
```

## 7. Pembagian Kerja Tim

| Nama | Role | Tanggung Jawab | Lokasi Kerja |
|---|---|---|---|
| Davina | AI Architect & Model Lead | Desain graf checkpoint klaim, implementasi algoritma UCS (`ucs_search.py`) | Lokal → push ke root repo |
| Jodi | Data & Knowledge Engineer | Validasi & penyusunan data graf (biaya/waktu tiap checkpoint) | Lokal → push ke repo (`data/`) |
| Pedro | QA, Evaluation & Ethics Lead | Penulisan `test_ucs_search.py`, eksekusi `uv run pytest`, penulisan bagian etika laporan | Lokal + dokumen laporan |
| Pedro | Integration & Interface Engineer | Penyusunan README, LICENSE, `.gitignore`, `pyproject.toml`, menjaga keseimbangan commit history | GitHub |
| Semua | — | Mengisi tautan repo & commit di laporan, review bersama, export PDF, upload ke ECourse | Dokumen laporan bersama |

## 8. Catatan Etika

Sistem ini bersifat **decision-support**, bukan pengambil keputusan final otomatis penuh — setiap klaim yang dialihkan ke jalur *Fast-Track* maupun *investigasi mendalam* tetap dapat diaudit ulang oleh manusia untuk mencegah *false rejection* yang merugikan nasabah. Detail lebih lanjut ada di Bagian 4.2 laporan.

## 9. Lisensi

Didistribusikan di bawah [MIT License](./LICENSE).
