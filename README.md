<pre align="center">
   ░███   ░██████                             ░██ 
  ░██░██    ░██                               ░██ 
 ░██  ░██   ░██  ░███████  ░███████  ░██████  ░██ 
░█████████  ░██ ░██       ░██    ░██      ░██ ░██ 
░██    ░██  ░██  ░███████ ░██    ░██ ░███████ ░██ 
░██    ░██  ░██        ░██░██    ░██░██   ░██ ░██ 
░██    ░██░██████░███████  ░███████  ░█████░██░██ 
</pre>

# AI Evaluation (Research Engine)

> **Peran:** Metrics Calculator & Statistical Analysis Engine

Modul ini adalah **Evaluation Layer** berbasis Python yang berfungsi sebagai mesin analisis untuk menguji kualitas soal yang dihasilkan oleh LLM. Modul ini tidak berjalan secara real-time dengan aplikasi, melainkan digunakan sebagai alat audit dan analisis statistik mendalam untuk kebutuhan publikasi ilmiah dan tesis.

Sesuai dengan **Bab 3 (Metodologi)** dan **Bab 4 (Hasil)** tesis, modul ini menangani:

1.  **Linguistic Evaluation**: Menghitung _Perplexity_ menggunakan model `SEA-LION` untuk mengukur kelancaran Bahasa Indonesia.
2.  **Semantic Evaluation**: Menghitung _BERTScore_ untuk mengukur keselarasan makna dengan teks sumber.
3.  **Operational Metrics**: Menganalisis _Latency_ (kecepatan) dan _Confidence Score_ dari setiap model.
4.  **Statistical Testing**: Melakukan uji signifikansi otomatis menggunakan metode _Kruskal-Wallis_ dan _One-way ANOVA_ (P-Value).
5.  **Visualization**: Menghasilkan grafik perbandingan performa model (Bar Chart, Heatmap, Distribution Chart).

---

## 🚀 Persiapan & Instalasi

Modul ini membutuhkan spesifikasi hardware yang cukup (disarankan menggunakan GPU/CUDA) karena melakukan pemuatan model bahasa lokal untuk perhitungan metrik.

### 1. Prasyarat

- Python 3.10 atau lebih tinggi.
- NVIDIA GPU dengan CUDA (Opsional, untuk akselerasi).

### 2. Instalasi Dependensi

```bash
# Buat virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows

# Install library
pip install -r requirements.txt
```

### 3. Struktur Data Input

Letakkan file JSON hasil ekspor dari aplikasi AIsoal ke dalam folder `data/`. Skrip akan memproses seluruh file `.json` di folder tersebut secara otomatis.

---

## 🛠️ Cara Penggunaan

Terdapat dua level evaluasi yang tersedia:

### A. Simple Evaluation (`simple_evaluate.py`)

Fokus pada metrik utama penelitian: Perplexity, BERTScore, Latency, dan Uji Statistik Dasar.

```bash
python simple_evaluate.py
```

### B. Deep Evaluation (`deep_evaluate.py`)

Analisis lebih mendalam termasuk ROUGE-L, METEOR, _Embedding Cosine Similarity_, dan _QA Answerability F1_ (menggunakan model XLM-RoBERTa).

```bash
python deep_evaluate.py
```

---

## 📊 Output Hasil (Runs)

Setiap kali skrip dijalankan, sistem akan membuat folder unik di dalam directory `runs/` yang berisi:

- **`universal/tables/`**: Berisi file CSV hasil perhitungan (raw data & summary).
- **`universal/figures/`**: Berisi grafik visualisasi (Combined Metrics, P-Value Heatmap, Type Distribution).
- **`by_type/`**: Analisis mendalam yang dipisah berdasarkan format soal (Pilihan Ganda, Esai, dll).

---

## 🧪 Metrik yang Diuji (Sesuai Bab 3)

| Kategori            | Metrik           | Tujuan                                                    |
| :------------------ | :--------------- | :-------------------------------------------------------- |
| **Kualitas Bahasa** | Perplexity (PPL) | Mengukur seberapa natural kalimat dalam Bahasa Indonesia. |
| **Akurasi Makna**   | BERTScore        | Mengukur kemiripan semantik antara soal dan referensi.    |
| **Keragaman**       | Distinct-n       | Mengukur variasi kosakata yang dihasilkan model.          |
| **Efisiensi**       | Latency/Q        | Mengukur waktu rata-rata pemrosesan per butir soal.       |
| **Kepercayaan**     | Confidence Score | Mengukur keyakinan model terhadap jawabannya.             |
| **Validitas**       | Kruskal-Wallis   | Membuktikan secara ilmiah perbedaan performa antar model. |
| **Validitas**       | One-way ANOVA    | Menguji signifikansi perbedaan rata-rata antar model.     |

---

## 📂 Struktur Repositori

```text
evaluate/
├── data/                 # Tempat menaruh file JSON hasil generate
├── runs/                 # Hasil output (CSV & PNG) otomatis
├── simple_evaluate.py    # Skrip evaluasi standar tesis
├── deep_evaluate.py      # Skrip evaluasi komprehensif
├── requirements.txt      # Daftar library Python
└── README.md
```
