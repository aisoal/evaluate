# 📓 Kaggle Runner Notebook

Folder ini berisi notebook Jupyter (`.ipynb`) yang berfungsi sebagai **Automated Runner** untuk menjalankan mesin evaluasi pada platform Cloud GPU (Kaggle).

## 💡 Peran Notebook

Notebook ini dirancang untuk menjembatani _dua repositori_:

1.  **`evaluate`**: Mengambil logika perhitungan metrik (skrip Python).
2.  **`experiments`**: Mengambil data mentah (JSON) dan menjadi target pengunggahan hasil (Releases).

## 🚀 Panduan Penggunaan di Kaggle

### 1. Impor ke Kaggle

1.  Download file `.ipynb` dari folder ini.
2.  Buka [Kaggle](https://www.kaggle.com/) dan buat **New Notebook**.
3.  Klik **File** > **Import Notebook** dan unggah file yang tadi di-download.

### 2. Konfigurasi Hardware

Pastikan fitur **Internet** dan **GPU** aktif:

- Settings > Accelerator: **GPU T4 x2** (Direkomendasikan).
- Settings > Internet: **On**.

### 3. Konfigurasi Secrets (Wajib)

Eksperimen ini membutuhkan akses ke GitHub API. Tambahkan kredensial di menu **Add-ons** > **Secrets**:

| Label          | Value                                    |
| :------------- | :--------------------------------------- |
| `github_token` | Personal Access Token (PAT) Classic Anda |
| `github_user`  | Username GitHub Anda                     |
| `github_email` | Email GitHub Anda                        |
| `logic_repo`   | `evaluate`                               |
| `data_repo`    | `experiments`                            |

### 4. Eksekusi

Jalankan seluruh sel (_Run All_). Notebook akan otomatis:

1.  Mengkloning kedua repositori.
2.  Menyinkronkan dataset ke folder kerja.
3.  Menghitung metrik (PPL, BERTScore, dll) menggunakan akselerasi GPU.
4.  Mengompresi hasil ke dalam file ZIP.
5.  Membuat **GitHub Release** baru di repositori `experiments`.

## 📂 Hasil Output

Setelah proses selesai, jangan mencari hasil di Kaggle. Buka repositori **`experiments`** dan periksa tab **Releases**. Hasil evaluasi tersimpan secara permanen di sana sesuai dengan timestamp eksekusi.
