# StuntLytics – Dashboard Analisis Stunting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg)


StuntLytics adalah sebuah aplikasi dashboard interaktif yang dibangun dengan Streamlit untuk analisis, monitoring, dan prediksi risiko stunting. Aplikasi ini dirancang untuk membantu pemerintah daerah dalam mengambil keputusan berbasis data untuk intervensi yang lebih tepat sasaran.



## ✨ Fitur Utama

-   **Dashboard Utama**: Ringkasan eksekutif dengan Key Performance Indicators (KPI) utama yang ter-update secara real-time sesuai filter.
-   **Arsitektur Multi-Halaman**: Setiap fitur diisolasi dalam halaman terpisah untuk skalabilitas dan kemudahan maintenance.
-   **Peta Risiko Interaktif**: Visualisasi sebaran keluarga berisiko stunting dalam bentuk heatmap dan titik data menggunakan Pydeck.
-   **Explorer Data**: Fitur untuk melakukan drill-down dan filtering lanjutan pada data mentah.
-   **Prediksi Stunting (Lokal)**: Memanfaatkan model Machine Learning (`.joblib`) yang di-load secara lokal untuk memprediksi risiko stunting pada individu baru melalui form interaktif.
-   **Analisis Tren & Korelasi**: Menampilkan tren proporsi risiko dari waktu ke waktu dan korelasi antar variabel.
-   **Desain Modern**: Antarmuka dengan tema gelap (dark mode) yang bersih dan profesional.

## 🛠️ Teknologi yang Digunakan

-   **Framework**: Streamlit
-   **Analisis Data**: Pandas, NumPy
-   **Machine Learning**: Scikit-learn
-   **Visualisasi**: Pydeck, Plotly (via Streamlit native charts)

## 📂 Struktur Proyek

Proyek ini disusun dengan arsitektur *multi-page* untuk menjaga *separation of concerns* (SoC) dan kemudahan pengembangan.

```

StuntLytics/
│
├── 📄 app.py                 \# Halaman utama / landing page (KPI Dashboard)
├── 📂 pages/                  \# Semua halaman lain ada di sini
│   ├── 📄 1\_Peta\_Risiko.py
│   └── 📄 ... (halaman lainnya)
├── 📂 src/                    \# Modul-modul reusable (jeroan aplikasi)
│   ├── 📂 components/
│   │   └── 📄 sidebar.py     \# Logika untuk filter sidebar global
│   ├── 📄 config.py          \# Konfigurasi statis (judul, API, rules)
│   ├── 📄 data\_loader.py     \# Fungsi untuk memuat dan membersihkan data
│   ├── 📄 model\_loader.py    \# Fungsi untuk memuat dan menjalankan model ML
│   ├── 📄 styles.py          \# Kumpulan CSS untuk styling
│   └── 📄 utils.py           \# Fungsi-fungsi helper
├── 📂 data/                   \# Tempat untuk menyimpan dataset CSV (opsional)
│   └── 📄 data\_keluarga.csv
├── 📂 models/                 \# Tempat untuk menyimpan file model ML
│   └── 📄 stunting\_model.joblib
└── 📄 requirements.txt         \# Daftar dependensi Python

````

## 🚀 Cara Menjalankan Proyek Secara Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda.

### Prasyarat

-   Python 3.9 atau yang lebih baru
-   Git

### 1. Clone Repository

Buka terminal atau command prompt Anda dan jalankan perintah berikut:

```bash
git clone https://github.com/yoga220802/StuntLytics.git
cd StuntLytics
````

### 2\. Buat dan Aktifkan Virtual Environment

Sangat disarankan untuk menggunakan *virtual environment* agar tidak mengganggu instalasi Python global Anda.

```bash
# Buat environment (cukup sekali)
python -m venv venv

# Aktifkan environment
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3\. Install Dependensi

Install semua library yang dibutuhkan yang tercantum di `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4\. ⚠️ Siapkan Model Machine Learning

Aplikasi ini memerlukan file model klasifikasi untuk fitur prediksi.

1.  Buat sebuah folder baru bernama `models` di dalam direktori root `StuntLytics`.
2.  Letakkan file model Anda yang bernama `stunting_model.joblib` ke dalam folder `models` tersebut.

Struktur akhir folder akan terlihat seperti ini:
`StuntLytics/models/stunting_model.joblib`

### 5\. Jalankan Aplikasi Streamlit

Setelah semua siap, jalankan aplikasi dengan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser default Anda. Selamat\!
