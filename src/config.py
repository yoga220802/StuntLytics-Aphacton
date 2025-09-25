import os
from dotenv import load_dotenv

# Muat environment variables dari file .env
load_dotenv()

# --- Konfigurasi Koneksi Elasticsearch ---
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
STUNTING_INDEX = os.getenv("STUNTING_INDEX", "stunting-data")
BALITA_INDEX = os.getenv("BALITA_INDEX", "jabar-balita-desa")
NUTRITION_INDEX = os.getenv("NUTRITION_INDEX", "jabar-tenaga-gizi")

# --- Konfigurasi API Lain ---
DEFAULT_INSIGHT_API = os.getenv("OPENAI_API_KEY")
DEFAULT_PREDICT_API = None
# --- Konfigurasi Aplikasi Utama ---
APP_TITLE = "StuntLytics - Dashboard Pemerintah"
APP_DESCRIPTION = "Dashboard e-Government untuk prediksi risiko stunting, monitoring, dan rekomendasi intervensi berbasis AI."

# StuntLytics/src/config.py

APP_TITLE = "StuntLytics Jawa Barat"
APP_DESCRIPTION = "Dashboard Analitik & Prediksi Stunting Berbasis Data Terintegrasi"

# --- Aturan untuk Insight Otomatis ---
# Aturan ini dijalankan pada sampel data yang diambil untuk analisis korelasi.
# Nama kolom harus sesuai dengan yang didefinisikan di elastic_client.get_numeric_sample_for_corr
# (cth: 'risk_score', 'bmi_pra_hamil', 'berat_lahir_gr', dll.)

INSIGHT_RULES = {
    "high_risk_avg": {
        "when": lambda df: "risk_score" in df.columns and df["risk_score"].mean() > 0.6,
        "msg": "Perhatian: Rata-rata **risiko stunting (probabilitas > 0.6)** di wilayah ini tergolong tinggi. Perlu investigasi mendalam terhadap faktor-faktor penyebabnya.",
    },
    "low_bmi_avg": {
        "when": lambda df: "bmi_pra_hamil" in df.columns
        and df["bmi_pra_hamil"].mean() < 18.5,
        "msg": "Insight: Rata-rata **Indeks Massa Tubuh (IMT) Ibu Pra-Hamil di bawah normal (< 18.5)**. Ini adalah faktor risiko signifikan yang perlu diintervensi melalui program gizi calon ibu.",
    },
    "high_bblr_rate": {
        # Menghitung persentase Bayi Berat Lahir Rendah (< 2500 gram)
        "when": lambda df: "berat_lahir_gr" in df.columns
        and (df["berat_lahir_gr"] < 2500).mean() > 0.15,
        "msg": "Waspada: Lebih dari **15% bayi lahir dengan berat badan rendah (BBLR)**. Perkuat pemantauan kesehatan dan gizi selama kehamilan.",
    },
    "low_anc_visits": {
        "when": lambda df: "kunjungan_anc" in df.columns
        and df["kunjungan_anc"].mean() < 4,
        "msg": "Insight: Rata-rata **kunjungan pemeriksaan kehamilan (ANC) kurang dari 4 kali**. Edukasi mengenai pentingnya ANC perlu ditingkatkan untuk deteksi dini masalah kehamilan.",
    },
    "high_anemia_risk": {
        "when": lambda df: "hb_g_dl" in df.columns and df["hb_g_dl"].mean() < 11.0,
        "msg": "Perhatian: Rata-rata **kadar Hemoglobin (Hb) Ibu tergolong rendah (< 11 g/dL)**, mengindikasikan risiko anemia yang tinggi. Program suplementasi zat besi sangat direkomendasikan.",
    },
}
