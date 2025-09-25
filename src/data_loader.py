import pandas as pd
import streamlit as st
from . import elastic_client, config
import numpy as np


# --- (Salin dari implementasi sebelumnya, karena ini masih relevan) ---
def create_dummy_data() -> pd.DataFrame:
    # Fungsi ini tidak diubah, tetap sama seperti sebelumnya
    st.warning(
        "Koneksi ke database gagal atau tidak ada data. Menggunakan dummy data untuk demo."
    )
    np.random.seed(42)
    kabupaten = [
        "Kab. Garut",
        "Kab. Bandung",
        "Kota Bandung",
        "Kab. Tasikmalaya",
        "Kab. Cirebon",
        "Kab. Bogor",
        "Kota Bogor",
        "Kab. Sukabumi",
        "Kota Depok",
        "Kab. Sumedang",
    ]
    kecamatan = [
        f"Kec. {x}"
        for x in [
            "Tarogong Kidul",
            "Cibatu",
            "Leles",
            "Cicalengka",
            "Rancaekek",
            "Sukajadi",
            "Lengkong",
            "Cisayong",
            "Sumber",
            "Cibinong",
            "Cimanggung",
        ]
    ]
    n = 2000
    df = pd.DataFrame(
        {
            "kabupaten": np.random.choice(kabupaten, n),
            "kecamatan": np.random.choice(kecamatan, n),
            "desa": [f"Desa-{i % 50:02d}" for i in range(n)],
            "tanggal": pd.to_datetime(
                np.random.choice(pd.date_range("2024-01-01", "2025-08-01"), n)
            ),
            "usia_anak_bulan": np.random.randint(0, 60, n),
            "jenis_kelamin": np.random.choice(["L", "P"], n),
            "bblr": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "asi_eksklusif": np.random.choice([0, 1], n, p=[0.4, 0.6]),
            "mp_asi_tepatsesuai": np.random.choice([0, 1], n, p=[0.5, 0.5]),
            "imunisasi_lengkap": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "pendidikan_ibu": np.random.choice(
                ["SD", "SMP", "SMA", "D3", "S1+"], n, p=[0.25, 0.3, 0.3, 0.1, 0.05]
            ),
            "akses_air_layak": np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "jamban_sehat": np.random.choice([0, 1], n, p=[0.25, 0.75]),
            "pengeluaran_bulan": np.random.normal(3200000, 1000000, n)
            .clip(500000, 10000000)
            .round(),
            "tanggungan": np.random.randint(1, 7, n),
        }
    )
    df["total_bayi_lahir"] = np.random.randint(1, 10, n)
    df["total_bayi_stunting"] = np.random.randint(0, df["total_bayi_lahir"] + 1)
    df["tahun"] = df["tanggal"].dt.year
    df["jumlah_nakes"] = df["kabupaten"].map(
        lambda x: np.random.randint(50, 500) if x else 0
    ) + np.random.choice(range(10, 50), len(df))
    score = (
        0.25 * df["bblr"]
        + 0.15 * (1 - df["asi_eksklusif"])
        + 0.12 * (1 - df["imunisasi_lengkap"])
        + 0.18 * (1 - df["akses_air_layak"])
        + 0.12 * (df["jamban_sehat"])
        + 0.08 * (df["usia_anak_bulan"] / 60.0)
        + 0.10 * (df["tanggungan"] / 7.0)
    )
    df["risk_score"] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    bins = [-0.01, 0.33, 0.66, 1.0]
    df["risk_label"] = pd.cut(
        df["risk_score"], bins=bins, labels=["Rendah", "Sedang", "Tinggi"]
    ).astype(str)
    return df


def _normalize_location(series: pd.Series) -> pd.Series:
    """Mengubah kolom lokasi menjadi format standar (UPPERCASE, STRIPPED)."""
    return series.astype(str).str.upper().str.strip()


# --- (Logika pemrosesan dan merge data sekarang lebih robust) ---
def process_and_merge_data(df_stunting, df_balita, df_nakes):
    st.info("Memulai mode debug: Cek terminal/console Anda untuk output detail.")

    if not df_balita.empty:
        df_balita["jumlah_balita"] = pd.to_numeric(
            df_balita["jumlah_balita"], errors="coerce"
        ).fillna(0)
    if not df_nakes.empty:
        df_nakes["jumlah_nakes_gizi"] = pd.to_numeric(
            df_nakes["jumlah_nakes_gizi"], errors="coerce"
        ).fillna(0)

    if df_stunting.empty:
        return pd.DataFrame()

    # --- TAHAP 1: Proses df_stunting (Data Utama) ---
    stunting_mapping = {
        "nama_kabupaten_kota": "kabupaten",
        "Kecamatan": "kecamatan",
        "Tanggal": "tanggal",
        "Usia Anak (bulan)": "usia_anak_bulan",
        "ASI Eksklusif (ya/tidak)": "asi_eksklusif",
        "Imunisasi (lengkap/tidak lengkap)": "imunisasi_lengkap",
        "Akses Air Bersih": "akses_air_layak",
        "Upah Keluarga (Rp/bulan)": "pengeluaran_bulan",
        "Jumlah Anak": "tanggungan",
        "Pendidikan Ibu": "pendidikan_ibu",
        "Berat Lahir (gram)": "berat_lahir_gram",
        # !! INI DIA BIANG KEROKNYA: Kolom 'Status Stunting (Biner)' tidak ada, kita hapus dari mapping !!
        # 'Status Stunting (Biner)': 'is_stunting'
    }
    df = df_stunting.rename(columns=stunting_mapping)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    # !! INI DIA SOLUSINYA: Buat kolom is_stunting dari kolom yang ADA !!
    if "Status Stunting (Stunting / Berisiko / Normal)" in df.columns:
        df["is_stunting"] = (
            df["Status Stunting (Stunting / Berisiko / Normal)"].astype(str)
            == "Stunting"
        ).astype(int)

    if "kabupaten" in df.columns:
        df["kabupaten"] = _normalize_location(df["kabupaten"])
    if "kecamatan" in df.columns:
        df["kecamatan"] = _normalize_location(df["kecamatan"])

    binary_cols = {
        "asi_eksklusif": "Ya",
        "imunisasi_lengkap": "Lengkap",
        "akses_air_layak": "Layak",
    }
    for col, pos_val in binary_cols.items():
        if col in df.columns:
            df[col] = (df[col].astype(str).str.lower() == pos_val.lower()).astype(int)

    if "berat_lahir_gram" in df.columns:
        df["berat_lahir_gram"] = pd.to_numeric(df["berat_lahir_gram"], errors="coerce")
        df["bblr"] = (df["berat_lahir_gram"] < 2500).astype(int)

    # --- TAHAP 2: Proses dan Agregasi Data Pendukung ---
    if not df_balita.empty:
        df_balita = df_balita.rename(
            columns={
                "bps_nama_kabupaten_kota": "kabupaten",
                "bps_nama_kecamatan": "kecamatan",
            }
        )
        df_balita["kabupaten"] = _normalize_location(df_balita["kabupaten"])
        df_balita["kecamatan"] = _normalize_location(df_balita["kecamatan"])
        balita_agg = (
            df_balita.groupby(["kabupaten", "kecamatan"])["jumlah_balita"]
            .sum()
            .reset_index()
        )
        balita_agg = balita_agg.rename(columns={"jumlah_balita": "total_bayi_lahir"})
        df = pd.merge(df, balita_agg, on=["kabupaten", "kecamatan"], how="left")

    if not df_nakes.empty:
        df_nakes = df_nakes.rename(
            columns={
                "nama_kabupaten_kota": "kabupaten",
                "jumlah_nakes_gizi": "jumlah_nakes",
            }
        )
        df_nakes["kabupaten"] = _normalize_location(df_nakes["kabupaten"])
        nakes_agg = df_nakes.groupby("kabupaten")["jumlah_nakes"].sum().reset_index()
        df = pd.merge(df, nakes_agg, on="kabupaten", how="left")

    # --- TAHAP 3: Finalisasi & Pembersihan ---
    if "is_stunting" in df.columns:
        stunting_count = (
            df.groupby(["kabupaten", "kecamatan"])["is_stunting"].sum().reset_index()
        )
        stunting_count = stunting_count.rename(
            columns={"is_stunting": "total_bayi_stunting"}
        )
        df = pd.merge(df, stunting_count, on=["kabupaten", "kecamatan"], how="left")

    required_cols = [
        "bblr",
        "asi_eksklusif",
        "imunisasi_lengkap",
        "akses_air_layak",
        "jamban_sehat",
        "usia_anak_bulan",
        "tanggungan",
        "total_bayi_lahir",
        "total_bayi_stunting",
        "jumlah_nakes",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- TAHAP 4: Kalkulasi Risk Score ---
    score = (
        0.25 * df["bblr"]
        + 0.15 * (1 - df["asi_eksklusif"])
        + 0.12 * (1 - df["imunisasi_lengkap"])
        + 0.18 * (1 - df["akses_air_layak"])
        + 0.12 * df["jamban_sehat"]
        + 0.08 * (df["usia_anak_bulan"] / 60.0)
        + 0.10 * (df["tanggungan"] / 7.0)
    )
    df["risk_score"] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    bins = [-0.01, 0.33, 0.66, 1.0]
    df["risk_label"] = pd.cut(
        df["risk_score"], bins=bins, labels=["Rendah", "Sedang", "Tinggi"]
    ).astype(str)

    return df


# !! PENTING: JIKA SUDAH BERHASIL, AKTIFKAN LAGI CACHE DI BAWAH INI !!
@st.cache_data(show_spinner="Memuat data dari database...")
def load_data() -> pd.DataFrame:
    """Fungsi utama untuk memuat dan memproses data dari Elasticsearch."""
    ok, msg = elastic_client.ping()
    if not ok:
        st.error(msg)
        return create_dummy_data()

    df_stunting = elastic_client.get_all_data(config.STUNTING_INDEX)
    df_balita = elastic_client.get_all_data(config.BALITA_INDEX)
    df_nakes = elastic_client.get_all_data(config.NUTRITION_INDEX)

    if df_stunting.empty:
        st.error(
            f"Data utama di index '{config.STUNTING_INDEX}' kosong. Tidak bisa melanjutkan."
        )
        return create_dummy_data()

    # Nonaktifkan mode debug jika sudah yakin
    # df_processed = process_and_merge_data_prod(df_stunting, df_balita, df_nakes)
    df_processed = process_and_merge_data(df_stunting, df_balita, df_nakes)

    if df_processed.empty:
        st.error("Gagal memproses data setelah diambil dari database.")
        return create_dummy_data()

    st.success("Data berhasil dimuat dan diproses dari Elasticsearch.")
    return df_processed
