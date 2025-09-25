# StuntLytics/src/components/sidebar.py
# VERSI FINAL - dengan nama fungsi render() yang standar dan filter risk level
import streamlit as st
from typing import Dict, Any, List
from src import elastic_client as es

# BARU: Menambahkan kembali definisi RISK_LEVELS
RISK_LEVELS: List[str] = [
    "Zona 3 (>=0.70)",
    "Zona 2 (0.40-<0.70)",
    "Zona 1 (0.10-<0.40)",
    "Zona 0 (<0.10)",
]


def render() -> Dict[str, Any]:
    """
    Merender sidebar filter dinamis yang mengambil opsi dari Elasticsearch
    dan mengembalikan dictionary berisi pilihan filter.
    """
    st.sidebar.header("Filter Data")

    # Filter Tanggal
    date_from = st.sidebar.date_input("Tanggal dari", value=None)
    date_to = st.sidebar.date_input("Tanggal sampai", value=None)

    base_filters = {"date_from": date_from, "date_to": date_to}

    # Filter Wilayah (Kabupaten/Kota)
    wilayah_field, wilayah_opts = es.get_filter_options(
        base_filters, es.CANDIDATES_WILAYAH
    )
    selected_wilayah = st.sidebar.multiselect("Kabupaten/Kota", options=wilayah_opts)

    # Filter Kecamatan (berdasarkan pilihan wilayah)
    kecamatan_field, kecamatan_opts = None, []
    if selected_wilayah:
        # Buat filter sementara untuk mengambil opsi kecamatan yang relevan
        tmp_filters_for_kec = dict(base_filters)
        tmp_filters_for_kec.update(
            {"wilayah_field": wilayah_field, "wilayah": selected_wilayah}
        )
        kecamatan_field, kecamatan_opts = es.get_filter_options(
            tmp_filters_for_kec, es.CANDIDATES_KECAMATAN, size=3000
        )

    selected_kecamatan = st.sidebar.multiselect("Kecamatan", options=kecamatan_opts)

    # BARU: Menambahkan kembali filter Level Risiko
    selected_risk_level = st.sidebar.multiselect("Level Risiko", options=RISK_LEVELS)

    return {
        "date_from": date_from,
        "date_to": date_to,
        "wilayah": selected_wilayah,
        "kecamatan": selected_kecamatan,
        "risk_level": selected_risk_level,  # BARU: Mengembalikan pilihan risk_level
        "wilayah_field": wilayah_field,
        "kecamatan_field": kecamatan_field,
    }
