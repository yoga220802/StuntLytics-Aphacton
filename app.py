# StuntLytics/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# BARU: Ganti import data_loader dengan elastic_client
from src import config, styles, elastic_client as es
from src.components.sidebar import render  # Ganti dengan sidebar dinamis


def main():
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    styles.load_css()

    # --- BARU: Pengecekan Koneksi (tanpa menampilkan status di sidebar) ---
    ok, _ = es.ping()
    if not ok:
        st.error("Tidak dapat terhubung ke server data. Aplikasi tidak dapat berjalan.")
        st.stop()

    # GANTI: sidebar.render_sidebar(df_all) menjadi render()
    # Tidak ada lagi df_all atau df_filtered, semua kalkulasi dilakukan di ES
    filters = render()
    st.session_state["filters"] = filters

    # --- BARU: Pengambilan Data Terpusat dari Elasticsearch ---
    try:
        with st.spinner("Mengambil dan memproses data dari Elasticsearch..."):
            summary_data = es.get_main_page_summary(filters)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        st.stop()

    # Header (TETAP SAMA)
    st.markdown(
        f'<div class="app-header">{config.APP_TITLE}</div>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="app-subtitle">{config.APP_DESCRIPTION}</div>',
        unsafe_allow_html=True,
    )

    # --- GANTI: Sumber data KPI menggunakan hasil dari ES ---
    kpi_data = summary_data["kpi"]
    chart_data = summary_data["charts"]

    total_bayi_lahir = kpi_data["total_bayi_lahir"]
    total_bayi_stunting = kpi_data["total_bayi_stunting"]
    total_nakes = kpi_data["jumlah_nakes"]
    imun_cov = kpi_data["cakupan_imunisasi_pct"]
    air_cov = kpi_data["akses_air_layak_pct"]

    col1, col2, col3, col4 = st.columns(4)

    # --- Kolom 1: Stunting (Tampilan TETAP SAMA, sumber data GANTI) ---
    with col1:
        st.markdown(
            """<div class="metric-card"><div class="metric-card-title">Total Stunting / Lahir</div>""",
            unsafe_allow_html=True,
        )
        st.metric(
            "Total Bayi Stunting / Total Bayi Lahir",
            f"{total_bayi_stunting:,} / {total_bayi_lahir:,}",
            label_visibility="hidden",
        )
        st.markdown(
            '<div class="small-muted">per wilayah setelah filter</div></div>',
            unsafe_allow_html=True,
        )
        stunting_data = {
            "Bayi Stunting": total_bayi_stunting,
            "Bayi Tidak Stunting": total_bayi_lahir - total_bayi_stunting,
        }
        fig_stunting = go.Figure(
            data=[
                go.Pie(
                    labels=list(stunting_data.keys()),
                    values=list(stunting_data.values()),
                    hole=0.5,
                    textinfo="percent",
                )
            ]
        )
        fig_stunting.update_layout(
            title="Proporsi Bayi Stunting",
            showlegend=False,
            margin=dict(t=30, b=10, l=10, r=10),
            height=150,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_stunting, use_container_width=True, config={"displayModeBar": False}
        )

    # --- Kolom 2: Nakes (Tampilan TETAP SAMA, sumber data GANTI) ---
    with col2:
        st.markdown(
            """<div class="metric-card"><div class="metric-card-title">Jumlah Nakes</div>""",
            unsafe_allow_html=True,
        )
        st.metric("Jumlah Nakes", f"{total_nakes:,.0f}", label_visibility="hidden")
        st.markdown(
            '<div class="small-muted">per wilayah setelah filter</div></div>',
            unsafe_allow_html=True,
        )
        nakes_grouped = chart_data["nakes_by_region"]

        # Logika judul dinamis TETAP SAMA
        if not filters["wilayah"]:
            title = "Jumlah Nakes per Kabupaten"
            yaxis_title = "Kabupaten"
        elif not filters["kecamatan"]:
            title = f"Jumlah Nakes per Kecamatan di {filters['wilayah'][0]}"
            yaxis_title = "Kecamatan"
        else:
            title = f"Jumlah Nakes di Kecamatan {filters['kecamatan'][0]}"
            yaxis_title = "Kecamatan"

        fig_nakes = go.Figure(
            data=[
                go.Bar(
                    x=nakes_grouped.values,
                    y=nakes_grouped.index,
                    orientation="h",
                    marker=dict(color="blue"),
                )
            ]
        )
        fig_nakes.update_layout(
            title=title,
            xaxis_title="Jumlah Nakes",
            yaxis_title=yaxis_title,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=30, b=10, l=10, r=10),
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_nakes, use_container_width=True, config={"displayModeBar": False}
        )

    # --- Kolom 3: Imunisasi (Tampilan TETAP SAMA, sumber data GANTI) ---
    with col3:
        st.markdown(
            """<div class="metric-card"><div class="metric-card-title">Cakupan Imunisasi</div>""",
            unsafe_allow_html=True,
        )
        st.metric("Cakupan Imunisasi", f"{imun_cov:.1f}%", label_visibility="hidden")
        st.markdown(
            '<div class="small-muted">indikator kunci SSGI</div></div>',
            unsafe_allow_html=True,
        )
        imunisasi_per_bulan = chart_data["imunisasi_trend"]
        fig_imun = go.Figure(
            data=[
                go.Scatter(
                    x=imunisasi_per_bulan["tanggal"],
                    y=imunisasi_per_bulan["imunisasi_lengkap"] * 100,
                    mode="lines+markers",
                    line=dict(color="green"),
                )
            ]
        )
        fig_imun.update_layout(
            title="Cakupan Imunisasi per Bulan",
            xaxis_title="Bulan",
            yaxis_title="Cakupan (%)",
            margin=dict(t=30, b=10, l=10, r=10),
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_imun, use_container_width=True, config={"displayModeBar": False}
        )

    # --- Kolom 4: Akses Air (Tampilan TETAP SAMA, sumber data GANTI) ---
    with col4:
        st.markdown(
            """<div class="metric-card"><div class="metric-card-title">Akses Air Layak</div>""",
            unsafe_allow_html=True,
        )
        st.metric("Akses Air Layak", f"{air_cov:.1f}%", label_visibility="hidden")
        st.markdown(
            '<div class="small-muted">sektor WASH</div></div>', unsafe_allow_html=True
        )
        air_layak_data = chart_data["air_distribusi"]
        fig_air = go.Figure(
            data=[
                go.Pie(
                    labels=air_layak_data.index,
                    values=air_layak_data.values,
                    hole=0.4,
                    textinfo="percent",
                )
            ]
        )
        fig_air.update_layout(
            title="Proporsi Akses Air Layak",
            showlegend=False,
            margin=dict(t=30, b=10, l=10, r=10),
            height=150,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_air, use_container_width=True, config={"displayModeBar": False}
        )

    # Footer Info (TETAP SAMA)
    st.info(
        "Selamat datang di Dashboard StuntLytics. Gunakan navigasi di sebelah kiri untuk menjelajahi fitur-fitur analisis.",
        icon="👋",
    )


if __name__ == "__main__":
    main()
