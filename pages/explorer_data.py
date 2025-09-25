import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import openai
import json

from src import styles
from src import elastic_client as es
from src.components import sidebar


# --- FUNGSI BARU UNTUK INSIGHT AI ---
def _get_openai_api_key():
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def generate_ai_summary(
    main_filters: dict, advanced_filters: dict, df: pd.DataFrame
) -> str:
    """
    Menghasilkan ringkasan cerdas dari AI berdasarkan data yang ditampilkan di explorer.
    """
    api_key = _get_openai_api_key()
    if not api_key:
        return "**Ringkasan AI tidak tersedia.** `OPENAI_API_KEY` belum diatur."

    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return f"Gagal menginisialisasi client OpenAI: {e}"

    # --- Merangkum DataFrame menjadi statistik untuk prompt ---
    if df.empty:
        return "Tidak ada data untuk dianalisis."

    summary = {
        "Jumlah Data Terfilter": len(df),
        "Statistik Z-Score": df["Z-Score"].describe().to_dict(),
        "Distribusi Pendidikan Ibu (%)": (
            df["Pendidikan Ibu"].value_counts(normalize=True) * 100
        )
        .round(2)
        .to_dict(),
        "Distribusi ASI Eksklusif (%)": (
            df["ASI Eksklusif"].value_counts(normalize=True) * 100
        )
        .round(2)
        .to_dict(),
        "Distribusi Akses Air Bersih (%)": (
            df["Akses Air Bersih"].value_counts(normalize=True) * 100
        )
        .round(2)
        .to_dict(),
        "Rata-rata Usia Anak (bulan)": round(df["Usia Anak (bulan)"].mean(), 1),
        "Rata-rata BMI Pra-Hamil": round(df["BMI Pra-Hamil"].mean(), 2)
        if "BMI Pra-Hamil" in df.columns
        else "N/A",
    }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)

    # --- Membangun Prompt ---
    prompt = f"""
    Anda adalah seorang analis data kesehatan masyarakat yang sangat teliti. Tugas Anda adalah memberikan analisis singkat dan tajam terhadap sub-kelompok data stunting yang sudah difilter.

    **Konteks Filter Aktif:**
    - Filter Utama: {main_filters}
    - Filter Lanjutan: {advanced_filters}

    **Ringkasan Statistik dari Data Terfilter:**
    ```json
    {summary_json}
    ```

    **Tugas Anda:**
    Berdasarkan **HANYA PADA RINGKASAN STATISTIK DI ATAS**, berikan 2-3 poin analisis utama dalam format bullet points (`-`). Fokus pada karakteristik yang paling menonjol dari kelompok ini. Apa yang bisa disimpulkan tentang profil risiko mereka? Jawaban harus singkat, padat, dan berbasis data.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "Anda adalah analis data kesehatan masyarakat yang ahli menganalisis sub-kelompok data.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=90,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gagal menghubungi server OpenAI: {e}"


# --- RENDER HALAMAN ---
def render_page():
    # --- Sidebar & Filter Utama ---
    st.subheader("Explorer Data – Filter, Visualisasi & Ekspor")
    main_filters = sidebar.render()

    # --- Filter Lanjutan (khusus halaman ini) ---
    st.markdown("##### Filter Lanjutan")
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            edu_opts = es.get_unique_field_values(main_filters, "Pendidikan Ibu")
            if not edu_opts:
                edu_opts = ["SD", "SMP", "SMA", "D3", "S1+"]
            edu = st.multiselect("Pendidikan Ibu", edu_opts, [])
        except Exception:
            edu = st.multiselect(
                "Pendidikan Ibu", ["SD", "SMP", "SMA", "D3", "S1+"], []
            )  # Fallback

    with c2:
        asi = st.select_slider(
            "ASI Eksklusif", options=["Semua", "Ya", "Tidak"], value="Semua"
        )
    with c3:
        air = st.select_slider(
            "Akses Air Layak", options=["Semua", "Ada", "Tidak"], value="Semua"
        )

    advanced_filters = {"pendidikan_ibu": edu, "asi_eksklusif": asi, "akses_air": air}

    # --- Pengambilan Data & Tampilan Tabel ---
    try:
        df_explorer = es.get_explorer_data(main_filters, advanced_filters, size=1000)

        st.caption(
            "Menampilkan hingga 1.000 data teratas yang paling berisiko. Gunakan fitur ekspor di bawah untuk mengunduh data lebih lengkap."
        )
        if not df_explorer.empty:
            df_display = df_explorer.copy()
            df_display["id_baris"] = range(len(df_display))
            st.dataframe(
                df_display.drop(columns=["id_baris"]),
                use_container_width=True,
                height=420,
            )

            # --- Chart Berjenjang ---
            st.markdown("---")
            # (Logika chart berjenjang dari sebelumnya, tidak berubah)
            if main_filters.get("kecamatan"):
                st.markdown(
                    "##### Top 5 Keluarga Paling Berisiko (Berdasarkan Z-Score)"
                )
                df_chart_data = df_display.nsmallest(5, "Z-Score").copy()
                df_chart_data["Identifier"] = (
                    "ID Baris: "
                    + df_chart_data["id_baris"].astype(str)
                    + " (Z-Score: "
                    + df_chart_data["Z-Score"].round(2).astype(str)
                    + ")"
                )
                fig = px.bar(
                    df_chart_data.sort_values("Z-Score", ascending=False),
                    x="Z-Score",
                    y="Identifier",
                    orientation="h",
                    title=f"5 Kasus Z-Score Terendah di Kec. {main_filters['kecamatan'][0]}",
                    labels={"Z-Score": "Z-Score", "Identifier": "Data Individual"},
                    text="Z-Score",
                )
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                    marker_color="#ef4444",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
            else:
                df_agg = es.get_top_counts_for_explorer_chart(
                    main_filters, advanced_filters
                )
                if not df_agg.empty:
                    y_col = df_agg.columns[0]
                    title = f"Top 5 {y_col} (Jumlah Data)"
                    if main_filters.get("wilayah"):
                        title = f"Top 5 Kecamatan di {main_filters['wilayah'][0]} (Jumlah Data)"
                    st.markdown(f"##### {title}")
                    fig = px.bar(
                        df_agg.sort_values("Jumlah Data", ascending=True),
                        x="Jumlah Data",
                        y=y_col,
                        orientation="h",
                        title=title,
                        labels={"Jumlah Data": "Jumlah Data", y_col: y_col},
                        text="Jumlah Data",
                    )
                    fig.update_traces(
                        texttemplate="%{text}",
                        textposition="outside",
                        marker_color="#3b82f6",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total descending"})
                else:
                    fig = None
            if "fig" in locals() and fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # --- ZONA EKSPOR BARU ---
            st.markdown("---")
            with st.expander("📥 Buka Panel Ekspor Data"):
                st.markdown(
                    "Unduh data yang Anda lihat (sesuai filter di atas) dalam format CSV atau JSON."
                )

                # Menggabungkan semua filter untuk fungsi ekspor
                all_filters = {**main_filters, **advanced_filters}

                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    if st.button("Siapkan File CSV (hingga 5.000 baris)"):
                        with st.spinner("Mempersiapkan file CSV..."):
                            df_export = es.get_explorer_data_for_export(
                                main_filters, advanced_filters, size=5000
                            )
                            st.session_state.export_df = (
                                df_export  # Simpan di session state
                            )
                            st.session_state.csv_ready = True
                            st.success("File CSV siap!")

                    if st.session_state.get("csv_ready"):
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_bytes = st.session_state.export_df.to_csv(
                            index=False
                        ).encode("utf-8")
                        st.download_button(
                            "⬇️ Unduh CSV",
                            csv_bytes,
                            f"stuntlytics_export_{now_str}.csv",
                            "text/csv",
                        )

                with export_col2:
                    if st.button("Siapkan File JSON (hingga 5.000 baris)"):
                        with st.spinner("Mempersiapkan file JSON..."):
                            # Kita bisa pakai data yang sama jika sudah di-fetch untuk CSV
                            if "export_df" not in st.session_state:
                                df_export = es.get_explorer_data_for_export(
                                    main_filters, advanced_filters, size=5000
                                )
                                st.session_state.export_df = df_export
                            st.session_state.json_ready = True
                            st.success("File JSON siap!")

                    if st.session_state.get("json_ready"):
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        json_string = st.session_state.export_df.to_json(
                            orient="records", indent=4, force_ascii=False
                        )
                        st.download_button(
                            "⬇️ Unduh JSON",
                            json_string,
                            f"stuntlytics_export_{now_str}.json",
                            "application/json",
                        )

            # --- BAGIAN BARU: INSIGHT AI ---
            st.markdown("---")
            st.subheader("🤖 Ringkasan Cerdas AI")
            with st.spinner("AI sedang menganalisis data yang ditampilkan..."):
                ai_summary = generate_ai_summary(
                    main_filters, advanced_filters, df_explorer
                )
                st.markdown(ai_summary)

        else:
            st.info("Tidak ada data yang cocok dengan kriteria filter yang dipilih.")

    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        st.exception(e)


# --- Main Execution ---
if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
