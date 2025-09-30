import streamlit as st
import pandas as pd
import plotly.express as px
import os
# Hapus OpenAI dan gunakan Google GenAI
from google import genai
from src import styles, elastic_client as es
from src.components import sidebar

# Tambahkan konfigurasi Gemini (selaras dengan halaman lain)
def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            return api_key
    except Exception:
        pass
    return None

# --- FUNGSI BARU UNTUK INSIGHT AI ---
# Hapus _get_openai_api_key dan ganti implementasi generate_ai_insight
def generate_ai_insight(
    filters: dict, trend_df: pd.DataFrame, corr_series: pd.Series
) -> str:
    """
    Menghasilkan insight dari AI berdasarkan data tren dan korelasi yang terfilter.
    """
    api_key = _configure_gemini()
    if not api_key:
        return "**Insight AI tidak tersedia.** `GEMINI_API_KEY` belum diatur."

    try:
        client = genai.Client()
    except Exception as e:
        return f"Gagal menginisialisasi Google Gemini client: {e}"

    # --- Merangkum data untuk prompt ---
    # 1. Ringkasan Tren
    trend_summary = "Data tren tidak cukup untuk dianalisis."
    if not trend_df.empty and len(trend_df) > 1:
        if not isinstance(trend_df.index, pd.DatetimeIndex):
            trend_df.index = pd.to_datetime(trend_df.index)
        start_date = trend_df.index[0].strftime("%Y-%m")
        end_date = trend_df.index[-1].strftime("%Y-%m")
        start_val = trend_df["Stunting %"].iloc[0]
        end_val = trend_df["Stunting %"].iloc[-1]
        max_val = trend_df["Stunting %"].max()
        min_val = trend_df["Stunting %"].min()
        trend_summary = (
            f"Dari {start_date} hingga {end_date}, tren prevalensi stunting "
            f"bergerak dari {start_val:.2f}% ke {end_val:.2f}%. "
            f"Puncak tertinggi adalah {max_val:.2f}% dan terendah {min_val:.2f}%."
        )

    # 2. Ringkasan Korelasi
    corr_summary = "Data korelasi tidak cukup untuk dianalisis."
    if not corr_series.empty:
        top_positive = corr_series[corr_series > 0].nlargest(3)
        top_negative = corr_series[corr_series < 0].nsmallest(3)
        pos_list = [f"{idx} ({val:.2f})" for idx, val in top_positive.items()]
        neg_list = [f"{idx} ({val:.2f})" for idx, val in top_negative.items()]
        pos_str = ", ".join(pos_list) if pos_list else "Tidak ada korelasi positif yang signifikan."
        neg_str = ", ".join(neg_list) if neg_list else "Tidak ada korelasi negatif yang signifikan."
        corr_summary = (
            f"Faktor dengan korelasi positif terkuat terhadap Z-Score (kondisi lebih baik): {pos_str}. "
            f"Faktor dengan korelasi negatif terkuat (kondisi lebih buruk): {neg_str}."
        )

    # --- Membangun Prompt ---
    prompt = f"""
    Anda adalah seorang analis data senior di dinas kesehatan, bertugas memberikan ringkasan eksekutif.
    
    Konteks Filter Data:
    {filters}

    Ringkasan Data Analisis:
    1. Analisis Tren: {trend_summary}
    2. Analisis Korelasi: {corr_summary}

    Tugas:
    Berdasarkan HANYA PADA DATA RINGKASAN DI ATAS, berikan 2-3 poin insight utama dalam format bullet points (gunakan tanda `-`).
    Fokus pada temuan paling signifikan atau actionable. Jawaban harus singkat, padat, dan langsung ke inti.
    """

    # Panggil Google GenAI (model sama dengan InsightNow)
    try:
        try:
            response = client.models.generate_content(
                model="models/gemma-3-27b-it",
                contents=prompt,
                config={
                    "temperature": 0.25,
                    "max_output_tokens": 800,
                },
            )
        except TypeError:
            # Fallback kompatibilitas SDK
            response = client.models.generate_content(
                model="models/gemma-3-27b-it",
                contents=prompt,
            )
        return response.text
    except Exception as e:
        return f"Gagal memanggil Google Gemini API: {e}"

# --- RENDER HALAMAN ---
def render_page():
    st.subheader("Tren & Korelasi – Analitik Pendukung Kebijakan")
    filters = sidebar.render()

    try:
        df_trend = es.get_monthly_trend(filters)
        df_corr_sample = es.get_numeric_sample_for_corr(filters)
    except Exception as e:
        st.error(f"Gagal mengambil data dari Elasticsearch: {e}")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tren Proporsi Stunting per Bulan**")
        if not df_trend.empty:
            st.line_chart(df_trend)
        else:
            st.warning("Data tren tidak tersedia untuk filter saat ini.")

    with c2:
        st.markdown("**Faktor Paling Berpengaruh (Korelasi thd Z-Score)**")
        target_col = "ZScore TB/U"

        corr_risk = pd.Series(dtype=float)

        if (
            not df_corr_sample.empty
            and target_col in df_corr_sample.columns
            and df_corr_sample.shape[1] > 1
        ):
            try:
                corr = df_corr_sample.corr(numeric_only=True)
                if target_col in corr:
                    corr_risk = corr[target_col].drop(target_col, errors="ignore")
                    if not corr_risk.empty:
                        top_features = corr_risk.abs().nlargest(6)
                        top_corr_values = corr_risk.loc[top_features.index]
                        df_radar = pd.DataFrame(
                            {
                                "Faktor": top_corr_values.index,
                                "Korelasi Asli": top_corr_values.values,
                                "Kekuatan Korelasi": top_corr_values.abs().values,
                            }
                        )

                        fig = px.line_polar(
                            df_radar,
                            r="Kekuatan Korelasi",
                            theta="Faktor",
                            line_close=True,
                            template="plotly_dark",
                            title="Kekuatan Pengaruh Faktor terhadap Z-Score TB/U",
                            range_r=[0, 1],
                        )
                        fig.update_traces(
                            fill="toself",
                            fillcolor="rgba(239, 68, 68, 0.3)",
                            line=dict(color="rgba(239, 68, 68, 0.8)"),
                            hovertemplate="<b>%{theta}</b><br>Kekuatan: %{r:.2f}<br>Korelasi Asli: %{customdata[0]:.2f}<extra></extra>",
                            customdata=df_radar[["Korelasi Asli"]],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Menunjukkan **kekuatan** pengaruh. Semakin rendah Z-Score, semakin tinggi risiko stunting."
                        )
                    else:
                        st.warning(
                            "Tidak ada cukup fitur lain untuk menghitung korelasi."
                        )
                else:
                    st.warning(
                        f"Kolom target '{target_col}' tidak dapat dihitung korelasinya (kemungkinan nilainya konstan)."
                    )
            except Exception as e:
                st.error(f"Gagal menghitung korelasi: {e}")
        else:
            st.warning(
                "Data tidak cukup untuk menghitung korelasi dengan filter saat ini."
            )

    # --- BAGIAN BARU: INSIGHT OTOMATIS AI ---
    st.markdown("---")
    st.subheader("🤖 Insight Otomatis AI")

    if df_trend.empty and corr_risk.empty:
        st.info("Tidak ada data yang cukup untuk dianalisis oleh AI.")
    else:
        with st.spinner("AI sedang menganalisis tren dan korelasi..."):
            ai_insight = generate_ai_insight(filters, df_trend, corr_risk)
            st.markdown(ai_insight)

# --- Main Execution ---
if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
