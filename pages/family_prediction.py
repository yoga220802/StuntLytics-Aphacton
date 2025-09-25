import streamlit as st
import pandas as pd
import os
import openai  # Mengganti requests dengan library resmi OpenAI
from src import prediction_service, styles, elastic_client as es


# ==============================================================================
# LOGIKA UNTUK FITUR REKOMENDASI AI (DI-UPGRADE KE OPENAI)
# ==============================================================================
def _get_openai_api_key():
    # Mencari OPENAI_API_KEY
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def generate_recommendation(
    user_data: dict, prediction_proba: float, prediction_result: str
) -> str:
    api_key = _get_openai_api_key()
    if not api_key:
        return (
            "**Rekomendasi AI tidak tersedia.**\n\n"
            "API Key untuk OpenAI (`OPENAI_API_KEY`) belum di-set."
        )

    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return f"Gagal menginisialisasi client OpenAI: {e}"

    # --- PROMPT ENGINEERING (Tetap Sama) ---
    friendly_names = {
        "tinggi_badan_ibu_cm": "Tinggi Badan Ibu (cm)",
        "lila_saat_hamil_cm": "LiLA saat Hamil (cm)",
        "bmi_pra_hamil": "BMI Pra-Hamil",
        "hb_g_dl": "Kadar Hb (g/dL)",
        "kenaikan_bb_hamil_kg": "Kenaikan BB saat Hamil (kg)",
        "usia_ibu_saat_hamil_tahun": "Usia Ibu saat Hamil (tahun)",
        "jarak_kehamilan_sebelumnya_bulan": "Jarak Kehamilan Sebelumnya (bulan)",
        "kunjungan_anc_x": "Jumlah Kunjungan ANC",
        "jumlah_anak": "Total Jumlah Anak Sebelumnya",
        "kepatuhan_ttd": "Kepatuhan Konsumsi TTD",
        "pendidikan_ibu": "Pendidikan Terakhir Ibu",
        "jenis_pekerjaan_orang_tua": "Pekerjaan Orang Tua",
        "status_pernikahan": "Status Pernikahan",
        "kepesertaan_program_bantuan": "Menerima Program Bantuan",
        "akses_air_bersih": "Akses Air Bersih Layak",
        "paparan_asap_rokok": "Paparan Asap Rokok di Rumah",
        "hipertensi_ibu": "Riwayat Hipertensi Ibu",
        "diabetes_ibu": "Riwayat Diabetes Ibu",
    }
    details = []
    for key, value in user_data.items():
        label = friendly_names.get(key, key)
        if key in ["hipertensi_ibu", "diabetes_ibu"]:
            val_str = "Ya" if value == 1 else "Tidak"
        else:
            val_str = str(value)
        details.append(f"- {label}: {val_str}")
    data_string = "\n".join(details)

    prompt = f"""
    Anda adalah seorang ahli gizi dan kesehatan anak senior dari dinas kesehatan Indonesia.
    Anda ditugaskan untuk memberikan analisis singkat, padat, dan actionable berdasarkan data individual seorang ibu hamil.

    **DATA YANG WAJIB DIANALISIS:**
    Berikut adalah data individual dari seorang ibu hamil:
    {data_string}

    **HASIL PREDIKSI SISTEM (UNTUK ANAK YANG AKAN LAHIR):**
    - Prediksi Risiko Stunting: {prediction_result}
    - Probabilitas Risiko: {prediction_proba:.2f}%

    **TUGAS ANDA:**
    Berdasarkan **HANYA PADA DATA DI ATAS**, berikan analisis dan rekomendasi Anda dalam format Markdown yang ketat sebagai berikut. Gunakan bahasa yang simpatik namun tetap profesional untuk dibaca oleh kader posyandu atau petugas lapangan.

    ### Ringkasan Analisis
    (Berikan kesimpulan 1-2 kalimat tentang status risiko kehamilan ini berdasarkan data yang paling menonjol.)

    ### Faktor Risiko Kunci
    (Identifikasi 2-3 faktor dari data yang paling signifikan meningkatkan risiko. Jelaskan secara singkat mengapa.)

    ### Kekuatan & Potensi
    (Jika ada, sebutkan faktor-faktor positif dari data yang sudah baik dan dapat menjadi modal bagi ibu.)

    ### Rekomendasi Prioritas
    (Berikan 3 poin rekomendasi yang paling penting, praktis, dan dapat segera ditindaklanjuti oleh ibu hamil ini. Gunakan poin bernomor.)
    """

    # --- PEMANGGILAN API BARU (Menggunakan OpenAI) ---
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Menggunakan model yang diminta
            messages=[
                {
                    "role": "system",
                    "content": "Anda adalah seorang ahli gizi dan kesehatan anak senior dari dinas kesehatan Indonesia.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=45,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gagal menghubungi server OpenAI. Mohon coba lagi nanti. Error: {e}"


# --- BAGIAN UTAMA APLIKASI STREAMLIT (TIDAK ADA PERUBAHAN) ---
def render_page():
    # Muat pipeline prediksi lokal
    pipeline = prediction_service.load_pipeline()

    st.subheader("Prediksi Risiko Stunting Selama Kehamilan")
    st.caption(
        "Isi form data ibu hamil untuk memprediksi risiko stunting pada anak yang akan lahir."
    )

    if not pipeline:
        st.error(
            "Gagal memuat pipeline prediksi. Mohon periksa file 'models/stunting_pipeline.joblib'."
        )
        return

    with st.form(key="prediction_form"):
        st.markdown("##### 1. Data Ibu & Kehamilan")
        col1, col2, col3 = st.columns(3)
        with col1:
            tinggi_badan_ibu_cm = st.number_input(
                "Tinggi Badan Ibu (cm)", min_value=130, max_value=200, value=155
            )
            lila_saat_hamil_cm = st.number_input(
                "LiLA saat Hamil (cm)",
                min_value=15.0,
                max_value=40.0,
                value=25.0,
                step=0.1,
            )
            bmi_pra_hamil = st.number_input(
                "BMI Pra-Hamil", min_value=10.0, max_value=40.0, value=22.0, step=0.1
            )
        with col2:
            usia_ibu_saat_hamil_tahun = st.number_input(
                "Usia Ibu saat Hamil (tahun)", min_value=15, max_value=50, value=28
            )
            kenaikan_bb_hamil_kg = st.number_input(
                "Kenaikan BB saat Hamil (kg)", min_value=0, max_value=30, value=12
            )
            jarak_kehamilan_sebelumnya_bulan = st.number_input(
                "Jarak Kehamilan Sebelumnya (bulan)",
                min_value=0,
                max_value=120,
                value=24,
            )
        with col3:
            hb_g_dl = st.number_input(
                "Kadar Hb (g/dL)", min_value=5.0, max_value=20.0, value=11.0, step=0.1
            )
            kunjungan_anc_x = st.number_input(
                "Jumlah Kunjungan ANC", min_value=0, max_value=20, value=4
            )
            kepatuhan_ttd = st.selectbox(
                "Kepatuhan Konsumsi TTD", ["Rutin", "Tidak Rutin"]
            )

        st.markdown("##### 2. Kondisi Keluarga & Lainnya")
        col1, col2, col3 = st.columns(3)
        with col1:
            pendidikan_ibu = st.selectbox(
                "Pendidikan Terakhir Ibu",
                ["SD", "SMP", "SMA", "Diploma", "S1", "S2/S3", "Tidak Sekolah"],
            )
            jenis_pekerjaan_orang_tua = st.selectbox(
                "Pekerjaan Orang Tua",
                [
                    "Buruh",
                    "Lainnya",
                    "Nelayan",
                    "PNS/TNI/Polri",
                    "Petani/Buruh Tani",
                    "TKI/TKW",
                    "Wiraswasta",
                ],
            )
            status_pernikahan = st.selectbox("Status Pernikahan", ["Menikah", "Cerai"])
        with col2:
            jumlah_anak = st.number_input(
                "Total Jumlah Anak Sebelumnya", min_value=0, max_value=15, value=1
            )
            kepesertaan_program_bantuan = st.radio(
                "Menerima Program Bantuan?", ["Ya", "Tidak"]
            )
            paparan_asap_rokok = st.radio(
                "Paparan Asap Rokok di Rumah?", ["Ya", "Tidak"]
            )
        with col3:
            akses_air_bersih = st.radio("Akses Air Bersih Layak?", ["Ya", "Tidak"])
            hipertensi_ibu = (
                1
                if st.radio("Riwayat Hipertensi Ibu?", ["Ya", "Tidak"], index=1) == "Ya"
                else 0
            )
            diabetes_ibu = (
                1
                if st.radio("Riwayat Diabetes Ibu?", ["Ya", "Tidak"], index=1) == "Ya"
                else 0
            )

        submit_button = st.form_submit_button(
            label="🔬 Prediksi & Dapatkan Rekomendasi"
        )

    if submit_button:
        input_data = {
            "tinggi_badan_ibu_cm": tinggi_badan_ibu_cm,
            "lila_saat_hamil_cm": lila_saat_hamil_cm,
            "bmi_pra_hamil": bmi_pra_hamil,
            "hb_g_dl": hb_g_dl,
            "kenaikan_bb_hamil_kg": kenaikan_bb_hamil_kg,
            "usia_ibu_saat_hamil_tahun": usia_ibu_saat_hamil_tahun,
            "jarak_kehamilan_sebelumnya_bulan": jarak_kehamilan_sebelumnya_bulan,
            "kunjungan_anc_x": kunjungan_anc_x,
            "jumlah_anak": jumlah_anak,
            "kepatuhan_ttd": kepatuhan_ttd,
            "pendidikan_ibu": pendidikan_ibu,
            "jenis_pekerjaan_orang_tua": jenis_pekerjaan_orang_tua,
            "status_pernikahan": status_pernikahan,
            "kepesertaan_program_bantuan": kepesertaan_program_bantuan,
            "akses_air_bersih": akses_air_bersih,
            "paparan_asap_rokok": paparan_asap_rokok,
            "hipertensi_ibu": hipertensi_ibu,
            "diabetes_ibu": diabetes_ibu,
        }

        prediction_result = prediction_service.run_prediction(pipeline, input_data)

        st.markdown("---")
        st.subheader("Hasil Analisis")
        if prediction_result["error"]:
            st.error(f"Gagal melakukan prediksi: {prediction_result['error']}")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label="Probabilitas Risiko Stunting",
                    value=f"{prediction_result['probability']:.2f}%",
                )
                st.write(f"Kategori: **{prediction_result['result']}**")

            with col2:
                st.subheader("💡 Rekomendasi AI")
                with st.spinner("AI sedang menganalisis dan membuat rekomendasi..."):
                    recommendation = generate_recommendation(
                        input_data,
                        prediction_result["probability"],
                        prediction_result["result"],
                    )
                    st.markdown(recommendation)


# --- Main Execution ---
if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
