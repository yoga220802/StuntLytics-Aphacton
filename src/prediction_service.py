import pandas as pd
import joblib
import streamlit as st
import os

PIPELINE_PATH = "models/stunting_pipeline.joblib"


@st.cache_resource(show_spinner="Memuat pipeline prediksi...")
def load_pipeline():
    """
    Memuat pipeline lengkap dari satu file .joblib.
    """
    try:
        if not os.path.exists(PIPELINE_PATH):
            st.error(f"File pipeline tidak ditemukan. Pastikan '{PIPELINE_PATH}' ada.")
            return None

        pipeline = joblib.load(PIPELINE_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Gagal memuat pipeline: {e}")
        return None


def run_prediction(pipeline: object, input_data: dict) -> dict:
    """
    Menjalankan prediksi menggunakan pipeline yang sudah dimuat.
    Input adalah data mentah dari form.
    """
    if not pipeline:
        return {"error": "Pipeline tidak berhasil dimuat."}

    try:
        # Konversi dictionary input menjadi DataFrame dengan satu baris
        input_df = pd.DataFrame([input_data])

        # Pipeline akan menangani semua preprocessing (scaling, encoding) secara otomatis
        prediction_proba_raw = pipeline.predict_proba(input_df)[0][1]
        prediction_result = (
            "Risiko Stunting" if prediction_proba_raw > 0.5 else "Risiko Rendah"
        )

        return {
            "probability": prediction_proba_raw * 100,
            "result": prediction_result,
            "error": None,
        }
    except Exception as e:
        return {"probability": 0, "result": "Gagal Prediksi", "error": str(e)}
