from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import requests

def post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Mengirim POST request ke API dan mengembalikan response JSON jika berhasil.
    Menampilkan warning di UI jika gagal.
    """
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"[API] {url} → status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.info(f"Tidak dapat menghubungkan API: {e}")
    return None

@st.cache_data(show_spinner=False)
def get_kabupaten_list(df: pd.DataFrame) -> List[str]:
    """Mengambil daftar unik kabupaten dari DataFrame, sudah di-cache."""
    return ["(Semua)"] + sorted(df["kabupaten"].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def get_kecamatan_list(df: pd.DataFrame, kab: Optional[str] = None) -> List[str]:
    """
    Mengambil daftar unik kecamatan. Jika kabupaten dipilih, daftar kecamatan
    akan difilter sesuai kabupaten tersebut.
    """
    d = df if not kab or kab == "(Semua)" else df[df["kabupaten"] == kab]
    return ["(Semua)"] + sorted(d["kecamatan"].dropna().unique().tolist())
