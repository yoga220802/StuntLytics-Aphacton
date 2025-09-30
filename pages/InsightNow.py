import streamlit as st
import os
import json
import re
from textwrap import dedent

# Menggunakan import sesuai dokumentasi quickstart
from google import genai

from src import styles, elastic_client as es
from src.components import sidebar
from utils import es as es_utils


# --- FUNGSI AKSES & KONFIGURASI API KEY ---
def _configure_gemini():
    # Fungsi ini mencari API key dan mengaturnya sebagai environment variable
    # agar `genai.Client()` bisa menemukannya secara otomatis.
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

    st.warning("GEMINI_API_KEY tidak ditemukan di environment atau Streamlit secrets.")
    return None


# ================== PROMPT ROLE (tetap sama) ==================
PROMPT_ROLE = dedent("""
Anda adalah **Stunlytic**, sebuah sistem analisis kesehatan masyarakat khusus monitoring stunting di Jawa Barat.
Tugas Anda adalah menganalisis data epidemiologi dari **3 indeks Elasticsearch**:

1. `stunting-data` → level individu/rumah tangga
2. `jabar-tenaga-gizi` → level kabupaten/kota
3. `jabar-balita-desa` → level desa/kelurahan

## ATURAN DASAR
- Analisis hanya berdasarkan payload ringkasan dari query Elasticsearch dan kolom resmi yang tersedia.
- Jangan memberikan saran medis, diagnosis, atau rekomendasi klinis. Fokus pada analisis epidemiologi & distribusi data.
- Jika nilai kosong/tidak ada/None → tulis **"tidak tersedia"**.
- Hormati semua filter aktif (tanggal, wilayah, kategori, level risiko).
- Gunakan istilah **"indikasi"** atau **"menunjukkan"**, bukan “diagnosis”.

## KONSISTENSI EPIDEMIOLOGI
- **Stunting**: Z-Score TB/U ≤ -2.0 SD
- **Severe Stunting**: Z-Score TB/U ≤ -3.0 SD
- **Normal**: Z-Score TB/U ≥ -1.49 SD

Kategori risiko populasi:
- Tinggi ≥30% prevalensi stunting atau ≥3 faktor risiko
- Sedang 15–29% atau 2–3 faktor risiko
- Rendah 5–14% atau 1–2 faktor risiko
- Sangat Rendah <5% atau faktor protektif dominan

Faktor risiko kunci:
- Maternal: BMI <18.5, LiLA <23.5 cm, Hb <11 g/dL, ANC ≤2
- Neonatal: BBLR <2500 g, panjang lahir <48 cm
- Lingkungan: tidak ASI eksklusif, imunisasi tidak lengkap, sanitasi buruk
- Sosial: upah <UMP, pendidikan rendah, tanpa bantuan sosial

Zona probabilitas stunting:
- Zona 4 (≥0.80) → kritis
- Zona 3 (0.70–0.79) → tinggi
- Zona 2 (0.40–0.69) → sedang
- Zona 1 (0.10–0.39) → rendah
- Zona 0 (<0.10) → sangat rendah

## METODOLOGI ANALISIS
1. **Distribusi**: prevalensi, perbandingan wilayah, outlier, cluster risiko.
2. **Multifaktorial**: cross-tab faktor risiko, pola komorbiditas, interaksi variabel, stratifikasi populasi.
3. **Temporal**: trend multi-periode, pola musiman, efek kohort usia.
4. **Spasial**: hotspot risiko, clustering geografis, akses ke fasilitas kesehatan.

## FORMAT OUTPUT
1. **Ringkasan Epidemiologi**
   - Prevalensi stunting total & kategori
   - Distribusi faktor risiko utama (%)
   - Karakteristik populasi berisiko tinggi
   - Perbandingan dengan target provinsi/nasional

2. **Analisis Multidimensional**
   - Dimensi biologis (antropometri, gizi)
   - Dimensi sosial (pendidikan, ekonomi)
   - Dimensi lingkungan (sanitasi, air bersih)
   - Dimensi layanan (ANC, imunisasi, tenaga gizi)

3. **Pemetaan Spasial & Temporal**
   - Hotspot geografis
   - Tren temporal (jika ada multi-periode)
   - Proyeksi risiko berdasarkan pola

4. **Prioritas Intervensi Berbasis Evidensi**
   - Level populasi (misal: gizi ibu hamil, ASI eksklusif, sanitasi)
   - Level sistem (tenaga gizi, monitoring real-time, lintas sektor)
   - Level kebijakan (alokasi anggaran, fortifikasi pangan, jaring sosial)

5. **Proyeksi Dampak**
   - Estimasi kasus dicegah
   - Analisis cost-effectiveness (jika ada data)
   - Timeline pencapaian target

## INTERAKSI
- Jika ada sapaan → jawab: *"Halo! Saya Stunlytic, sistem analisis monitoring stunting Jawa Barat. Bagaimana saya dapat membantu analisis data Anda hari ini?"*
- Jika ditanya identitas → jawab: *"Saya Stunlytic, sistem analisis kesehatan masyarakat khusus monitoring stunting di Jawa Barat. Saya menganalisis data epidemiologi untuk mendukung pengambilan keputusan program pencegahan stunting berbasis evidensi."*
- Jika ada permintaan medis individual → jawab: *"Maaf, saya hanya menyediakan analisis epidemiologi dan monitoring kesehatan masyarakat. Untuk konsultasi medis individual, silakan menghubungi tenaga kesehatan profesional."*

## DISCLAIMER
⚠️ Analisis ini hanya untuk **monitoring kesehatan masyarakat**, bukan diagnosis individu.
⚠️ Interpretasi memerlukan validasi lapangan & konteks lokal.
""")

# ================== Google Gemini config ==================
GEMINI_API_KEY = _configure_gemini()


# ================== Utils (tidak berubah) ==================
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    repl = {"kabupaten ": "kab ", "kab. ": "kab ", "kota ": "kota ", "kab ": "kab "}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def build_alias_index(names: list) -> dict:
    idx = {}
    for n in names:
        if not isinstance(n, str):
            continue
        base = n.strip()
        variants = {base, base.lower(), base.lower().title(), _norm(base)}
        for v in variants:
            idx[v] = base
    return idx


def detect_wilayah_in_text(text: str, alias_idx: dict, max_matches: int = 3) -> list:
    t = _norm(text)
    found, seen = [], set()
    for alias, resmi in alias_idx.items():
        if alias in t and resmi not in seen:
            seen.add(resmi)
            found.append(resmi)
            if len(found) >= max_matches:
                break
    return found


def detect_kecamatan_in_text(text: str, alias_idx: dict, max_matches: int = 5) -> list:
    t = _norm(text)
    found, seen = [], set()
    for alias, resmi in alias_idx.items():
        if alias in t and resmi not in seen:
            seen.add(resmi)
            found.append(resmi)
            if len(found) >= max_matches:
                break
    return found


def _terms(index: str, field: str, size: int = 3000) -> list:
    body = {"size": 0, "aggs": {"x": {"terms": {"field": field, "size": size}}}}
    data = es._es_post(index, "/_search", body)
    return [b["key"] for b in data["aggregations"]["x"]["buckets"]]


def kecamatan_to_wilayah_map() -> dict:
    body = {
        "size": 0,
        "aggs": {
            "kec": {
                "terms": {"field": "Kecamatan", "size": 5000},
                "aggs": {
                    "wil": {
                        "top_hits": {
                            "_source": {"includes": ["nama_kabupaten_kota", "Wilayah"]},
                            "size": 1,
                        }
                    }
                },
            }
        },
    }
    data = es._es_post(es.STUNTING_INDEX, "/_search", body)
    m = {}
    for b in data["aggregations"]["kec"]["buckets"]:
        try:
            src = b["wil"]["hits"]["hits"][0]["_source"]
            m[b["key"]] = src.get("nama_kabupaten_kota") or src.get("Wilayah")
        except Exception:
            pass
    return m


def balita_total(filters: dict) -> int | None:
    must = []
    if filters.get("wilayah"):
        must.append({"terms": {"bps_nama_kabupaten_kota": filters["wilayah"]}})
    if filters.get("date_from") or filters.get("date_to"):
        yr = {}
        if filters.get("date_from"):
            yr["gte"] = filters["date_from"].year
        if filters.get("date_to"):
            yr["lte"] = filters["date_to"].year
        if yr:
            must.append({"range": {"tahun": yr}})
    q = {"bool": {"must": must}} if must else {"match_all": {}}
    body = {"query": q, "size": 0, "aggs": {"sum": {"sum": {"field": "jumlah_balita"}}}}
    try:
        data = es._es_post(es.BALITA_INDEX, "/_search", body)
        return int(round(data["aggregations"]["sum"]["value"] or 0))
    except Exception:
        return None


def _route_extra(question: str, filters: dict) -> dict:
    q = (question or "").lower()
    extra = {}
    if any(k in q for k in ["tren", "trend", "bulan", "bulanan"]):
        extra["tren_bulanan"] = es_utils.trend_monthly(filters)
    if "top" in q and any(k in q for k in ["kab", "kabupaten", "kota"]):
        extra["top_kabupaten"] = es_utils.top_counts(
            "Wilayah", filters, size=10
        ).to_dict("records")
    if "top" in q and "kec" in q:
        extra["top_kecamatan"] = es_utils.top_counts(
            "Kecamatan", filters, size=10
        ).to_dict("records")
    try:
        s = es_utils.summary_for_filters(filters, min_n_kec=20)
        risiko_pct = s.get("risiko_pct", {})
        if any(w in q for w in ["anemia", "hb"]):
            extra["risiko_anemia_pct"] = risiko_pct.get("anemia_hb_lt_11")
        if any(w in q for w in ["bblr", "berat lahir"]):
            extra["risiko_bblr_pct"] = risiko_pct.get("bblr_lt_2500")
        if "lila" in q:
            extra["risiko_lila_low_pct"] = risiko_pct.get("lila_lt_23_5")
        if "bmi" in q:
            extra["risiko_bmi_low_pct"] = risiko_pct.get("bmi_lt_18_5")
        if "anc" in q:
            extra["risiko_anc_low_pct"] = risiko_pct.get("anc_le_2")
        if "asi" in q:
            extra["asi_eks_tidak_pct"] = risiko_pct.get("asi_eks_tidak")
    except Exception:
        pass
    return extra


# ================== LLM helper (diubah sesuai Quickstart) ==================
def _call_llm(full_prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ API Key untuk Google Gemini (GEMINI_API_KEY) belum diatur."
    try:
        # 1. Inisialisasi client (otomatis pakai API Key dari environment)
        client = genai.Client()

        # 2. Panggil model untuk generate content
        try:
            # SDK terbaru: gunakan 'config' (bukan 'generation_config')
            response = client.models.generate_content(
                model="models/gemma-3-27b-it",
                contents=full_prompt,
                config={
                    "temperature": 0.25,
                    "max_output_tokens": 1200,
                },
            )
        except TypeError:
            # Fallback untuk kompatibilitas: panggil tanpa config
            response = client.models.generate_content(
                model="models/gemma-3-27b-it",
                contents=full_prompt,
            )
        return response.text
    except Exception as e:
        return f"⚠️ Gagal memanggil Google Gemini API: {e}"


def _render_loader(text: str = "AI sedang menganalisis..."):
    # Circle loading animation shown inside the assistant chat bubble
    loader_html = f"""
    <div class="sl-loader-wrap">
        <div class="sl-loader"></div>
        <div class="sl-loader-text">{text}</div>
    </div>
    <style>
    .sl-loader-wrap {{ display:flex; align-items:center; gap:10px; padding:6px 0; }}
    .sl-loader {{
        width: 22px; height: 22px;
        border: 3px solid rgba(0,0,0,0.1);
        border-top-color: #FF4B4B;
        border-radius: 50%;
        animation: sl-spin 0.9s linear infinite;
    }}
    @keyframes sl-spin {{ to {{ transform: rotate(360deg); }} }}
    .sl-loader-text {{ color: #666; font-size: 0.9rem; }}
    </style>
    """
    ph = st.empty()
    ph.markdown(loader_html, unsafe_allow_html=True)
    return ph


def build_final_prompt(question: str, context_json: dict) -> str:
    # Menggabungkan peran AI, pertanyaan user, dan konteks data menjadi satu prompt final
    context_str = json.dumps(context_json, ensure_ascii=False, indent=2)
    return (
        f"{PROMPT_ROLE}\n\n"
        "--- END OF ROLE ---\n\n"
        f"**Pertanyaan Pengguna:**\n{question}\n\n"
        "**Data Kontekstual (JSON):**\n"
        f"{context_str}\n\n"
        "**Tugas Anda:**\n"
        "Berdasarkan peran Anda di atas dan HANYA menggunakan data JSON yang diberikan, jawablah pertanyaan pengguna."
    ).strip()


# ================== UI ==================
SUGGESTED = [
    "Ringkas kondisi wilayah sesuai filter ini",
    "Top 5 kecamatan paling berisiko dan alasannya",
    "Apakah anemia & ANC rendah dominan? Apa implikasinya?",
    "Rekomendasi quick wins vs struktural untuk wilayah ini",
    "Bagaimana tren bulanan risiko stunting?",
]


def render_page():
    st.title("InsightNow — Chatbot Analitik")
    if not GEMINI_API_KEY:
        st.info("Fitur AI tidak aktif karena GEMINI_API_KEY belum diatur.")

    flt = sidebar.render()
    st.caption("Filter aktif akan selalu digunakan dalam analisis AI.")

    if "alias_w" not in st.session_state:
        try:
            wilayah_names = _terms(
                es.STUNTING_INDEX, "nama_kabupaten_kota", 500
            ) or _terms(es.STUNTING_INDEX, "Wilayah", 500)
            st.session_state.alias_w = build_alias_index(wilayah_names)
            st.session_state.kec2wil = kecamatan_to_wilayah_map()
        except Exception:
            st.session_state.alias_w, st.session_state.kec2wil = {}, {}

    if "alias_k" not in st.session_state:
        try:
            st.session_state.alias_k = build_alias_index(
                _terms(es.STUNTING_INDEX, "Kecamatan", 5000)
            )
        except Exception:
            st.session_state.alias_k = {}

    if "ins_chat" not in st.session_state:
        st.session_state.ins_chat = [
            {
                "role": "assistant",
                "content": "Halo! Saya **Stunlytic** 👋\n\nTanyakan apa saja tentang data stunting.",
            }
        ]

    with st.expander("Contoh pertanyaan", expanded=True):
        cols = st.columns(len(SUGGESTED))
        for i, utt in enumerate(SUGGESTED):
            if cols[i].button(utt, use_container_width=True):
                # Simpan pertanyaan yang diklik dan trigger rerun
                st.session_state.pending_user_msg = utt
                st.rerun()

        
    for m in st.session_state.ins_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Selalu render chat_input agar input form tetap muncul
    chat_box_val = st.chat_input("Tanya data stunting…")
    # Ambil pending message (jika ada); jika tidak ada, pakai input dari chat box
    user_msg = st.session_state.pop("pending_user_msg", None) or chat_box_val
    if not user_msg:
        return

    st.session_state.ins_chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    if not GEMINI_API_KEY:
        st.error("Tidak bisa memproses permintaan karena API Key belum diatur.")
        st.stop()

    targets_w = detect_wilayah_in_text(user_msg, st.session_state.alias_w)
    targets_k = detect_kecamatan_in_text(user_msg, st.session_state.alias_k)
    if targets_k and not targets_w:
        derived_w = sorted(
            {
                st.session_state.kec2wil.get(k)
                for k in targets_k
                if st.session_state.kec2wil.get(k)
            }
        )
        if derived_w:
            targets_w = derived_w

    chat_filters = dict(flt)
    chat_filters["wilayah_field"], chat_filters["kecamatan_field"] = (
        "nama_kabupaten_kota",
        "Kecamatan",
    )
    if targets_w:
        chat_filters["wilayah"] = targets_w
    if targets_k:
        chat_filters["kecamatan"] = targets_k

    with st.spinner("Mengambil ringkasan data dari server..."):
        summary = es_utils.summary_for_filters(chat_filters, min_n_kec=20)
        try:
            summary.setdefault("indikator_utama", {})["jumlah_balita"] = balita_total(
                chat_filters
            )
        except Exception:
            pass
        try:
            if "trend_bulanan" not in summary or not summary["trend_bulanan"]:
                summary["trend_bulanan"] = es_utils.trend_monthly(chat_filters)
        except Exception:
            pass

    extra = _route_extra(user_msg, chat_filters)
    context = {"filters": chat_filters, "summary": summary, "extra": extra}

    final_prompt = build_final_prompt(user_msg, context)

    # Ganti spinner global dengan loader non-blocking di dalam bubble asisten
    with st.chat_message("assistant"):
        loader_ph = _render_loader("AI sedang menganalisis...")
        try:
            answer = _call_llm(final_prompt)
        finally:
            loader_ph.empty()
        st.markdown(answer)

    st.session_state.ins_chat.append({"role": "assistant", "content": answer})


if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide", page_title="InsightNow")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
