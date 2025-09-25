import streamlit as st
import os
import json
import openai  # Mengganti requests dengan library resmi OpenAI
from src import styles, elastic_client as es
from src.components import sidebar


# --- FUNGSI AKSES LLM (DI-UPGRADE KE OPENAI) ---
def _get_openai_api_key():
    # Mencari OPENAI_API_KEY
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


# stunting_app/components/insightnow.py
import os, json, re
import json
import streamlit as st
from openai import OpenAI
from utils import es
# from utils.filters import sidebar_filters
from textwrap import dedent

# ================== PROMPT ROLE (pakai yang versi 3 indeks) ==================
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


# ================== OpenAI config ==================
MODEL_ID = os.getenv("INSIGHT_MODEL_ID", "gpt-4.1-nano")
OPENAI_API_KEY = _get_openai_api_key()  # perbaikan: panggil fungsi untuk mengambil API key

# ================== Utils: alias & deteksi entitas ==================
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    repl = {"kabupaten ": "kab ", "kab. ": "kab ", "kota ": "kota ", "kab ": "kab "}
    for k, v in repl.items(): s = s.replace(k, v)
    return s

def build_alias_index(names: list) -> dict:
    idx = {}
    for n in names:
        if not isinstance(n, str): continue
        base = n.strip()
        variants = {base, base.lower(), base.lower().title(), _norm(base)}
        for v in variants: idx[v] = base
    return idx

def detect_wilayah_in_text(text: str, alias_idx: dict, max_matches: int = 3) -> list:
    t = _norm(text); found, seen = [], set()
    for alias, resmi in alias_idx.items():
        if alias in t and resmi not in seen:
            seen.add(resmi); found.append(resmi)
            if len(found) >= max_matches: break
    return found

def detect_kecamatan_in_text(text: str, alias_idx: dict, max_matches: int = 5) -> list:
    t = _norm(text); found, seen = [], set()
    for alias, resmi in alias_idx.items():
        if alias in t and resmi not in seen:
            seen.add(resmi); found.append(resmi)
            if len(found) >= max_matches: break
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
                    "wil": {"top_hits": {"_source": {"includes": ["nama_kabupaten_kota","Wilayah"]}, "size": 1}}
                }
            }
        }
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
        if filters.get("date_from"): yr["gte"] = filters["date_from"][:4]
        if filters.get("date_to"):   yr["lte"] = filters["date_to"][:4]
        if yr: must.append({"range": {"tahun": yr}})
    q = {"bool": {"must": must}} if must else {"match_all": {}}
    body = {"query": q, "size": 0, "aggs": {"sum": {"sum": {"field": "jumlah_balita"}}}}
    try:
        data = es._es_post(es.BALITA_INDEX, "/_search", body)
        return int(round(data["aggregations"]["sum"]["value"] or 0))
    except Exception:
        return None

# ================== Router tambahan (tambahkan tren/top dsb) ==================
def _route_extra(question: str, filters: dict) -> dict:
    q = (question or "").lower()
    extra = {}

    # Tren bulanan
    if any(k in q for k in ["tren", "trend", "bulan", "bulanan"]):
        extra["tren_bulanan"] = es.trend_monthly(filters)

    # Top wilayah/kecamatan
    if "top" in q and any(k in q for k in ["kab", "kabupaten", "kota"]):
        extra["top_kabupaten"] = es.top_counts("Wilayah", filters, size=10).to_dict("records")
    if "top" in q and "kec" in q:
        extra["top_kecamatan"] = es.top_counts("Kecamatan", filters, size=10).to_dict("records")

    # Metrik risiko spesifik
    try:
        s = es.summary_for_filters(filters, min_n_kec=20)
        if any(w in q for w in ["anemia","hb"]):
            extra["risiko_anemia_pct"] = s["risiko_pct"]["anemia_hb_lt_11"]
        if any(w in q for w in ["bblr","berat lahir"]):
            extra["risiko_bblr_pct"] = s["risiko_pct"]["bblr_lt_2500"]
        if "lila" in q:
            extra["risiko_lila_low_pct"] = s["risiko_pct"]["lila_lt_23_5"]
        if "bmi" in q:
            extra["risiko_bmi_low_pct"] = s["risiko_pct"]["bmi_lt_18_5"]
        if "anc" in q:
            extra["risiko_anc_low_pct"] = s["risiko_pct"]["anc_le_2"]
        if "asi" in q:
            extra["asi_eks_tidak_pct"] = s["risiko_pct"]["asi_eks_tidak"]
    except Exception:
        pass

    return extra

# ================== LLM helper ==================
def _call_llm(messages: list) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY belum diatur (env/secrets)."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.25,
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Gagal memanggil OpenAI API: {e}"

# ================== System prompt builder ==================
def _build_system_prompt(context_json: dict) -> str:
    # Isi “Batasan tambahan” agar model WAJIB pakai data tren kalau ada
    return (
        PROMPT_ROLE
        + "\n\nBatasan Tambahan:\n"
        + "- Jawab HANYA berdasarkan 'context_json' yang diberikan (bagian 'summary' dan 'extra').\n"
        + "- Jika 'extra.tren_bulanan' ADA, WAJIB jelaskan tren bulanan (naik/turun/stabil), bulan puncak & terendah, dan kisaran persentasenya.\n"
        + "- Jika 'extra.tren_bulanan' TIDAK ADA, gunakan 'summary.trend_bulanan' bila tersedia.\n"
        + "- Jika metrik null/tidak tersedia, tulis 'tidak tersedia' tanpa mengarang."
    ).strip()

# ================== UI (chatbot ala beta.py) ==================
SUGGESTED = [
    "Ringkas kondisi wilayah sesuai filter ini",
    "Top 5 kecamatan paling berisiko dan alasannya",
    "Apakah anemia & ANC rendah dominan? Apa implikasinya?",
    "Rekomendasi quick wins vs struktural untuk wilayah ini",
    "Bagaimana tren bulanan risiko stunting?"
]

# --- RENDER HALAMAN ---
def render_page():
    st.title("InsightNow — Chatbot")
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY belum diatur di environment/secrets.")

    # ===== Filter dari sidebar =====
    flt = sidebar.render()
    st.caption("Chat selalu menghormati filter aktif.")

    # ===== Kamus alias nama wilayah/kecamatan =====
    try:
        wilayah_names = _terms(es.STUNTING_INDEX, "nama_kabupaten_kota", 500) or \
                        _terms(es.STUNTING_INDEX, "Wilayah", 500)
    except Exception:
        wilayah_names = []
    alias_w = build_alias_index(wilayah_names)

    try:
        must = []
        if flt.get("wilayah"):
            must.append({"terms": {(flt.get("wilayah_field") or "nama_kabupaten_kota"): flt["wilayah"]}})
        body = {"query": {"bool": {"must": must}}} if must else {"query": {"match_all": {}}}
        body.update({"size": 0, "aggs": {"k": {"terms": {"field": "Kecamatan", "size": 5000}}}})
        data = es._es_post(es.STUNTING_INDEX, "/_search", body)
        kecamatan_names = [b["key"] for b in data["aggregations"]["k"]["buckets"]]
    except Exception:
        kecamatan_names = []
    alias_k = build_alias_index(kecamatan_names)
    kec2wil = kecamatan_to_wilayah_map()

    # ===== History =====
    if "ins_chat" not in st.session_state:
        st.session_state.ins_chat = []

    # Jika belum ada riwayat, tampilkan sapaan awal di gelembung assistant
    if not st.session_state.ins_chat:
        welcome = (
            "Halo! Aku **Stunlytic** 👋\n\n"
            "Tanyakan apa saja tentang data stunting sesuai filter di sidebar.\n"
            "Contoh: *'Bandung vs Bekasi'*, *'kec Baleendah'*, atau *'faktor risiko tertinggi di wilayah ini?'*"
        )
        st.session_state.ins_chat.append({"role": "assistant", "content": welcome})

    # SUGGESTED chips
    with st.expander("Contoh pertanyaan", expanded=False):
        if 'SUGGESTED' in globals() and isinstance(SUGGESTED, (list, tuple)) and len(SUGGESTED) > 0:
            cols = st.columns(min(len(SUGGESTED), 4))
            for i, utt in enumerate(SUGGESTED):
                if cols[i % 4].button(utt):
                    st.session_state.ins_chat.append({"role": "user", "content": utt})
        else:
            st.caption("Belum ada daftar contoh pertanyaan.")

    # Render seluruh riwayat sebagai gelembung chat
    for m in st.session_state.ins_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ===== Input =====
    user_msg = st.chat_input("Tanya data stunting… (contoh: 'Bandung vs Bekasi', 'kec Baleendah')")
    if not user_msg:
        return  # biar UI tetap tampil tanpa menghentikan seluruh app

    # Simpan & tampilkan pesan user
    st.session_state.ins_chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # ===== Deteksi entitas =====
    targets_w = detect_wilayah_in_text(user_msg, alias_w)
    targets_k = detect_kecamatan_in_text(user_msg, alias_k)
    if targets_k and not targets_w:
        derived_w = sorted({kec2wil.get(k) for k in targets_k if kec2wil.get(k)})
        if derived_w:
            targets_w = derived_w

    # ===== Filter khusus chat =====
    chat_filters = dict(flt)
    chat_filters["wilayah_field"] = "nama_kabupaten_kota"
    chat_filters["kecamatan_field"] = "Kecamatan"
    if targets_w:
        chat_filters["wilayah"] = targets_w
    if targets_k:
        chat_filters["kecamatan"] = targets_k

    # ===== Ringkasan & ekstra =====
    with st.spinner("Mengambil ringkasan data…"):
        summary = es.summary_for_filters(chat_filters, min_n_kec=20)
        # tambahkan total balita (beban populasi)
        try:
            summary.setdefault("indikator_utama", {})["jumlah_balita"] = balita_total(chat_filters)
        except Exception:
            pass
        # pastikan trend_bulanan ada minimal dari helper
        try:
            if "trend_bulanan" not in summary or not summary["trend_bulanan"]:
                summary["trend_bulanan"] = es.trend_monthly(chat_filters)
        except Exception:
            pass

    extra = _route_extra(user_msg, chat_filters)

    # ===== context_json untuk model =====
    context = {"filters": chat_filters, "summary": summary, "extra": extra}
    system_msg = _build_system_prompt(context)
    user_payload = f"Pertanyaan:\n{user_msg}\n\ncontext_json:\n{json.dumps(context, ensure_ascii=False)}"

    # ===== Kirim ke model =====
    recent = [m for m in st.session_state.ins_chat[-8:] if m["role"] in ("user","assistant")]
    messages = [{"role": "system", "content": system_msg}, *recent, {"role": "user", "content": user_payload}]
    with st.spinner("Meminta analisis dari LLM…"):
        answer = _call_llm(messages)

    # Tampilkan jawaban di gelembung assistant + simpan
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.ins_chat.append({"role": "assistant", "content": answer})


# --- Main Execution ---
if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
