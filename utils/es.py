# utils/es.py
import os, requests
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# --- (opsional) load .env ---
try:
    from pathlib import Path
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")
except Exception:
    pass

ES_URL = os.getenv("ES_URL", "http://178.128.219.5:9200")
STUNTING_INDEX = os.getenv("STUNTING_INDEX", "stunting-data")
BALITA_INDEX = os.getenv("BALITA_INDEX", "jabar-balita-desa")
NUTRITION_INDEX = os.getenv("NUTRITION_INDEX", "jabar-tenaga-gizi")


# ------------------- HTTP helpers -------------------
def _es_post(index: str, path: str, body: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    r = requests.post(f"{ES_URL}/{index}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _es_get(index: str, path: str, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(f"{ES_URL}/{index}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def ping() -> Tuple[bool, str]:
    try:
        r = requests.get(ES_URL, timeout=5)
        return r.status_code == 200, f"ES {ES_URL} status {r.status_code}"
    except Exception as e:
        return False, f"Gagal hubungi ES: {e}"

# ------------------- filter & query builder -------------------
RISK_LABELS = {
    "Zona 3 (>=0.70)": {"gte": 0.70},
    "Zona 2 (0.40-<0.70)": {"gte": 0.40, "lt": 0.70},
    "Zona 1 (0.10-<0.40)": {"gte": 0.10, "lt": 0.40},
    "Zona 0 (<0.10)": {"lt": 0.10},
}

def _date_range(field: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
    if not (start or end):
        return {"match_all": {}}
    rng = {}
    if start: rng["gte"] = start
    if end:   rng["lte"] = end
    return {"range": {field: rng}}

# ==== kandidat field tanpa ".keyword" ====
CANDIDATES_WILAYAH   = ["nama_kabupaten_kota", "Wilayah", "bps_nama_kabupaten_kota"]
CANDIDATES_KECAMATAN = ["Kecamatan", "bps_nama_kecamatan"]

def build_query(filters: Dict[str, Any]) -> Dict[str, Any]:
    must = []
    if filters.get("date_from") or filters.get("date_to"):
        must.append(_date_range("Tanggal", filters.get("date_from"), filters.get("date_to")))

    # pakai field yang terdeteksi oleh sidebar
    field_w = filters.get("wilayah_field") or "Wilayah"
    field_k = filters.get("kecamatan_field") or "Kecamatan"

    if filters.get("wilayah"):
        must.append({"terms": {field_w: filters["wilayah"]}})
    if filters.get("kecamatan"):
        must.append({"terms": {field_k: filters["kecamatan"]}})

    if filters.get("risk_level"):
        ranges = []
        for rl in filters["risk_level"]:
            if rl == "Zona 3 (>=0.70)":
                ranges.append({"range": {"Probabilitas Stunting (simulasi)": {"gte": 0.70}}})
            elif rl == "Zona 2 (0.40-<0.70)":
                ranges.append({"range": {"Probabilitas Stunting (simulasi)": {"gte": 0.40, "lt": 0.70}}})
            elif rl == "Zona 1 (0.10-<0.40)":
                ranges.append({"range": {"Probabilitas Stunting (simulasi)": {"gte": 0.10, "lt": 0.40}}})
            elif rl == "Zona 0 (<0.10)":
                ranges.append({"range": {"Probabilitas Stunting (simulasi)": {"lt": 0.10}}})
        if ranges:
            must.append({"bool": {"should": ranges, "minimum_should_match": 1}})
    return {"query": {"bool": {"must": must}}}


# ------------------- sampler & KPI ringkas -------------------
def fetch_sample(filters: Dict[str, Any], size: int = 3000, fields: Optional[List[str]] = None) -> pd.DataFrame:
    body = build_query(filters)
    body.update({"_source": fields or True, "size": size, "track_total_hits": True})
    data = _es_post(STUNTING_INDEX, "/_search", body)
    hits = data.get("hits", {}).get("hits", [])
    return pd.DataFrame([h.get("_source", {}) for h in hits])

def count_stunting_and_total(filters: Dict[str, Any]) -> Dict[str, Any]:
    """total = semua dokumen sesuai filter; stunting = biner OR kategori OR Z<=-2."""
    stunting_labels = ["Stunting","Ya","YA","ya","1","true","TRUE","True"]
    body = build_query(filters)
    body.update({
        "size": 0,
        "track_total_hits": True,
        "aggs": {
            "total": {"filter": {"match_all": {}}},
            "stunting_any": {
                "filter": {
                    "bool": {
                        "should": [
                            {"terms": {"Status Stunting (Biner)": stunting_labels}},
                            {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting","stunting"]}},
                            {"range": {"Z-Score TB/U": {"lte": -2.0}}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
        }
    })
    data = _es_post(STUNTING_INDEX, "/_search", body)
    tot = int(data["aggregations"]["total"]["doc_count"])
    st  = int(data["aggregations"]["stunting_any"]["doc_count"])
    return {"total": tot, "stunting": st, "ratio": (st / tot) if tot else None}

def coverage_immunization(filters: Dict[str, Any]) -> Optional[float]:
    """Cakupan 'lengkap' di salah satu dari 2 kolom (fallback)."""
    for fld in ["Imunisasi (lengkap/tidak lengkap)", "Status Imunisasi Anak"]:
        body = build_query(filters)
        body.update({
            "size": 0,
            "aggs": {
                "complete": {"filter": {"terms": {fld: ["lengkap","Lengkap","complete","Complete"]}}},
                "total": {"value_count": {"field": fld}}
            }
        })
        try:
            res = _es_post(STUNTING_INDEX, "/_search", body)
            tot = res["aggregations"]["total"]["value"]
            comp = res["aggregations"]["complete"]["doc_count"]
            if tot: return comp / tot
        except Exception:
            continue
    return None

def coverage_safe_water(filters: Dict[str, Any]) -> Optional[float]:
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "ok1": {"filter": {"terms": {"Akses Air": ["Layak","Ya","Bersih","Aman"]}}},
            "ok2": {"filter": {"terms": {"Akses Air Bersih": ["Layak","Ya","Bersih","Aman"]}}},
            "total1": {"value_count": {"field": "Akses Air"}},
            "total2": {"value_count": {"field": "Akses Air Bersih"}},
        }
    })
    res = _es_post(STUNTING_INDEX, "/_search", body)
    t1, t2 = res["aggregations"]["total1"]["value"], res["aggregations"]["total2"]["value"]
    n1, n2 = res["aggregations"]["ok1"]["doc_count"], res["aggregations"]["ok2"]["doc_count"]
    denom = (t1 or 0) + (t2 or 0)
    return ((n1 or 0) + (n2 or 0)) / denom if denom else None

# ------------------- Nakes (index jabar-tenaga-gizi) -------------------
def jumlah_nakes(filters: Dict[str, Any]) -> int:
    must: List[Dict[str, Any]] = []
    if filters.get("wilayah"):
        must.append({"terms": {"nama_kabupaten_kota": filters["wilayah"]}})
    # tahun dari rentang tanggal (ambil yyyy)
    yr = {}
    if filters.get("date_from"): yr["gte"] = filters["date_from"][:4]
    if filters.get("date_to"):   yr["lte"] = filters["date_to"][:4]
    if yr: must.append({"range": {"tahun": yr}})
    body = {"query": {"bool": {"must": must}} if must else {"match_all": {}},
            "size": 0, "aggs": {"sum_nakes": {"sum": {"field": "jumlah_nakes_gizi"}}}}
    data = _es_post(NUTRITION_INDEX, "/_search", body)
    return int(round(data["aggregations"]["sum_nakes"]["value"] or 0))


def trend_monthly(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Seri waktu bulanan: total dokumen, jumlah stunting, % stunting, avg probabilitas.
    Menghormati semua filter yang aktif.
    """
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "per_month": {
                "date_histogram": {"field": "Tanggal", "calendar_interval": "month"},
                "aggs": {
                    "stunting_any": {"filter": {"bool": {"should": [
                        {"terms": {"Status Stunting (Biner)": ["Stunting","Ya","YA","ya","1","true","TRUE","True"]}},
                        {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting","stunting"]}},
                        {"range": {"Z-Score TB/U": {"lte": -2.0}}}
                    ], "minimum_should_match": 1}}},
                    "tot": {"filter": {"match_all": {}}},
                    "avg_prob": {"avg": {"field": "Probabilitas Stunting (simulasi)"}}
                }
            }
        }
    })
    res = _es_post(STUNTING_INDEX, "/_search", body)
    out = []
    for b in res["aggregations"]["per_month"]["buckets"]:
        tot = b["tot"]["doc_count"]
        stg = b["stunting_any"]["doc_count"]
        pct = (stg / tot * 100) if tot else 0.0
        out.append({
            "periode": b["key_as_string"][:7],
            "total": tot,
            "stunting": stg,
            "stunting_pct": round(pct, 2),
            "avg_prob": b["avg_prob"]["value"],
        })
    return out

# ------------------- agregasi level wilayah/kecamatan -------------------
def _agg_terms(level_field: str, filters: Dict[str, Any], size: int = 2000) -> pd.DataFrame:
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "by": {
                "terms": {"field": level_field, "size": size},
                "aggs": {
                    "stunting": {"filter": {"bool": {"should": [
                        {"terms": {"Status Stunting (Biner)": ["Stunting","Ya","YA","ya","1","true","TRUE","True"]}},
                        {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting","stunting"]}},
                        {"range": {"Z-Score TB/U": {"lte": -2.0}}}
                    ], "minimum_should_match": 1}}}
                }
            }
        }
    })
    data = _es_post(STUNTING_INDEX, "/_search", body)
    buckets = data.get("aggregations", {}).get("by", {}).get("buckets", [])
    rows = [{"key": b["key"], "jumlah_anak": b["doc_count"], "jumlah_stunting": b["stunting"]["doc_count"]}
            for b in buckets]
    # >>> penting: selalu kembalikan schema yang sama meski kosong
    return pd.DataFrame(rows, columns=["key", "jumlah_anak", "jumlah_stunting"])
def _terms_df_with_candidates(filters: Dict[str, Any], candidates: List[str], size: int = 1000) -> pd.DataFrame:
    for field in candidates:
        body = build_query(filters)
        body.update({
            "size": 0,
            "aggs": {
                "by": {
                    "terms": {"field": field, "size": size},
                    "aggs": {
                        "stunting": {"filter": {"bool": {"should": [
                            {"terms": {"Status Stunting (Biner)": ["Stunting","Ya","YA","ya","1","true","TRUE","True"]}},
                            {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting","stunting"]}},
                            {"range": {"Z-Score TB/U": {"lte": -2.0}}}
                        ], "minimum_should_match": 1}}}
                    }
                }
            }
        })
        try:
            data = _es_post(STUNTING_INDEX, "/_search", body)
            buckets = data["aggregations"]["by"]["buckets"]
            if buckets:
                rows = [{"key": b["key"], "jumlah_anak": b["doc_count"], "jumlah_stunting": b["stunting"]["doc_count"]}
                        for b in buckets]
                return pd.DataFrame(rows, columns=["key","jumlah_anak","jumlah_stunting"])
        except Exception:
            continue
    return pd.DataFrame(columns=["key","jumlah_anak","jumlah_stunting"])


def top_counts(level: str, filters: Dict[str, Any], size: int = 10) -> pd.DataFrame:
    if level.lower().startswith("wil"):
        df = _terms_df_with_candidates(filters, CANDIDATES_WILAYAH, size=2000).rename(columns={"key":"Wilayah"})
    else:
        df = _terms_df_with_candidates(filters, CANDIDATES_KECAMATAN, size=3000).rename(columns={"key":"Kecamatan"})
    if df.empty: return df
    return df.sort_values("jumlah_stunting", ascending=False).head(size)

def counts_by_level(level: str, filters: Dict[str, Any]) -> pd.DataFrame:
    if level.lower().startswith("wil"):
        return _terms_df_with_candidates(filters, CANDIDATES_WILAYAH, size=2000).rename(columns={"key":"Wilayah"})
    else:
        return _terms_df_with_candidates(filters, CANDIDATES_KECAMATAN, size=3000).rename(columns={"key":"Kecamatan"})


def kecamatan_table(filters: Dict[str, Any], min_n: int = 20) -> pd.DataFrame:
    """Ringkasan per-kecamatan: avg_prob, %stunting, %anemia, %BBLR, %LiLA<23.5, %ANC<=2."""
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "kec": {
                "terms": {"field": "Kecamatan", "size": 5000},  # << tanpa .keyword
                "aggs": {
                    "avg_prob": {"avg": {"field": "Probabilitas Stunting (simulasi)"}},
                    "stunting": {"filter": {"bool": {"should": [
                        {"terms": {"Status Stunting (Biner)": ["Stunting","Ya","YA","ya","1","true","TRUE","True"]}},
                        {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting","stunting"]}},
                        {"range": {"Z-Score TB/U": {"lte": -2.0}}}
                    ], "minimum_should_match": 1}}},
                    "anemia":   {"filter": {"range": {"Hb (g/dL)": {"lt": 11.0}}}},
                    "bblr":     {"filter": {"range": {"Berat Lahir (gram)": {"lt": 2500}}}},
                    "lila_low": {"filter": {"range": {"LiLA saat Hamil (cm)": {"lt": 23.5}}}},
                    "anc_low":  {"filter": {"range": {"Kunjungan ANC (x)": {"lte": 2}}}},
                    "sample_wil": {"top_hits": {"_source": {"includes": ["nama_kabupaten_kota","Wilayah"]}, "size": 1}}
                }
            }
        }
    })
    data = _es_post(STUNTING_INDEX, "/_search", body)
    rows = []
    for b in data["aggregations"]["kec"]["buckets"]:
        n = b["doc_count"]
        if n < min_n:
            continue
        wil = None
        try:
            src = b["sample_wil"]["hits"]["hits"][0]["_source"]
            wil = src.get("nama_kabupaten_kota") or src.get("Wilayah")
        except Exception:
            pass
        rows.append({
            "Wilayah": wil,
            "Kecamatan": b["key"],
            "n": n,
            "avg_prob": b["avg_prob"]["value"],
            "stunting_pct": round(100.0 * b["stunting"]["doc_count"] / n, 2),
            "anemia_pct":   round(100.0 * b["anemia"]["doc_count"]   / n, 2),
            "bblr_pct":     round(100.0 * b["bblr"]["doc_count"]     / n, 2),
            "lila_low_pct": round(100.0 * b["lila_low"]["doc_count"] / n, 2),
            "anc_low_pct":  round(100.0 * b["anc_low"]["doc_count"]  / n, 2),
        })
    df = pd.DataFrame(rows, columns=["Wilayah","Kecamatan","n","avg_prob","stunting_pct","anemia_pct","bblr_pct","lila_low_pct","anc_low_pct"])
    if not df.empty:
        df = df.sort_values(["stunting_pct","avg_prob"], ascending=[False, False])
    return df


def summary_for_filters(filters: Dict[str, Any], min_n_kec: int = 30) -> Dict[str, Any]:
    """Ringkasan padat untuk InsightNow & panel lain — setara pola di beta.py."""
    # inti
    cards = count_stunting_and_total(filters)
    imun  = coverage_immunization(filters)
    air   = coverage_safe_water(filters)

    # agregat & distribusi
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "avg_prob": {"avg": {"field": "Probabilitas Stunting (simulasi)"}},
            "avg_bmi":  {"avg": {"field": "BMI Pra-Hamil"}},
            "avg_lila": {"avg": {"field": "LiLA saat Hamil (cm)"}},
            "avg_hb":   {"avg": {"field": "Hb (g/dL)"}},
            "avg_upah": {"avg": {"field": "Upah Keluarga (Rp/bulan)"}},
            "avg_ump":  {"avg": {"field": "Rata-rata UMP Wilayah (Rp/bulan)"}},
            "pct_bmi":  {"percentiles": {"field": "BMI Pra-Hamil", "percents": [5,25,50,75,95]}},
            "pct_lila": {"percentiles": {"field": "LiLA saat Hamil (cm)", "percents": [5,25,50,75,95]}},
            "pct_hb":   {"percentiles": {"field": "Hb (g/dL)", "percents": [5,25,50,75,95]}},
            "pct_z":    {"percentiles": {"field": "Z-Score TB/U", "percents": [5,25,50,75,95]}},
            "pendidikan": {"terms": {"field": "Pendidikan Ibu", "size": 10}},
            "air_bersih": {"terms": {"field": "Akses Air Bersih", "size": 10}},
            "imunisasi":  {"terms": {"field": "Status Imunisasi Anak", "size": 10}},
            "status_biner":{"terms": {"field": "Status Stunting (Biner)", "size": 10}},
            "tipe_wilayah":{"terms": {"field": "Tipe Wilayah", "size": 10}},
            "rokok":      {"terms": {"field": "Paparan Asap Rokok", "size": 10}},
            "bantuan":    {"terms": {"field": "Kepesertaan Program Bantuan", "size": 10}},
            "asi":        {"terms": {"field": "ASI Eksklusif", "size": 10}},
            "pekerjaan":  {"terms": {"field": "Jenis Pekerjaan Orang Tua", "size": 15}},
            # risiko biner
            "risk_bblr":      {"filter": {"range": {"Berat Lahir (gram)": {"lt": 2500}}}},
            "risk_anemia":    {"filter": {"range": {"Hb (g/dL)": {"lt": 11.0}}}},
            "risk_lila":      {"filter": {"range": {"LiLA saat Hamil (cm)": {"lt": 23.5}}}},
            "risk_bmi_low":   {"filter": {"range": {"BMI Pra-Hamil": {"lt": 18.5}}}},
            "risk_anc_low":   {"filter": {"range": {"Kunjungan ANC (x)": {"lte": 2}}}},
            "risk_z_stunt":   {"filter": {"range": {"Z-Score TB/U": {"lte": -2.0}}}},
            "risk_asi_tidak": {"filter": {"terms": {"ASI Eksklusif": ["Tidak","tidak","No","no"]}}},
            # histogram
            "usia_anak": {"histogram": {"field": "Usia Anak (bulan)", "interval": 6}},
            "usia_ibu":  {"histogram": {"field": "Usia Ibu saat Hamil (tahun)", "interval": 5}},
        }
    })
    data = _es_post(STUNTING_INDEX, "/_search", body)
    agg = data["aggregations"]

    # helper
    total = max(1, int(cards["total"]))
    def pct(n: int) -> float: return round(100.0 * n / total, 2)
    def pvals(a): 
        v = a.get("values", {}) if a else {}
        return {k: v.get(k) for k in ["5.0","25.0","50.0","75.0","95.0"]}
    def dist(bkts): 
        return [{"key": b["key"], "count": b["doc_count"], "pct": pct(b["doc_count"])} for b in bkts]

    # rangkum kecamatan (top/bottom) berdasarkan % stunting
    try:
        df_kec = kecamatan_table(filters, min_n=min_n_kec)
        top = df_kec.nlargest(5, "stunting_pct")[["Wilayah","Kecamatan","n","stunting_pct","avg_prob"]].to_dict("records")
        bot = df_kec.nsmallest(5, "stunting_pct")[["Wilayah","Kecamatan","n","stunting_pct","avg_prob"]].to_dict("records")
        kec_summary = {"min_n": min_n_kec, "considered": int(df_kec.shape[0]), "top": top, "bottom": bot}
    except Exception:
        kec_summary = {"min_n": min_n_kec, "considered": 0, "top": [], "bottom": []}

    avg_upah, avg_ump = agg["avg_upah"]["value"], agg["avg_ump"]["value"]
    rasio_upah_ump = (avg_upah / avg_ump) if (avg_upah and avg_ump and avg_ump != 0) else None

    trend = []
    try:
        trend = trend_monthly(filters)[-24:]  # ambil 24 bulan terakhir
    except Exception:
        trend = []

    return {
        "filters": filters,
        "indikator_utama": {
            "total_lahir": cards["total"],
            "total_stunting": cards["stunting"],
            "rasio_stunting": cards["ratio"],    # 0..1
            "cakupan_imunisasi": coverage_immunization(filters),
            "akses_air_layak": coverage_safe_water(filters),
            "jumlah_nakes_gizi": jumlah_nakes(filters),
        },
        "stat_rerata": {
            "avg_prob": agg["avg_prob"]["value"],
            "avg_bmi":  agg["avg_bmi"]["value"],
            "avg_lila": agg["avg_lila"]["value"],
            "avg_hb":   agg["avg_hb"]["value"],
            "avg_upah": avg_upah,
            "avg_ump":  avg_ump,
            "rasio_upah_ump": rasio_upah_ump,
        },
        "percentiles": {
            "bmi":  pvals(agg.get("pct_bmi")),
            "lila": pvals(agg.get("pct_lila")),
            "hb":   pvals(agg.get("pct_hb")),
            "z":    pvals(agg.get("pct_z")),
        },
        "distribusi": {
            "pendidikan":   dist(agg["pendidikan"]["buckets"]),
            "air_bersih":   dist(agg["air_bersih"]["buckets"]),
            "imunisasi":    dist(agg["imunisasi"]["buckets"]),
            "status_biner": dist(agg["status_biner"]["buckets"]),
            "tipe_wilayah": dist(agg["tipe_wilayah"]["buckets"]),
            "rokok":        dist(agg["rokok"]["buckets"]),
            "bantuan":      dist(agg["bantuan"]["buckets"]),
            "asi":          dist(agg["asi"]["buckets"]),
            "pekerjaan":    dist(agg["pekerjaan"]["buckets"]),
        },
        "risiko_count": {
            "bblr_lt_2500":    agg["risk_bblr"]["doc_count"],
            "anemia_hb_lt_11": agg["risk_anemia"]["doc_count"],
            "lila_lt_23_5":    agg["risk_lila"]["doc_count"],
            "bmi_lt_18_5":     agg["risk_bmi_low"]["doc_count"],
            "anc_le_2":        agg["risk_anc_low"]["doc_count"],
            "zscore_stunting": agg["risk_z_stunt"]["doc_count"],
            "asi_eks_tidak":   agg["risk_asi_tidak"]["doc_count"],
        },
        "risiko_pct": {
            "bblr_lt_2500":    pct(agg["risk_bblr"]["doc_count"]),
            "anemia_hb_lt_11": pct(agg["risk_anemia"]["doc_count"]),
            "lila_lt_23_5":    pct(agg["risk_lila"]["doc_count"]),
            "bmi_lt_18_5":     pct(agg["risk_bmi_low"]["doc_count"]),
            "anc_le_2":        pct(agg["risk_anc_low"]["doc_count"]),
            "zscore_stunting": pct(agg["risk_z_stunt"]["doc_count"]),
            "asi_eks_tidak":   pct(agg["risk_asi_tidak"]["doc_count"]),
        },
        "histogram": {
            "usia_anak_6_bulanan": [{"bin_start": b["key"], "count": b["doc_count"], "pct": pct(b["doc_count"])} for b in agg["usia_anak"]["buckets"]],
            "usia_ibu_5_tahunan":  [{"bin_start": b["key"], "count": b["doc_count"], "pct": pct(b["doc_count"])} for b in agg["usia_ibu"]["buckets"]],
        },
        "kecamatan_rank": kec_summary,
        "top10_kabupaten": top_counts("Wilayah", filters, size=10).to_dict("records"),
        "top10_kecamatan": top_counts("Kecamatan", filters, size=10).to_dict("records"),
        "trend_bulanan": trend,  # <<— BARU


    }

# ------------------- untuk korelasi -------------------
def numeric_sample_for_corr(filters: Dict[str, Any], size: int = 5000) -> pd.DataFrame:
    df = fetch_sample(filters, size=size)
    if df.empty: return df
    return df.select_dtypes(include=["number"]).copy()
