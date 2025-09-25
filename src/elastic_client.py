# StuntLytics/src/elastic_client.py
# VERSI BARU — query & logika diselaraskan dengan utils/es.py
# - Konsisten pakai field "Z-Score TB/U" (bukan "ZScore TB/U")
# - Pola stunting_any: (Status Biner) OR (Status Kategori) OR (Z-Score TB/U <= -2)
# - Tanpa pemakaian ".keyword" di source; untuk terms agg diasumsikan field bertipe keyword.
#   (Jika mapping text, aktifkan fielddata/normalizer atau tambahkan subfield keyword di ES.)

import os
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

try:
    from pathlib import Path
    from dotenv import load_dotenv

    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")
except Exception:
    pass

ES_URL = os.getenv("ES_URL", "http://178.128.219.5:9200")
STUNTING_INDEX = os.getenv("STUNTING_INDEX", "stunting-data")
NUTRITION_INDEX = os.getenv("NUTRITION_INDEX", "jabar-tenaga-gizi")

# ==== kandidat field tanpa ".keyword" (selaras dengan utils/es.py) ====
CANDIDATES_WILAYAH = ["nama_kabupaten_kota", "Wilayah", "bps_nama_kabupaten_kota"]
CANDIDATES_KECAMATAN = ["Kecamatan", "bps_nama_kecamatan"]

# ------------------- HTTP helpers -------------------
_SESSION = requests.Session()


def _es_post(index: str, path: str, body: Dict[str, Any], timeout: int = 60, retries: int = 1) -> Dict[str, Any]:
    url = f"{ES_URL}/{index}{path}"
    last = None
    for attempt in range(retries + 1):
        try:
            r = _SESSION.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last = e
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
    raise ConnectionError(f"Gagal menghubungi Elasticsearch di {url}: {last}")


def ping() -> Tuple[bool, str]:
    try:
        r = _SESSION.get(ES_URL, timeout=5)
        return r.status_code == 200, f"ES {ES_URL} status {r.status_code}"
    except Exception as e:
        return False, f"Gagal hubungi ES: {e}"


# ------------------- filter & query builder -------------------

def _date_range(field: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Dict[str, Any]:
    if not (start or end):
        return {"match_all": {}}
    rng: Dict[str, Any] = {}
    if start:
        rng["gte"] = (start.isoformat() if hasattr(start, "isoformat") else str(start))
    if end:
        rng["lte"] = (end.isoformat() if hasattr(end, "isoformat") else str(end))
    return {"range": {field: rng}}


def build_query(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Selaras utils/es.py: gunakan wilayah_field/kecamatan_field jika ada, fallback default.
    - date_from/date_to boleh pd.Timestamp atau string ISO.
    - terms diisi dari list pada filters["wilayah"]/["kecamatan"].
    """
    must: List[Dict[str, Any]] = []

    if filters.get("date_from") or filters.get("date_to"):
        must.append(_date_range("Tanggal", filters.get("date_from"), filters.get("date_to")))

    field_w = filters.get("wilayah_field") or "Wilayah"
    field_k = filters.get("kecamatan_field") or "Kecamatan"

    if filters.get("wilayah"):
        must.append({"terms": {field_w: filters["wilayah"]}})
    if filters.get("kecamatan"):
        must.append({"terms": {field_k: filters["kecamatan"]}})

    # Risk bucket (opsional, jika dipakai di beberapa layar)
    if filters.get("risk_level"):
        ranges: List[Dict[str, Any]] = []
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

    return {"query": {"bool": {"must": must}}} if must else {"query": {"match_all": {}}}


# ------------------- Pola filter stunting_any (shared) -------------------
_STUNTING_BINER = ["Stunting", "Ya", "YA", "ya", "1", "true", "TRUE", "True"]


def _stunting_any_filter() -> Dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"terms": {"Status Stunting (Biner)": _STUNTING_BINER}},
                {"terms": {"Status Stunting (Stunting / Berisiko / Normal)": ["Stunting", "stunting"]}},
                {"range": {"Z-Score TB/U": {"lte": -2.0}}},
            ],
            "minimum_should_match": 1,
        }
    }


# ------------------- Fungsi untuk Sidebar (deteksi opsi) -------------------

def get_filter_options(base_filters: Dict[str, Any], field_candidates: List[str], size: int = 500) -> Tuple[Optional[str], List[str]]:
    """Coba field candidates berurutan, return (field_terpakai, opsi_terurut)."""
    for field in field_candidates:
        try:
            body = build_query(base_filters)
            body.update({"size": 0, "aggs": {"opts": {"terms": {"field": field, "size": size}}}})
            data = _es_post(STUNTING_INDEX, "/_search", body)
            buckets = data.get("aggregations", {}).get("opts", {}).get("buckets", [])
            if buckets:
                options = [b["key"] for b in buckets]
                return field, sorted(options)
        except Exception:
            continue
    return None, []


# ------------------- Halaman Utama (summary) -------------------

def get_main_page_summary(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Ambil KPI & chart, diselaraskan dengan utils/es.py (tanpa .keyword)."""
    # 1) Query utama stunting
    stunting_body = build_query(filters)
    stunting_body.update(
        {
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "stunting_count": {"filter": _stunting_any_filter()},
                "imunisasi_lengkap": {
                    "filter": {"bool": {"should": [
                        {"terms": {"Imunisasi (lengkap/tidak lengkap)": ["lengkap", "Lengkap", "complete", "Complete"]}},
                        {"terms": {"Status Imunisasi Anak": ["lengkap", "Lengkap", "complete", "Complete"]}},
                    ], "minimum_should_match": 1}},
                },
                "total_imunisasi_field_1": {"value_count": {"field": "Imunisasi (lengkap/tidak lengkap)"}},
                "total_imunisasi_field_2": {"value_count": {"field": "Status Imunisasi Anak"}},
                "air_bersih_dist": {"terms": {"field": "Akses Air Bersih", "size": 10}},
                "imunisasi_trend": {
                    "date_histogram": {"field": "Tanggal", "calendar_interval": "month", "format": "yyyy-MM"},
                    "aggs": {
                        "imunisasi_lengkap_in_bucket": {
                            "filter": {"bool": {"should": [
                                {"terms": {"Imunisasi (lengkap/tidak lengkap)": ["lengkap", "Lengkap", "complete", "Complete"]}},
                                {"terms": {"Status Imunisasi Anak": ["lengkap", "Lengkap", "complete", "Complete"]}},
                            ], "minimum_should_match": 1}}
                        }
                    },
                },
            },
        }
    )

    # 2) Query nakes (index jabar-tenaga-gizi)
    nakes_must: List[Dict[str, Any]] = []
    if filters.get("wilayah"):
        nakes_must.append({"terms": {"nama_kabupaten_kota": filters["wilayah"]}})
    yr: Dict[str, Any] = {}
    if filters.get("date_from"):
        yr["gte"] = str(filters["date_from"])[:4]
    if filters.get("date_to"):
        yr["lte"] = str(filters["date_to"])[:4]
    if yr:
        nakes_must.append({"range": {"tahun": yr}})
    nakes_body = {"query": {"bool": {"must": nakes_must}} if nakes_must else {"match_all": {}}, "size": 0,
                  "aggs": {
                      "total_nakes": {"sum": {"field": "jumlah_nakes_gizi"}},
                      "nakes_by_region": {"terms": {"field": "nama_kabupaten_kota", "size": 100},
                                            "aggs": {"sum_nakes_in_bucket": {"sum": {"field": "jumlah_nakes_gizi"}}}},
                  }}

    stunting_data = _es_post(STUNTING_INDEX, "/_search", stunting_body)
    nakes_data = _es_post(NUTRITION_INDEX, "/_search", nakes_body)

    s_agg = stunting_data.get("aggregations", {})
    n_agg = nakes_data.get("aggregations", {})

    total_lahir = stunting_data.get("hits", {}).get("total", {}).get("value", 0)
    total_stunting = s_agg.get("stunting_count", {}).get("doc_count", 0)

    # imunisasi coverage
    imun_lengkap = s_agg.get("imunisasi_lengkap", {}).get("doc_count", 0)
    tot1 = s_agg.get("total_imunisasi_field_1", {}).get("value", 0)
    tot2 = s_agg.get("total_imunisasi_field_2", {}).get("value", 0)
    imun_total = (tot1 or 0) + (tot2 or 0)
    imun_cov_pct = (imun_lengkap / imun_total * 100.0) if imun_total else 0.0

    air_buckets = s_agg.get("air_bersih_dist", {}).get("buckets", [])
    air_layak_count = sum(b["doc_count"] for b in air_buckets if b["key"] in ["Layak", "Ya", "Bersih", "Aman"])
    air_total = sum(b["doc_count"] for b in air_buckets)
    air_cov_pct = (air_layak_count / air_total * 100.0) if air_total else 0.0

    nakes_buckets = n_agg.get("nakes_by_region", {}).get("buckets", [])
    if nakes_buckets:
        nakes_grouped = (
            pd.DataFrame([
                {"region": b["key"], "jumlah_nakes": b["sum_nakes_in_bucket"]["value"]}
                for b in nakes_buckets
            ]).set_index("region")["jumlah_nakes"].sort_values(ascending=False)
        )
    else:
        nakes_grouped = pd.Series([], dtype="float64", name="jumlah_nakes")

    imun_trend_rows: List[Dict[str, Any]] = []
    for b in s_agg.get("imunisasi_trend", {}).get("buckets", []):
        total_in_bucket = b["doc_count"]
        lengkap_in_bucket = b["imunisasi_lengkap_in_bucket"]["doc_count"]
        imun_trend_rows.append({
            "tanggal": pd.to_datetime(b["key_as_string"]),
            "imunisasi_lengkap": (lengkap_in_bucket / total_in_bucket) if total_in_bucket > 0 else 0,
        })
    imunisasi_per_bulan = pd.DataFrame(imun_trend_rows)

    air_layak_data = pd.Series({"Layak": air_layak_count, "Tidak Layak": max(0, air_total - air_layak_count)})

    return {
        "kpi": {
            "total_bayi_lahir": total_lahir,
            "total_bayi_stunting": total_stunting,
            "jumlah_nakes": n_agg.get("total_nakes", {}).get("value", 0),
            "cakupan_imunisasi_pct": imun_cov_pct,
            "akses_air_layak_pct": air_cov_pct,
        },
        "charts": {
            "nakes_by_region": nakes_grouped,
            "imunisasi_trend": imunisasi_per_bulan,
            "air_distribusi": air_layak_data,
        },
    }


# ------------------- Correlation trend (mirror utils/es.py) -------------------

def get_monthly_trend(filters: Dict[str, Any]) -> pd.DataFrame:
    body = build_query(filters)
    body.update({
        "size": 0,
        "aggs": {
            "per_month": {
                "date_histogram": {"field": "Tanggal", "calendar_interval": "month", "format": "yyyy-MM"},
                "aggs": {
                    "stunting_any": {"filter": _stunting_any_filter()},
                    "total_in_month": {"filter": {"match_all": {}}},
                },
            }
        },
    })
    res = _es_post(STUNTING_INDEX, "/_search", body)
    rows: List[Dict[str, Any]] = []
    for b in res["aggregations"]["per_month"]["buckets"]:
        total = b["total_in_month"]["doc_count"]
        stunting = b["stunting_any"]["doc_count"]
        percent = (stunting / total * 100) if total > 0 else 0
        rows.append({"Bulan": b["key_as_string"], "Stunting %": round(percent, 2)})
    return pd.DataFrame(rows).set_index("Bulan")


# ------------------- Numeric sample for correlation -------------------

def get_numeric_sample_for_corr(filters: Dict[str, Any], size: int = 5000) -> pd.DataFrame:
    body = build_query(filters)
    body.update({"size": size})
    data = _es_post(STUNTING_INDEX, "/_search", body)
    hits = data.get("hits", {}).get("hits", [])
    df_sample = pd.DataFrame([h.get("_source", {}) for h in hits])
    if df_sample.empty:
        return pd.DataFrame()
    return df_sample.select_dtypes(include=["number"]).copy()


# ------------------- Explorer Data -------------------

def _apply_advanced_filters_to_query(body: dict, advanced_filters: dict) -> dict:
    has_advanced = any(
        (isinstance(advanced_filters.get(k), list) and advanced_filters.get(k))
        or (
            not isinstance(advanced_filters.get(k), list)
            and advanced_filters.get(k)
            and advanced_filters.get(k) != "Semua"
        )
        for k in advanced_filters
    )
    if not has_advanced:
        return body

    if "match_all" in body["query"]:
        body["query"] = {"bool": {"must": []}}

    must = body["query"]["bool"].setdefault("must", [])

    if advanced_filters.get("pendidikan_ibu"):
        must.append({"terms": {"Pendidikan Ibu": advanced_filters["pendidikan_ibu"]}})

    if advanced_filters.get("asi_eksklusif") != "Semua":
        val = (["Ya", "ya", "True", "true", "1"] if advanced_filters["asi_eksklusif"] == "Ya"
               else ["Tidak", "tidak", "False", "false", "0"])
        must.append({"bool": {"should": [
            {"terms": {"ASI Eksklusif": val}},
            {"terms": {"ASI Eksklusif (ya/tidak)": val}},
        ], "minimum_should_match": 1}})

    if advanced_filters.get("akses_air") != "Semua":
        val = (["Layak", "Ada", "Ya", "Bersih", "Aman"] if advanced_filters["akses_air"] == "Ada"
               else ["Tidak Layak", "Tidak", "Tidak Ada"])
        must.append({"bool": {"should": [
            {"terms": {"Akses Air": val}},
            {"terms": {"Akses Air Bersih": val}},
        ], "minimum_should_match": 1}})

    return body


def get_explorer_data(filters: dict, advanced_filters: dict, size: int = 1000) -> pd.DataFrame:
    body = build_query(filters)
    body = _apply_advanced_filters_to_query(body, advanced_filters)

    source_fields = [
        "Tanggal",
        "nama_kabupaten_kota",
        "Kecamatan",
        "Status Stunting (Biner)",
        "Z-Score TB/U",
        "Usia Anak (bulan)",
        "Berat Lahir (gram)",
        "ASI Eksklusif",
        "Status Imunisasi Anak",
        "Pendidikan Ibu",
        "Akses Air Bersih",
    ]

    body["_source"] = source_fields
    body["size"] = size
    body["sort"] = [{"Z-Score TB/U": "asc"}]

    data = _es_post(STUNTING_INDEX, "/_search", body)
    hits = data.get("hits", {}).get("hits", [])
    df = pd.DataFrame([h.get("_source", {}) for h in hits])

    if not df.empty:
        df = df.rename(columns={
            "nama_kabupaten_kota": "Kabupaten/Kota",
            "Status Stunting (Biner)": "Status Stunting",
            "Z-Score TB/U": "Z-Score",
            "Status Imunisasi Anak": "Imunisasi",
        })
    return df


def get_top_counts_for_explorer_chart(filters: dict, advanced_filters: dict) -> pd.DataFrame:
    """Chart berjenjang 5 besar; jika ada filters['wilayah'] -> agregasi kecamatan, else kabupaten."""
    if filters.get("wilayah"):
        agg_field, level_label = "Kecamatan", "Kecamatan"
    else:
        agg_field, level_label = "nama_kabupaten_kota", "Kabupaten/Kota"

    body = build_query(filters)
    body = _apply_advanced_filters_to_query(body, advanced_filters)
    body["size"] = 0
    body["aggs"] = {"counts_by_region": {"terms": {"field": agg_field, "size": 5}}}

    data = _es_post(STUNTING_INDEX, "/_search", body)
    buckets = data.get("aggregations", {}).get("counts_by_region", {}).get("buckets", [])

    if not buckets:
        return pd.DataFrame(columns=[level_label, "Jumlah Data"])

    df = pd.DataFrame(buckets).rename(columns={"key": level_label, "doc_count": "Jumlah Data"})
    return df


# ------------------- Ekspor data explorer -------------------

def get_explorer_data_for_export(filters: dict, advanced_filters: dict, size: int = 5000) -> pd.DataFrame:
    body = build_query(filters)
    body = _apply_advanced_filters_to_query(body, advanced_filters)
    source_fields = [
        "Tanggal",
        "nama_kabupaten_kota",
        "Kecamatan",
        "Status Stunting (Biner)",
        "Z-Score TB/U",
        "Probabilitas Stunting (simulasi)",
        "Usia Anak (bulan)",
        "Berat Lahir (gram)",
        "ASI Eksklusif",
        "Status Imunisasi Anak",
        "Pendidikan Ibu",
        "Akses Air Bersih",
        "Kepesertaan Program Bantuan",
        "Upah Keluarga (Rp/bulan)",
        "Jumlah Anak",
        "Tinggi Badan Ibu (cm)",
        "BMI Pra-Hamil",
        "Hb (g/dL)",
        "LiLA saat Hamil (cm)",
        "Kunjungan ANC (x)",
        "Paparan Asap Rokok",
        "Jenis Pekerjaan Orang Tua",
    ]
    body["_source"] = source_fields
    body["size"] = size
    body["sort"] = [{"Z-Score TB/U": "asc"}]

    data = _es_post(STUNTING_INDEX, "/_search", body)
    hits = data.get("hits", {}).get("hits", [])
    return pd.DataFrame([h.get("_source", {}) for h in hits])


# ------------------- Risk Map (kabupaten & kecamatan) -------------------

def get_risk_map_data(filters: dict) -> pd.DataFrame:
    body = build_query(filters)
    body["size"] = 0
    body["aggs"] = {
        "by_kab": {
            "terms": {"field": "nama_kabupaten_kota", "size": 100},
            "aggs": {
                "by_kec": {
                    "terms": {"field": "Kecamatan", "size": 5000},
                    "aggs": {"stunting_count": {"filter": _stunting_any_filter()}}
                }
            }
        }
    }

    data = _es_post(STUNTING_INDEX, "/_search", body)

    rows: List[Dict[str, Any]] = []
    kab_buckets = data.get("aggregations", {}).get("by_kab", {}).get("buckets", [])
    for kab_b in kab_buckets:
        kab_name = kab_b["key"]
        for kec_b in kab_b.get("by_kec", {}).get("buckets", []):
            rows.append({
                "kabupaten": kab_name,
                "kecamatan": kec_b["key"],
                "total_anak": kec_b["doc_count"],
                "jumlah_stunting": kec_b["stunting_count"]["doc_count"],
            })

    return pd.DataFrame(rows)
