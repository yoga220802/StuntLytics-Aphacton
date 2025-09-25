import streamlit as st
import pandas as pd
import json
import pathlib
import pydeck as pdk
import math

from src import styles
from src import elastic_client as es
from src.components import sidebar

# --- Konfigurasi & Fungsi Helper ---
GEOJSON_PATH = pathlib.Path(__file__).parents[1] / "geojson" / "jawa-barat.geojson"


@st.cache_data(show_spinner="Memuat data GeoJSON...")
def load_geojson():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(v: str) -> str:
    if not v:
        return ""
    s = v.upper().strip()
    prefixes_to_remove = ["KABUPATEN", "KOTA", "KAB.", "KEC.", "KEC"]
    for prefix in prefixes_to_remove:
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    return " ".join(s.split())


def _prevalence_to_color(prevalence: float):
    if prevalence is None or pd.isna(prevalence):
        return [200, 200, 200, 80]
    score = max(0.0, min(1.0, prevalence / 100.0))
    if score < 0.5:
        t = score * 2.0
        r, g, b = int(t * 255), int(t * 255), int(255 * (1 - t))
    else:
        t = (score - 0.5) * 2.0
        r, g, b = 255, int(255 * (1 - t)), 0
    return [r, g, b, 180]


def _enrich_geojson(geojson: dict, agg_df: pd.DataFrame):
    if not agg_df.empty:
        agg_df["kab_key"] = agg_df["kabupaten"].apply(_normalize_name)
        agg_df["kec_key"] = agg_df["kecamatan"].apply(_normalize_name)
        lookup = {(r.kab_key, r.kec_key): r for r in agg_df.itertuples()}
    else:
        lookup = {}

    for feature in geojson.get("features", []):
        prop = feature.get("properties", {})
        kab_key = _normalize_name(prop.get("KABKOT", ""))
        kec_key = _normalize_name(prop.get("KECAMATAN", ""))

        rec = lookup.get((kab_key, kec_key))
        if rec and rec.total_anak > 0:
            prevalence = (rec.jumlah_stunting / rec.total_anak) * 100
            prop.update(
                {
                    "prevalensi_stunting": round(prevalence, 2),
                    "jumlah_stunting": int(rec.jumlah_stunting),
                    "total_anak_terdata": int(rec.total_anak),
                }
            )
            prop["fill_color"] = _prevalence_to_color(prevalence)
        else:
            prop.update(
                {
                    "prevalensi_stunting": "N/A",
                    "jumlah_stunting": 0,
                    "total_anak_terdata": 0,
                }
            )
            prop["fill_color"] = _prevalence_to_color(None)
        feature["properties"] = prop
    return geojson


# --- HELPER BARU UNTUK FOKUS PETA ---
def filter_geojson_features(geojson, selected_kab, selected_kec):
    if not selected_kab and not selected_kec:
        return geojson["features"]

    filtered_features = []
    norm_kab = _normalize_name(selected_kab[0]) if selected_kab else None
    norm_kec = _normalize_name(selected_kec[0]) if selected_kec else None

    for feature in geojson["features"]:
        prop = feature["properties"]
        kab_key = _normalize_name(prop.get("KABKOT", ""))
        kec_key = _normalize_name(prop.get("KECAMATAN", ""))

        if norm_kec:  # Filter paling spesifik dulu
            if kab_key == norm_kab and kec_key == norm_kec:
                filtered_features.append(feature)
        elif norm_kab:
            if kab_key == norm_kab:
                filtered_features.append(feature)
    return filtered_features


def _walk_coords(coords, lats, lons):
    if isinstance(coords, (list, tuple)):
        if len(coords) > 0 and isinstance(coords[0], (int, float)):
            lon, lat = coords
            lats.append(lat)
            lons.append(lon)
        else:
            for c in coords:
                _walk_coords(c, lats, lons)


def compute_view_state(features):
    if not features:
        return pdk.ViewState(latitude=-6.91, longitude=107.61, zoom=7.5, pitch=0)

    lats, lons = [], []
    for f in features:
        _walk_coords(f["geometry"]["coordinates"], lats, lons)

    if not lats or not lons:
        return pdk.ViewState(latitude=-6.91, longitude=107.61, zoom=7.5, pitch=0)

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    lat_span = max(abs(max_lat - min_lat), 1e-5)
    lon_span = max(abs(max_lon - min_lon), 1e-5)

    zoom_lon = math.log2(360 / lon_span)
    zoom_lat = math.log2(180 / lat_span)
    zoom = min(zoom_lon, zoom_lat) * 0.95  # Sedikit zoom out

    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)


# --- RENDER HALAMAN ---
def render_page():
    st.subheader("Peta Risiko Stunting Jawa Barat")
    st.caption(
        "Peta diwarnai berdasarkan Tingkat Prevalensi Stunting (jumlah kasus / total anak) per kecamatan."
    )

    main_filters = sidebar.render()

    try:
        agg_df = es.get_risk_map_data(main_filters)
        geojson_data = load_geojson()
        enriched_geojson = _enrich_geojson(geojson_data, agg_df)

        # Logika BARU: filter fitur dan hitung view state
        features_to_display = filter_geojson_features(
            enriched_geojson, main_filters["wilayah"], main_filters["kecamatan"]
        )
        view_state = compute_view_state(features_to_display)

        display_geojson = {"type": "FeatureCollection", "features": features_to_display}

        # ---- Perubahan: gunakan JS accessor untuk mengambil fill_color dari properties ----
        layer = pdk.Layer(
            "GeoJsonLayer",
            display_geojson,
            opacity=0.8,
            stroked=True,
            filled=True,
            get_fill_color="[properties.fill_color[0]*1, properties.fill_color[1]*1, properties.fill_color[2]*1, properties.fill_color[3]*1]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
            # Paksa re-evaluasi bila source data berubah
            update_triggers={"get_fill_color": display_geojson},
        )

        tooltip_html = """
        <div style="background-color: #333; color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;">
            <h4 style="margin: 0 0 5px 0;">{KABKOT}</h4>
            <h5 style="margin: 0 0 10px 0;">Kec. {KECAMATAN}</h5>
            <p style="margin: 0;"><strong>Tingkat Prevalensi:</strong> {prevalensi_stunting}%</p>
            <p style="margin: 0;"><strong>Kasus Stunting:</strong> {jumlah_stunting}</p>
            <p style="margin: 0;"><strong>Total Anak Terdata:</strong> {total_anak_terdata}</p>
        </div>
        """

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v9",
            tooltip={"html": tooltip_html},
        )

        st.pydeck_chart(r, use_container_width=True)

        st.markdown(
            """
        <div style="margin-top: 10px;">
            <b>Legenda Prevalensi Stunting (%)</b><br>
            <div style="display:flex; align-items:center;">
                <div style="width:100%; height:15px; background:linear-gradient(90deg, #00F 0%, #FF0 50%, #F00 100%); border:1px solid #FFF;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px;">
                <span>0% (Rendah)</span>
                <span>50%</span>
                <span>100% (Tinggi)</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Gagal membuat peta risiko: {e}")


# --- Main Execution ---
if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.page_config_set = True
styles.load_css()
render_page()
