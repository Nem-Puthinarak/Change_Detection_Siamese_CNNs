"""
SENTINEL WATCH — Cambodia-Thailand Border Monitoring Dashboard
Run with: streamlit run dashboard.py
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Watch",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    :root {
        --bg:      #0D1B2A;
        --surface: #112236;
        --card:    #162840;
        --border:  #1E3A5F;
        --accent:  #00D4AA;
        --red:     #FF4757;
        --yellow:  #FFD32A;
        --green:   #2ED573;
        --text:    #E8F4FD;
        --muted:   #6B8CAE;
    }

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp { background-color: var(--bg) !important; }

    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    .stSelectbox > div, .stMultiSelect > div {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .method-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-left: 6px;
    }

    .badge-ai     { background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid #00D4AA; }
    .badge-ndvi   { background: rgba(255,211,42,0.15); color: #FFD32A; border: 1px solid #FFD32A; }
    .badge-fire   { background: rgba(255,71,87,0.2);  color: #FF4757; border: 1px solid #FF4757; }
    .badge-clear  { background: rgba(46,213,115,0.2); color: #2ED573; border: 1px solid #2ED573; }
    .badge-watch  { background: rgba(255,211,42,0.2); color: #FFD32A; border: 1px solid #FFD32A; }
    .badge-nodata { background: rgba(107,140,174,0.2);color: #6B8CAE; border: 1px solid #6B8CAE; }

    .hotspot-row {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: border-color 0.2s;
    }

    .hotspot-row:hover { border-color: var(--accent); }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .hero-sub {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 4px;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    hr { border-color: var(--border) !important; }

    .stMarkdown h3 {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────
BG = '#0D1B2A'

HOTSPOTS = {
    "hotspot_01_osmach_west":        {"lat": 14.430, "lon": 103.655, "label": "Osmach West"},
    "hotspot_02_o3chean_village":    {"lat": 13.788, "lon": 102.743, "label": "O3chean Village"},
    "hotspot_03_jorkjei_village":    {"lat": 13.766, "lon": 102.703, "label": "Jorkjei Village"},
    "hotspot_04_poipet_thai":        {"lat": 13.671, "lon": 102.553, "label": "Poipet Thai"},
    "hotspot_05_poipet_road":        {"lat": 13.663, "lon": 102.713, "label": "Poipet Road"},
    "hotspot_06_osmach_east":        {"lat": 14.427, "lon": 103.708, "label": "Osmach East"},
    "hotspot_07_choam_chom_sngan":   {"lat": 14.345, "lon": 104.063, "label": "Choam Chom Sngan"},
    "hotspot_08_bos_sbov":           {"lat": 14.429, "lon": 104.753, "label": "Bos Sbov"},
    "hotspot_09_chok_chey":          {"lat": 13.763, "lon": 102.693, "label": "Chok Chey"},
    "hotspot_10_osmach_jan2026":     {"lat": 14.423, "lon": 103.670, "label": "Osmach Jan2026"},
    "hotspot_11_an_ses":             {"lat": 14.389, "lon": 104.965, "label": "An Ses"},
    "hotspot_12_ta_moan_thom":       {"lat": 14.348, "lon": 103.266, "label": "Ta Moan Thom"},
    "hotspot_13_ta_krabei":          {"lat": 14.358, "lon": 103.374, "label": "Ta Krabei"},
    "hotspot_14_preahvihear_thai":   {"lat": 14.399, "lon": 104.680, "label": "Preah Vihear Thai"},
    "hotspot_15_preahvihear_temple": {"lat": 14.392, "lon": 104.680, "label": "Preah Vihear Temple"},
    "hotspot_16_thmor_da":           {"lat": 12.132, "lon": 102.735, "label": "Thmor Da"},
    "hotspot_17_prei_chan":          {"lat": 13.818, "lon": 102.737, "label": "Prei Chan"},
}

STATIC_PERIODS = [
    "Dec2025_conflict",
    "Jan2026_persistent",
    "Feb2026_baseline",
    "Mar2026_baseline",
    "Apr2026_baseline",
    "Mar2026_fire",
    "Apr2026_fire",
]

PERIOD_LABELS = {
    "Dec2025_conflict":   "Dec 2025",
    "Jan2026_persistent": "Jan 2026",
    "Feb2026_baseline":   "Feb 2026",
    "Mar2026_baseline":   "Mar 2026",
    "Apr2026_baseline":   "Apr 2026",
    "Mar2026_fire":       "Mar 2026 🔥",
    "Apr2026_fire":       "Apr 2026 🔥",
}

PERIOD_LABELS_MPL = {
    "Dec2025_conflict":   "Dec 2025",
    "Jan2026_persistent": "Jan 2026",
    "Feb2026_baseline":   "Feb 2026",
    "Mar2026_baseline":   "Mar 2026",
    "Apr2026_baseline":   "Apr 2026",
    "Apr2026_fire":       "Apr 2026 [FIRE]",
    "Mar2026_fire":       "Mar 2026 [FIRE]",
}

STATIC_PERIOD_MAP = {
    'Dec2025_conflict':   ('20251130', '20251201', '20251231'),
    'Jan2026_persistent': ('20251215', '20260120', '20260315'),
    'Feb2026_baseline':   ('20260131', '20260201', '20260228'),
    'Mar2026_baseline':   ('20260228', '20260301', '20260331'),
    'Apr2026_baseline':   ('20260331', '20260401', '20260425'),
    'Apr2026_fire':       ('20260410', '20260413', '20260425'),
    'Mar2026_fire':       ('20260323', '20260326', '20260415'),
}

# ── HELPERS ───────────────────────────────────────────────
def period_display_label(p):
    if p in PERIOD_LABELS:
        return PERIOD_LABELS[p]
    return p.replace("_auto", " (auto)").replace("_", " ")

def period_display_label_mpl(p):
    if p in PERIOD_LABELS_MPL:
        return PERIOD_LABELS_MPL[p]
    return p.replace("_auto", " (auto)").replace("_", " ")

def method_badge_html(method):
    """Return a coloured HTML badge showing AI model vs dNDVI."""
    if not method or method == "unknown" or method == "baseline_only":
        return ""
    if "AI" in method or "model" in method.lower():
        return '<span class="method-badge badge-ai">AI MODEL</span>'
    return '<span class="method-badge badge-ndvi">dNDVI/dNBR</span>'

# ── DATA LOADING ──────────────────────────────────────────
@st.cache_data
def load_rgb_image(path):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")

@st.cache_data
def load_npy(path):
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)

def load_meta(meta_path):
    """Load _meta.json alongside an RGB file. Returns {} if missing."""
    if not meta_path or not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def scan_available_data(rgb_dir, npy_dir):
    """
    Scan RGB images + meta JSONs + NPY files.

    For auto periods, meta JSON supplies exact before/after dates so we
    can look up the correct NPY files instead of guessing by order.
    """
    available = {}

    # ── Build NPY index: hotspot → sorted [(date_str, path)] ─
    npy_by_hotspot = {}
    if os.path.exists(npy_dir):
        for f in sorted(os.listdir(npy_dir)):
            if not f.endswith('.npy'):
                continue
            parts = f.replace('_stack.npy', '').split('_')
            try:
                date_idx = next(
                    i for i, p in enumerate(parts)
                    if p.isdigit() and len(p) == 8
                )
            except StopIteration:
                continue
            name = '_'.join(parts[:date_idx])
            date = parts[date_idx]
            npy_by_hotspot.setdefault(name, [])
            npy_by_hotspot[name].append((date, os.path.join(npy_dir, f)))
    for k in npy_by_hotspot:
        npy_by_hotspot[k].sort(key=lambda x: x[0])

    # ── Scan RGB + meta files ──────────────────────────────
    if os.path.exists(rgb_dir):
        for f in sorted(os.listdir(rgb_dir)):
            if not f.endswith('_rgb.png'):
                continue
            base = f.replace('_rgb.png', '')

            matched_hotspot = None
            matched_period  = None
            for hk in HOTSPOTS:
                if base.startswith(hk + "_"):
                    matched_hotspot = hk
                    matched_period  = base[len(hk) + 1:]
                    break
            if matched_hotspot is None:
                continue

            rgb_path  = os.path.join(rgb_dir, f)
            meta_path = os.path.join(rgb_dir, f"{base}_meta.json")
            meta      = load_meta(meta_path)

            before_npy = None
            after_npy  = None
            dates      = npy_by_hotspot.get(matched_hotspot, [])

            # ── Auto period: use dates from meta JSON ──────
            if matched_period.endswith("_auto") and meta:
                before_date = (meta.get("before_date") or "").replace("-", "")
                after_date  = (meta.get("after_date")  or "").replace("-", "")
                if before_date:
                    found = [p for d, p in dates if d == before_date]
                    before_npy = found[0] if found else None
                if after_date:
                    found = [p for d, p in dates if d == after_date]
                    after_npy = found[0] if found else None
                # fallback: pair last two if meta dates not matched
                if not before_npy and not after_npy and len(dates) >= 2:
                    before_npy = dates[-2][1]
                    after_npy  = dates[-1][1]

            # ── Static period: use date windows ───────────
            elif matched_period in STATIC_PERIOD_MAP:
                before_end, after_start, after_end = STATIC_PERIOD_MAP[matched_period]
                before_list = [(d, p) for d, p in dates if d <= before_end]
                after_list  = [(d, p) for d, p in dates if after_start <= d <= after_end]
                if before_list:
                    before_npy = before_list[-1][1]
                if after_list:
                    after_npy  = after_list[0][1]

            available.setdefault(matched_hotspot, {"periods": {}})
            available[matched_hotspot]["periods"][matched_period] = {
                "rgb":        rgb_path,
                "meta":       meta,
                "before_npy": before_npy,
                "after_npy":  after_npy,
            }

    return available


def get_all_periods(available):
    periods = set()
    for hdata in available.values():
        periods.update(hdata["periods"].keys())

    def period_sort_key(p):
        if p in STATIC_PERIODS:
            return (0, STATIC_PERIODS.index(p))
        return (1, p)

    return sorted(periods, key=period_sort_key)


def compute_ndvi(arr):
    b08   = arr[:, :, 3].astype(np.float32)
    b04   = arr[:, :, 2].astype(np.float32)
    denom = b08 + b04
    denom[denom == 0] = 1e-6
    return (b08 - b04) / denom


def compute_dnvi_stats(before_npy, after_npy):
    before = load_npy(before_npy)
    after  = load_npy(after_npy)
    if before is None or after is None:
        return None
    h = min(before.shape[0], after.shape[0])
    w = min(before.shape[1], after.shape[1])
    before, after = before[:h, :w], after[:h, :w]

    ndvi_b = compute_ndvi(before)
    ndvi_a = compute_ndvi(after)
    dndvi  = ndvi_b - ndvi_a
    mask   = dndvi > 0.15

    cleared_km2 = float(np.sum(mask)) * 100 / 1e6
    pct_changed = float(np.mean(mask)) * 100
    return {
        "dndvi":            dndvi,
        "mask":             mask,
        "cleared_km2":      cleared_km2,
        "pct_changed":      pct_changed,
        "mean_ndvi_before": float(np.mean(ndvi_b)),
        "mean_ndvi_after":  float(np.mean(ndvi_a)),
    }


def get_status(period, stats):
    if stats is None:
        return "nodata"
    if "fire" in period.lower():
        return "fire" if stats["cleared_km2"] > 0.01 else "watch"
    if stats["cleared_km2"] > 0.5:
        return "watch"
    return "clear"


STATUS_COLORS = {
    "fire":   "#FF4757",
    "watch":  "#FFD32A",
    "clear":  "#2ED573",
    "nodata": "#6B8CAE",
}

STATUS_LABELS = {
    "fire":   "🔥 ACTIVE",
    "watch":  "⚠️ WATCH",
    "clear":  "✅ CLEAR",
    "nodata": "— NO DATA",
}

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 0.65rem;
                    color: #00D4AA; letter-spacing: 0.2em; text-transform: uppercase;'>
            Cambodia · Thailand
        </div>
        <div style='font-size: 1.4rem; font-weight: 800; margin-top: 4px;'>
            Sentinel Watch
        </div>
        <div style='font-size: 0.75rem; color: #6B8CAE; margin-top: 2px;'>
            Border Deforestation Monitor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Paths</div>', unsafe_allow_html=True)

    rgb_dir = st.text_input(
        "RGB directory",
        value="sentinel_display/truecolor",
    )
    npy_dir = st.text_input(
        "NPY directory",
        value="sentinel_display/npy_flat",
    )

    st.markdown('<div class="section-title" style="margin-top:16px">Filters</div>',
                unsafe_allow_html=True)

    period_filter_placeholder = st.empty()
    show_status = st.multiselect(
        "Show status",
        options=["fire", "watch", "clear", "nodata"],
        default=["fire", "watch", "clear", "nodata"],
        format_func=lambda x: STATUS_LABELS[x]
    )

    st.markdown("---")

    # ── Pipeline Runner ───────────────────────────────────
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)

    AUTO_SCRIPT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel_auto.py")
    script_exists = os.path.exists(AUTO_SCRIPT)

    if not script_exists:
        st.warning("sentinel_auto.py not found in same folder as dashboard.py")
    else:
        last_log = sorted(glob.glob("sentinel_display/auto/run_*.json"), reverse=True)
        if last_log:
            try:
                with open(last_log[0]) as f:
                    last = json.load(f)
                last_time  = last.get("run_time", "")[:16].replace("T", " ")
                n_alerts   = len(last.get("alerts", []))
                model_on   = last.get("model_used", False)
                model_str  = "AI model ON" if model_on else "dNDVI fallback"
                model_col  = "#00D4AA" if model_on else "#FFD32A"
                st.markdown(f"""
                <div style='font-size: 0.7rem; color: #6B8CAE; margin-bottom: 8px;'>
                    Last run: <span style='color: #00D4AA;'>{last_time}</span><br>
                    Alerts: <span style='color: {"#FF4757" if n_alerts else "#2ED573"};'>{n_alerts}</span><br>
                    Method: <span style='color: {model_col};'>{model_str}</span>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

        run_clicked = st.button("▶ Run Pipeline Now", use_container_width=True, type="primary")
        if run_clicked:
            st.session_state["pipeline_running"] = True

        if st.session_state.get("pipeline_running"):
            log_box = st.empty()
            status  = st.empty()
            status.info("Pipeline running...")
            try:
                proc = subprocess.Popen(
                    [sys.executable, AUTO_SCRIPT],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                lines = []
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    log_box.code("\n".join(lines[-30:]), language=None)
                proc.wait()
                if proc.returncode == 0:
                    status.success("Pipeline complete!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    status.error(f"Pipeline failed (exit {proc.returncode})")
            except Exception as e:
                status.error(f"Error: {e}")
            finally:
                st.session_state["pipeline_running"] = False

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size: 0.7rem; color: #6B8CAE;'>
        <div>Model: SiameseUNet12</div>
        <div>Threshold: dNDVI > 0.15</div>
        <div style='margin-top: 8px; color: #00D4AA;'>
            Last updated: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 24px;'>
    <div class='hero-title'>🛰️ Border Watch</div>
    <div class='hero-sub'>
        Real-time deforestation monitoring · Cambodia–Thailand border · 17 hotspot locations
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────
with st.spinner("Scanning available data..."):
    available = scan_available_data(rgb_dir, npy_dir)

if not available:
    st.warning(f"""
    **No data found yet.**

    Run `sentinel_auto.py` at least once to populate:
    - RGB dir: `{rgb_dir}`
    - NPY dir: `{npy_dir}`
    """)
    AUTO_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel_auto.py")
    if os.path.exists(AUTO_SCRIPT):
        if st.button("▶ Run Pipeline Now", type="primary"):
            st.session_state["pipeline_running"] = True
            st.rerun()
    else:
        st.error("sentinel_auto.py not found.")
    st.stop()

# ── PERIOD FILTER ─────────────────────────────────────────
all_periods = get_all_periods(available)

with period_filter_placeholder:
    selected_period = st.selectbox(
        "Period",
        options=["All"] + all_periods,
        format_func=lambda x: "All Periods" if x == "All" else period_display_label(x)
    )

# ── COMPUTE SUMMARY STATS ─────────────────────────────────
@st.cache_data
def compute_all_stats(available_json, selected_period):
    available = json.loads(available_json)
    rows = []
    for hotspot, data in available.items():
        label = HOTSPOTS.get(hotspot, {}).get("label", hotspot)
        for period, pdata in data["periods"].items():
            if selected_period != "All" and period != selected_period:
                continue
            stats = None
            if pdata.get("before_npy") and pdata.get("after_npy"):
                stats = compute_dnvi_stats(pdata["before_npy"], pdata["after_npy"])
            status = get_status(period, stats)
            meta   = pdata.get("meta", {})
            rows.append({
                "hotspot":      hotspot,
                "label":        label,
                "period":       period,
                "status":       status,
                "cleared_km2":  stats["cleared_km2"] if stats else 0,
                "pct_changed":  stats["pct_changed"]  if stats else 0,
                "ndvi_before":  stats["mean_ndvi_before"] if stats else None,
                "ndvi_after":   stats["mean_ndvi_after"]  if stats else None,
                "rgb_path":     pdata.get("rgb"),
                "before_npy":   pdata.get("before_npy"),
                "after_npy":    pdata.get("after_npy"),
                # ── NEW: from meta JSON ────────────────────
                "method":       meta.get("method", ""),
                "model_used":   meta.get("model_used", False),
                "flagged":      meta.get("flagged", False),
                "before_date":  meta.get("before_date", ""),
                "after_date":   meta.get("after_date", ""),
            })
    return pd.DataFrame(rows)

df = compute_all_stats(json.dumps(available), selected_period)
df_filtered = df[df["status"].isin(show_status)] if not df.empty else df

# ── TOP METRICS ───────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_cleared = df["cleared_km2"].sum() if not df.empty else 0
active_fires  = int((df["status"] == "fire").sum())  if not df.empty else 0
watch_count   = int((df["status"] == "watch").sum()) if not df.empty else 0
clear_count   = int((df["status"] == "clear").sum()) if not df.empty else 0

with col1:
    st.metric("Total Cleared Area",
              f"{total_cleared:.2f} km²" if total_cleared < 1000 else f"{total_cleared:.1f} km²")
with col2:
    st.metric("🔥 Active Alerts", active_fires)
with col3:
    st.metric("⚠️ Watch", watch_count)
with col4:
    st.metric("✅ Clear", clear_count)
with col5:
    st.metric("Hotspots Monitored", len(HOTSPOTS))

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Hotspot Overview",
    "🔍 Detailed View",
    "📊 Timeline Analysis",
    "🤖 Live Monitor",
])

# ─────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown('<div class="section-title">Hotspot Status</div>', unsafe_allow_html=True)

        if df_filtered.empty:
            st.info("No hotspots match current filters.")
        else:
            STATUS_PRIORITY = {"fire": 0, "watch": 1, "clear": 2, "nodata": 3}
            summary = (
                df_filtered.groupby(["hotspot", "label"])
                .apply(lambda g: g.loc[g["status"].map(STATUS_PRIORITY).idxmin()])
                .reset_index(drop=True)
            )
            summary = summary.sort_values("status", key=lambda s: s.map(STATUS_PRIORITY))

            for _, row in summary.iterrows():
                color  = STATUS_COLORS[row["status"]]
                badge  = STATUS_LABELS[row["status"]]
                mbadge = method_badge_html(row.get("method", ""))
                st.markdown(f"""
                <div class="hotspot-row">
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <div style='font-weight: 600; font-size: 0.9rem;'>
                                {row['label']}{mbadge}
                            </div>
                            <div style='font-size: 0.7rem; color: #6B8CAE;
                                        font-family: Space Mono, monospace;'>
                                {row['hotspot']}
                            </div>
                        </div>
                        <div style='text-align: right;'>
                            <span class="status-badge badge-{row['status']}">{badge}</span>
                            <div style='font-family: Space Mono, monospace; font-size: 0.75rem;
                                        color: {color}; margin-top: 4px;'>
                                {row['cleared_km2']:.3f} km²
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Cleared Area by Hotspot</div>', unsafe_allow_html=True)

        chart_data = (
            df_filtered.groupby("label")["cleared_km2"]
            .sum()
            .sort_values(ascending=True)
        ) if not df_filtered.empty else pd.Series(dtype=float)

        if not chart_data.empty:
            fig, ax = plt.subplots(figsize=(7, max(3, len(chart_data) * 0.4)))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            STATUS_PRIORITY = {"fire": 0, "watch": 1, "clear": 2, "nodata": 3}
            status_map      = {0: "#FF4757", 1: "#FFD32A", 2: "#2ED573", 3: "#6B8CAE"}
            status_by_label = df_filtered.groupby("label")["status"].apply(
                lambda s: s.map(STATUS_PRIORITY).min()
            )
            colors_bar = [status_map.get(status_by_label.get(lbl, 3), "#6B8CAE")
                          for lbl in chart_data.index]

            bars = ax.barh(chart_data.index, chart_data.values, color=colors_bar, height=0.6)
            for bar, val in zip(bars, chart_data.values):
                if val > 0:
                    ax.text(bar.get_width() + 0.002,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center",
                            color="white", fontsize=7, fontfamily="monospace")

            ax.set_xlabel("Cleared area (km²)", color="#6B8CAE", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#1E3A5F")
            ax.set_xlim(0, chart_data.max() * 1.2 + 0.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-title" style="margin-top:16px">Summary Table</div>',
                    unsafe_allow_html=True)
        if not df_filtered.empty:
            display_df = df_filtered[
                ["label", "period", "status", "cleared_km2", "pct_changed", "method"]
            ].copy()
            display_df.columns = ["Location", "Period", "Status", "Cleared (km²)", "Changed (%)", "Method"]
            display_df["Period"]        = display_df["Period"].apply(period_display_label)
            display_df["Cleared (km²)"] = display_df["Cleared (km²)"].round(3)
            display_df["Changed (%)"]   = display_df["Changed (%)"].round(2)
            display_df["Method"]        = display_df["Method"].replace("", "—")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = display_df.to_csv(index=False)
            st.download_button(
                "⬇️ Export CSV",
                data=csv,
                file_name=f"sentinel_watch_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────────────────
# TAB 2 — DETAILED VIEW
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Hotspot Inspector</div>', unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        hotspot_options  = {v["label"]: k for k, v in HOTSPOTS.items() if k in available}
        selected_label   = st.selectbox("Select hotspot", options=list(hotspot_options.keys()))
        selected_hotspot = hotspot_options[selected_label]

    with col_sel2:
        periods_available = list(available.get(selected_hotspot, {}).get("periods", {}).keys())
        selected_period_detail = st.selectbox(
            "Select period",
            options=periods_available,
            format_func=period_display_label
        )

    if selected_hotspot and selected_period_detail:
        pdata  = available[selected_hotspot]["periods"].get(selected_period_detail, {})
        meta   = pdata.get("meta", {})
        method = meta.get("method", "")
        stats  = None

        if pdata.get("before_npy") and pdata.get("after_npy"):
            with st.spinner("Computing NDVI change..."):
                stats = compute_dnvi_stats(pdata["before_npy"], pdata["after_npy"])

        status = get_status(selected_period_detail, stats)
        color  = STATUS_COLORS[status]

        # ── Method indicator banner ────────────────────────
        if method and method not in ("unknown", "baseline_only"):
            is_ai   = "AI" in method or "model" in method.lower()
            m_color = "#00D4AA" if is_ai else "#FFD32A"
            m_label = "AI Model (SiameseUNet12)" if is_ai else "Spectral Index (dNDVI / dNBR)"
            m_icon  = "🤖" if is_ai else "📊"
            st.markdown(f"""
            <div style='background: rgba({int(m_color[1:3],16)},{int(m_color[3:5],16)},
                        {int(m_color[5:7],16)},0.08);
                        border: 1px solid {m_color}; border-radius: 6px;
                        padding: 8px 14px; margin-bottom: 12px;
                        font-family: Space Mono, monospace; font-size: 0.78rem; color: {m_color};'>
                {m_icon} Detection method: <strong>{m_label}</strong>
                {"&nbsp;&nbsp;·&nbsp;&nbsp;cross-confirmed by dNDVI" if is_ai and meta.get("flagged") else ""}
            </div>
            """, unsafe_allow_html=True)

        # Before/after date range from meta
        bd = meta.get("before_date", "")
        ad = meta.get("after_date",  "")
        date_range = f"{bd} → {ad}" if bd and ad else period_display_label(selected_period_detail)

        st.markdown(f"""
        <div style='background: rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1);
                    border: 1px solid {color}; border-radius: 8px; padding: 12px 16px;
                    margin-bottom: 16px; display: flex; align-items: center; gap: 12px;'>
            <div style='font-size: 1.5rem;'>{STATUS_LABELS[status].split()[0]}</div>
            <div>
                <div style='font-weight: 700; color: {color};'>{STATUS_LABELS[status]}</div>
                <div style='font-size: 0.8rem; color: #6B8CAE;'>
                    {selected_label} · {date_range}
                </div>
            </div>
            {'<div style="margin-left:auto; font-family: Space Mono, monospace; color: ' + color + '; font-size: 1.2rem; font-weight: 700;">' + f"{stats['cleared_km2']:.3f} km²</div>" if stats else ''}
        </div>
        """, unsafe_allow_html=True)

        if stats:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Cleared Area",  f"{stats['cleared_km2']:.3f} km²")
            with m2:
                st.metric("Area Changed",  f"{stats['pct_changed']:.2f}%")
            with m3:
                st.metric("NDVI Before",   f"{stats['mean_ndvi_before']:.3f}")
            with m4:
                delta = stats['mean_ndvi_after'] - stats['mean_ndvi_before']
                st.metric("NDVI After", f"{stats['mean_ndvi_after']:.3f}", delta=f"{delta:.3f}")

        rgb_img = load_rgb_image(pdata.get("rgb")) if pdata.get("rgb") else None
        if rgb_img is not None:
            st.markdown('<div class="section-title" style="margin-top:16px">True Color Imagery</div>',
                        unsafe_allow_html=True)
            w          = rgb_img.width // 2
            before_img = rgb_img.crop((0, 0, w, rgb_img.height))
            after_img  = rgb_img.crop((w, 0, rgb_img.width, rgb_img.height))
            ic1, ic2   = st.columns(2)
            cap_before = f"Before ({bd})" if bd else "Before"
            cap_after  = f"After ({ad})"  if ad else "After"
            with ic1:
                st.image(before_img, caption=cap_before, use_container_width=True)
            with ic2:
                st.image(after_img,  caption=cap_after,  use_container_width=True)

        if stats is not None:
            st.markdown('<div class="section-title" style="margin-top:16px">dNDVI Change Map</div>',
                        unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor(BG)

            im1 = axes[0].imshow(stats["dndvi"], cmap="RdYlGn_r", vmin=-0.3, vmax=0.3)
            axes[0].set_title("dNDVI (before − after)", color="white", fontsize=11)
            axes[0].axis("off")
            cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            cb1.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")

            axes[1].imshow(stats["mask"], cmap="RdYlGn_r")
            axes[1].set_title(
                f"Clearance mask (dNDVI > 0.15)\n{stats['cleared_km2']:.3f} km²",
                color="white", fontsize=11
            )
            axes[1].axis("off")

            for ax in axes:
                ax.set_facecolor(BG)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No NPY data available for this hotspot/period combination.")

# ─────────────────────────────────────────────────────────
# TAB 3 — TIMELINE
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Clearance Area Over Time</div>', unsafe_allow_html=True)

    hotspot_options_multi = {v["label"]: k for k, v in HOTSPOTS.items() if k in available}
    selected_compare = st.multiselect(
        "Select hotspots to compare",
        options=list(hotspot_options_multi.keys()),
        default=list(hotspot_options_multi.keys())[:5]
    )

    timeline_periods = get_all_periods(available)

    if selected_compare:
        timeline_data = []
        for label in selected_compare:
            hotspot = hotspot_options_multi[label]
            for period in timeline_periods:
                pdata = available.get(hotspot, {}).get("periods", {}).get(period)
                if not pdata:
                    continue
                cleared = 0
                if pdata.get("before_npy") and pdata.get("after_npy"):
                    s = compute_dnvi_stats(pdata["before_npy"], pdata["after_npy"])
                    if s:
                        cleared = s["cleared_km2"]
                timeline_data.append({
                    "label":       label,
                    "period":      period_display_label_mpl(period),
                    "cleared_km2": cleared,
                    "order":       timeline_periods.index(period),
                })

        tdf = pd.DataFrame(timeline_data).sort_values("order") if timeline_data else pd.DataFrame()

        if not tdf.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            palette = ["#00D4AA", "#FF4757", "#FFD32A", "#2ED573",
                       "#A29BFE", "#FD79A8", "#74B9FF", "#FDCB6E",
                       "#E17055", "#55EFC4", "#6C5CE7", "#B2BEC3"]

            for idx, label in enumerate(selected_compare):
                sub   = tdf[tdf["label"] == label]
                color = palette[idx % len(palette)]
                ax.plot(sub["period"], sub["cleared_km2"],
                        marker="o", linewidth=2, markersize=6,
                        label=label, color=color)

            ax.set_xlabel("Period", color="#6B8CAE", fontsize=9)
            ax.set_ylabel("Cleared area (km²)", color="#6B8CAE", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            ax.spines[:].set_color("#1E3A5F")
            ax.legend(facecolor="#112236", labelcolor="white", fontsize=8, framealpha=0.8)
            ax.grid(axis="y", color="#1E3A5F", linewidth=0.5, alpha=0.7)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown('<div class="section-title" style="margin-top:24px">Cumulative Clearance Heatmap</div>',
                unsafe_allow_html=True)

    if not df.empty:
        pivot = df.pivot_table(
            index="label", columns="period",
            values="cleared_km2", aggfunc="sum", fill_value=0
        )
        pivot = pivot.reindex(columns=[p for p in timeline_periods if p in pivot.columns])

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.4)))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(
                [period_display_label_mpl(p) for p in pivot.columns],
                rotation=30, ha="right", color="white", fontsize=8
            )
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, color="white", fontsize=8)

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if val > 0:
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="black" if val > pivot.values.max() * 0.5 else "white",
                                fontsize=7, fontfamily="monospace")

            cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
            cb.set_label("km²", color="white", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────
# TAB 4 — LIVE MONITOR
# ─────────────────────────────────────────────────────────
with tab4:
    AUTO_DIR   = "sentinel_display/auto"
    STATE_FILE = "sentinel_display/auto_state.json"

    def load_latest_run_log():
        if not os.path.exists(AUTO_DIR):
            return None
        logs = sorted(glob.glob(os.path.join(AUTO_DIR, "run_*.json")), reverse=True)
        if not logs:
            return None
        with open(logs[0]) as f:
            return json.load(f), logs[0]

    def load_auto_state():
        if not os.path.exists(STATE_FILE):
            return {}
        with open(STATE_FILE) as f:
            return json.load(f)

    log_result = load_latest_run_log()
    auto_state = load_auto_state()

    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown('<div class="section-title">Automated Pipeline Status</div>',
                    unsafe_allow_html=True)
    with col_h2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if log_result is None:
        st.info("No automated runs found yet. Click **▶ Run Pipeline Now** in the sidebar.")
    else:
        log, log_path = log_result
        run_time    = log.get("run_time", "unknown")
        alerts      = log.get("alerts", [])
        skipped     = log.get("skipped", [])
        model_used  = log.get("model_used", False)
        stac_col    = log.get("stac_collection", "—")
        bl_config   = log.get("baseline_config", {})

        model_color = "#00D4AA" if model_used else "#FFD32A"
        model_str   = "ON — SiameseUNet12" if model_used else "OFF — dNDVI/dNBR fallback"

        st.markdown(f"""
        <div style='background: #112236; border: 1px solid #1E3A5F; border-radius: 8px;
                    padding: 12px 16px; margin-bottom: 16px;
                    display: flex; gap: 32px; align-items: center; flex-wrap: wrap;'>
            <div>
                <div style='font-size: 0.65rem; color: #6B8CAE; text-transform: uppercase;
                            letter-spacing: 0.1em;'>Last Run</div>
                <div style='font-family: Space Mono, monospace; font-size: 0.9rem;
                            color: #E8F4FD;'>{run_time[:16].replace("T"," ")}</div>
            </div>
            <div>
                <div style='font-size: 0.65rem; color: #6B8CAE; text-transform: uppercase;
                            letter-spacing: 0.1em;'>AI Model</div>
                <div style='font-family: Space Mono, monospace; font-size: 0.85rem;
                            color: {model_color};'>{model_str}</div>
            </div>
            <div>
                <div style='font-size: 0.65rem; color: #6B8CAE; text-transform: uppercase;
                            letter-spacing: 0.1em;'>Alerts</div>
                <div style='font-family: Space Mono, monospace; font-size: 0.9rem;
                            color: {"#FF4757" if alerts else "#2ED573"};'>
                    {len(alerts)} hotspot{"s" if len(alerts)!=1 else ""}</div>
            </div>
            <div>
                <div style='font-size: 0.65rem; color: #6B8CAE; text-transform: uppercase;
                            letter-spacing: 0.1em;'>Skipped</div>
                <div style='font-family: Space Mono, monospace; font-size: 0.9rem;
                            color: #6B8CAE;'>{len(skipped)}</div>
            </div>
            <div>
                <div style='font-size: 0.65rem; color: #6B8CAE; text-transform: uppercase;
                            letter-spacing: 0.1em;'>Baseline Config</div>
                <div style='font-family: Space Mono, monospace; font-size: 0.75rem;
                            color: #6B8CAE;'>
                    month={bl_config.get("target_month","—")} · 
                    lookback={bl_config.get("lookback_days","—")}d
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if alerts:
            st.markdown('<div class="section-title">New Activity Detected</div>',
                        unsafe_allow_html=True)
            for a in alerts:
                label       = a["hotspot"].replace("hotspot_", "").replace("_", " ").title()
                model_km2   = a.get("model_km2", 0)
                dndvi_km2   = a.get("dndvi_km2", 0)
                dnbr_km2    = a.get("dnbr_km2",  0)
                primary     = a.get("primary", "dNDVI/dNBR")
                primary_km2 = a.get("primary_km2", max(dndvi_km2, dnbr_km2))

                is_ai_primary = "AI" in primary or "model" in primary.lower()
                p_color       = "#00D4AA" if is_ai_primary else "#FFD32A"
                p_icon        = "🤖" if is_ai_primary else "📊"

                result_img_path = os.path.join(
                    AUTO_DIR, "results", f"{a['hotspot']}_{a['latest']}.png"
                )

                with st.expander(
                    f"🔴  {label}  ·  {a['baseline']} → {a['latest']}  ·  {primary_km2:.4f} km²",
                    expanded=True
                ):
                    # Method banner per alert
                    st.markdown(f"""
                    <div style='background: rgba({int(p_color[1:3],16)},{int(p_color[3:5],16)},
                                {int(p_color[5:7],16)},0.08);
                                border: 1px solid {p_color}; border-radius: 6px;
                                padding: 6px 12px; margin-bottom: 10px;
                                font-family: Space Mono, monospace; font-size: 0.75rem;
                                color: {p_color};'>
                        {p_icon} Primary detection: <strong>{primary}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Primary", f"{primary_km2:.4f} km²",
                                  help=f"Source: {primary}")
                    with m2:
                        st.metric("AI Model",
                                  f"{model_km2:.4f} km²" if model_km2 > 0 else "N/A")
                    with m3:
                        st.metric("dNDVI Clearance", f"{dndvi_km2:.4f} km²")
                    with m4:
                        st.metric("dNBR Burn", f"{dnbr_km2:.4f} km²")

                    if os.path.exists(result_img_path):
                        st.image(result_img_path, use_container_width=True,
                                 caption=f"Before ({a['baseline']}) vs After ({a['latest']})")
                    else:
                        st.info("Result image not found.")
        else:
            if len(skipped) >= len(HOTSPOTS):
                st.error("All hotspots were skipped. Check credentials or run sentinel_auto.py in terminal.")
            else:
                st.success(
                    f"No new activity detected. "
                    f"({len(skipped)} skipped, {len(HOTSPOTS) - len(skipped)} checked)"
                )

        # ── Baseline state table ──────────────────────────
        st.markdown('<div class="section-title" style="margin-top:24px">Baseline State per Hotspot</div>',
                    unsafe_allow_html=True)

        if auto_state:
            state_rows = []
            for hotspot, info in auto_state.items():
                label = HOTSPOTS.get(hotspot, {}).get("label", hotspot)
                state_rows.append({
                    "Location":      label,
                    "Hotspot ID":    hotspot,
                    "Baseline Date": info.get("baseline_date", "—"),
                    "Baseline File": os.path.basename(info.get("baseline_npy", "—")),
                })
            st.dataframe(pd.DataFrame(state_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No baseline state yet.")

        # ── All result images browser ─────────────────────
        results_dir = os.path.join(AUTO_DIR, "results")
        if os.path.exists(results_dir):
            result_images = sorted(glob.glob(os.path.join(results_dir, "*.png")), reverse=True)
            if result_images:
                st.markdown('<div class="section-title" style="margin-top:24px">All Auto Run Results</div>',
                            unsafe_allow_html=True)

                img_by_hotspot = {}
                for img_path in result_images:
                    fname = os.path.basename(img_path)
                    for hk in HOTSPOTS:
                        if fname.startswith(hk):
                            img_by_hotspot.setdefault(hk, []).append(img_path)
                            break

                hotspot_choices = {HOTSPOTS[k]["label"]: k for k in img_by_hotspot}
                if hotspot_choices:
                    sel_label   = st.selectbox("Browse hotspot results",
                                               options=list(hotspot_choices.keys()),
                                               key="auto_browse")
                    sel_hotspot = hotspot_choices[sel_label]
                    imgs        = img_by_hotspot[sel_hotspot]
                    sel_img     = st.selectbox(
                        "Select run date",
                        options=imgs,
                        format_func=lambda p: os.path.basename(p)
                            .replace(sel_hotspot + "_", "").replace(".png", ""),
                        key="auto_img_sel"
                    )
                    st.image(sel_img, use_container_width=True)

        # ── Run history ───────────────────────────────────
        all_logs = sorted(glob.glob(os.path.join(AUTO_DIR, "run_*.json")), reverse=True)
        if len(all_logs) > 1:
            st.markdown('<div class="section-title" style="margin-top:24px">Run History</div>',
                        unsafe_allow_html=True)
            history_rows = []
            for lp in all_logs:
                try:
                    with open(lp) as f:
                        l = json.load(f)
                    history_rows.append({
                        "Date":       l.get("run_time", "")[:10],
                        "Time":       l.get("run_time", "")[11:16],
                        "Alerts":     len(l.get("alerts", [])),
                        "Skipped":    len(l.get("skipped", [])),
                        "Model Used": "Yes" if l.get("model_used") else "No",
                        "STAC":       l.get("stac_collection", "—"),
                    })
                except Exception:
                    pass
            if history_rows:
                st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.7rem; color: #6B8CAE; padding: 8px 0;'>
    Sentinel Watch · Cambodia–Thailand Border Deforestation Monitor ·
    Sentinel-2 · SiameseUNet12 · dNDVI threshold 0.15
</div>
""", unsafe_allow_html=True)