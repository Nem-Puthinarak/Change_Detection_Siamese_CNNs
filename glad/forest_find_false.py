"""
glad_clearance_validation.py
-----------------------------
Fetches GFW Integrated Deforestation Alerts (GLAD-L + RADD combined) for all
31 Sentinel Watch hotspots via the GFW Data API, rasterizes them onto the
Sentinel-2 pixel grid, cross-validates against dNDVI clearance detections, and
exports patch-level labels ready for Siamese CNN finetuning.

Covers your full date range: May 2025 → May 2026.

Requirements:
    pip install requests numpy scipy matplotlib pandas

Setup:
    1. Create a free account at https://www.globalforestwatch.org/my-gfw/
       (use email signup, NOT Google/Facebook — you need a password for the API)
    2. Get an access token:
         curl -X POST https://data-api.globalforestwatch.org/auth/token \
           -H 'Content-Type: application/x-www-form-urlencoded' \
           --data-urlencode 'username=YOUR_EMAIL' \
           --data-urlencode 'password=YOUR_PASSWORD'
    3. Use that token to get an API key:
         curl -X POST https://data-api.globalforestwatch.org/auth/apikey \
           -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
           -H 'Content-Type: application/json' \
           -d '{"alias":"sentinel-watch","email":"YOUR_EMAIL","organization":"research","domains":[]}'
    4. Paste your API key into GFW_API_KEY below.

Outputs (all under sentinel_outputs/glad/):
    alerts_raw_false/{hotspot}_{label}.json        — raw alert response
    patches/{hotspot}_{label}/
        {row}_{col}_before.npy               — (6, 224, 224) float32
        {row}_{col}_after.npy                — (6, 224, 224) float32
        {row}_{col}_mask.npy                 — (224, 224) uint8
        {row}_{col}_meta.json
    dataset_stats.json                       — per-band mean/std
    manifest.csv                             — full patch registry
    validation_summary.csv
    validation_summary.png

FIX LOG (v4):
    - v2: Removed broken /v1/geostore registration (404), replaced SQL bbox
      fallback with POST + GeoJSON geometry body.
    - v3 (this version):
      * Fixed SQL column name: alert_date → alert__date (double underscore).
        GFW raster datasets use the pixel_meaning name with double underscores
        as the SQL column identifier. Using alert_date caused HTTP 500.
      * Added /fields endpoint introspection at startup to auto-detect the
        correct date column name so the script self-heals if GFW renames it.
      * Fixed version selection: sorts known-good versions newest-first and
        probes each until the /fields endpoint responds 200, avoiding broken
        newly-published versions (e.g. v20260506 was returning 500).
      * Fixed fallback query: removed invalid end_date param; fallback now
        sends only geometry + sql (matching the primary path), just without
        the date WHERE clause, then filters client-side.
      * Added per-request delay increase when 500s are sustained (API overload
        signal) and a MAX_QUERY_WAIT cap.
      * Cleared stale cache files that contain empty [] due to wrong column
        name — call clear_bad_cache() or delete sentinel_outputs/glad/alerts_raw_false/
        before re-running.
"""

import os
import json
import time
import warnings
import signal
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from scipy.ndimage import binary_dilation
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")


GFW_API_KEY = os.getenv("GFW_API_KEY")

GFW_BASE        = "https://data-api.globalforestwatch.org"
DATASET         = "gfw_integrated_alerts"   

KNOWN_VERSIONS  = [
    "v20250401",
    "v20241201",
    "v20240901",
    "v20240601",
    "v20240301",
    "v20231201",
]

RAW_NPY_DIR = "sentinel_outputs/raw"
OUTPUT_DIR  = "sentinel_outputs/glad"

PATCH_SIZE           = 224
PATCH_STRIDE         = 112
MIN_PATCH_VALID_FRAC = 0.70
NDVI_CLEAR_THRESH    = 0.15
NDVI_POS_FRAC        = 0.05
GLAD_POS_FRAC        = 0.01


GLAD_DILATION_PX = 3

MAX_RETRIES   = 4
RETRY_BACKOFF = 2.0   

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "alerts_raw_false"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "patches"), exist_ok=True)


HOTSPOTS = [
    ("hotspot_01_osmach_west",            103.630, 14.410, 103.680, 14.450),
    ("hotspot_02_o3chean_village",        102.720, 13.765, 102.765, 13.810),
    ("hotspot_03_phnom_proek",            102.313, 13.295, 102.403, 13.385),
    ("hotspot_04_poipet_thai",            102.530, 13.653, 102.575, 13.688),
    ("hotspot_05_poipet_road",            102.690, 13.645, 102.735, 13.680),
    ("hotspot_06_osmach_east",            103.685, 14.408, 103.730, 14.445),
    ("hotspot_07_choam_chom_sngan",       104.040, 14.330, 104.085, 14.360),
    ("hotspot_08_bos_sbov",               104.730, 14.412, 104.775, 14.445),
    ("hotspot_09_chok_chey",              102.665, 13.745, 102.720, 13.780),
    ("hotspot_10_osmach_jan2026",         103.645, 14.400, 103.695, 14.445),
    ("hotspot_11_an_ses",                 104.942, 14.371, 104.987, 14.406),
    ("hotspot_12_ta_moan_thom",           103.243, 14.330, 103.288, 14.365),
    ("hotspot_13_ta_krabei",              103.351, 14.335, 103.396, 14.380),
    ("hotspot_14_preahvihear_thai",       104.657, 14.381, 104.702, 14.416),
    ("hotspot_15_preahvihear_temple",     104.657, 14.374, 104.702, 14.409),
    ("hotspot_16_thmor_da",               102.712, 12.114, 102.757, 12.150),
    ("hotspot_17_prei_chan",              102.714, 13.800, 102.759, 13.835),
    ("hotspot_18_beoung_trakoun",         102.830, 13.972, 102.877, 14.004),
    ("hotspot_19_thmor_doun",             103.269, 14.315, 103.359, 14.405),
    ("hotspot_20_emerald_triangle",       105.160, 14.299, 105.205, 14.350),
    ("hotspot_21_phnom_trob",             104.622, 14.368, 104.687, 14.433),
    ("hotspot_22_cham_yeam",              102.893, 11.627, 102.948, 11.682),
    ("hotspot_23_koh_kong",               102.801, 11.830, 102.879, 11.896),
    ("hotspot_24_botum_sakor",            102.698, 12.213, 102.753, 12.273),
    ("hotspot_25_Area_400",               102.661, 12.512, 102.714, 12.565),
    ("hotspot_26_South-wing-Area400",     102.674, 12.503, 102.746, 12.560),
    ("hotspot_27_Sompov_lun",             102.315, 13.394, 102.402, 13.481),
    ("hotspot_28_ta_moan_thom_mountain",  103.250, 14.318, 103.318, 14.386),
    ("hotspot_29_ta_krabei_mountain",     103.352, 14.320, 103.420, 14.388),
    ("hotspot_30_Anses_mountain",         104.950, 14.363, 104.994, 14.407),
    ("hotspot_31_Anses_mountain_south",   104.920, 14.353, 104.982, 14.415),
]

# Date pairs: (before_start, before_end, after_start, after_end, label)
# GLAD alert window = after_start → after_end
DATE_PAIRS = [
    ("2024-04-01", "2024-04-30", "2024-05-01", "2024-05-31", "May2024_event"),
    ("2024-05-01", "2024-05-31", "2024-06-01", "2024-06-30", "Jun2024_event"),
    ("2024-06-01", "2024-06-30", "2024-07-01", "2024-07-31", "Jul2024_event"),
    ("2024-07-01", "2024-07-31", "2024-08-01", "2024-08-31", "Aug2024_event"),
    ("2024-08-01", "2024-08-31", "2024-09-01", "2024-09-30", "Sep2024_event"),
    ("2024-09-01", "2024-09-30", "2024-10-01", "2024-10-31", "Oct2024_event"),
    ("2024-10-01", "2024-10-31", "2024-11-01", "2024-11-30", "Nov2024_event"),
    ("2024-11-01", "2024-11-30", "2024-12-01", "2024-12-31", "Dec2024_event"),
    ("2024-12-01", "2024-12-31", "2025-01-01", "2025-01-31", "Jan2025_event"),
    ("2025-01-01", "2025-01-31", "2025-02-01", "2025-02-28", "Feb2025_event"),
    ("2025-02-01", "2025-02-28", "2025-03-01", "2025-03-31", "Mar2025_event"),
    ("2025-03-01", "2025-03-31", "2025-04-01", "2025-04-30", "Apr2025_event"),
   
]

_interrupted      = False
_partial_patches  = []
_partial_scenes   = []

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\n  [!] Interrupted — saving partial results...")
    _interrupted = True

signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────────────────────
#  GFW API HELPERS
# ─────────────────────────────────────────────
def _headers():
    return {
        "x-api-key":    GFW_API_KEY,
        "Content-Type": "application/json",
        "Origin":       "https://localhost",  
    }


def _bbox_geojson(lon_min, lat_min, lon_max, lat_max):
    """Return a GeoJSON Polygon dict for the bounding box."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]]
    }


def probe_version_and_fields():
    """
    Find the newest quarterly version where /query actually works end-to-end.
    Tests a real tiny POST query (not just /fields) to confirm the version is
    fully operational before committing to it for the whole pipeline run.
    Returns (version_str, date_col_name, confidence_col_name).
    """
    DEFAULT_DATE_COL = "gfw_integrated_alerts__date"
    DEFAULT_CONF_COL = "gfw_integrated_alerts__confidence"

    
    TEST_GEOM = {
        "type": "Polygon",
        "coordinates": [[[102.313, 13.295], [102.403, 13.295],
                          [102.403, 13.385], [102.313, 13.385], [102.313, 13.295]]]
    }
    TEST_SQL = (
        f"SELECT latitude, longitude, {DEFAULT_DATE_COL}, {DEFAULT_CONF_COL} "
        f"FROM results "
        f"WHERE {DEFAULT_DATE_COL} >= '2024-01-01' "
        f"AND {DEFAULT_DATE_COL} <= '2024-06-30' LIMIT 1"
    )

    
    candidate_versions = []
    try:
        resp = requests.get(
            f"{GFW_BASE}/dataset/{DATASET}",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code == 200:
            live_versions = resp.json()["data"].get("versions", [])
            print(f"  Live versions available ({len(live_versions)} total)")
            for v in live_versions:
                # Keep only vYYYYMM01 quarterly snapshots (day part = "01")
                if len(v) == 9 and v.startswith("v") and v[7:] == "01":
                    candidate_versions.append(v)
            # Newest first
            candidate_versions = sorted(set(candidate_versions), reverse=True)
            print(f"  Quarterly versions found: {candidate_versions}")
    except Exception as e:
        print(f"  Could not list live versions: {e}")

    # Always include our known-good list as final fallback
    for v in KNOWN_VERSIONS:
        if v not in candidate_versions:
            candidate_versions.append(v)

    for version in candidate_versions:
        url = f"{GFW_BASE}/dataset/{DATASET}/{version}/query/json"
        try:
            resp = requests.post(
                url,
                headers=_headers(),
                json={"geometry": TEST_GEOM, "sql": TEST_SQL},
                timeout=60,
            )
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                if rows:
                    print(f"  Version {version} /query OK ({len(rows)} test rows) — using this version")
                    return version, DEFAULT_DATE_COL, DEFAULT_CONF_COL
                else:
                    print(f"  Version {version} /query 200 but empty data, skipping")
            else:
                print(f"  Version {version} /query → HTTP {resp.status_code}, skipping")
        except Exception as e:
            print(f"  Version {version} probe error: {e}")
        time.sleep(0.5)

    # Total fallback — use v20250401 which we confirmed works manually
    print(f"  WARNING: All probes failed. Forcing v20250401.")
    return "v20250401", DEFAULT_DATE_COL, DEFAULT_CONF_COL


def clear_bad_cache():
    
    raw_dir = os.path.join(OUTPUT_DIR, "alerts_raw_false")
    deleted = 0
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == 0:
                os.remove(path)
                deleted += 1
        except Exception:
            pass
    if deleted:
        print(f"  Cleared {deleted} stale empty cache files from {raw_dir}")
    else:
        print(f"  No stale cache files found in {raw_dir}")


def _post_query_with_retry(url, payload):
    """
    POST to GFW query endpoint with exponential back-off retry.
    Returns parsed JSON or None on total failure.
    """
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=_headers(),
                                 json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            # 429 = rate limit, 5xx = server error → retry
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"  HTTP {resp.status_code} on attempt {attempt}/{MAX_RETRIES}"
                      f" — retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            
            print(f"  GFW API error {resp.status_code}: {resp.text[:300]}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None
    print(f"  All {MAX_RETRIES} attempts failed.")
    return None


def fetch_glad_alerts(name, lon_min, lat_min, lon_max, lat_max,
                      start_date, end_date, label, version,
                      date_col, conf_col):
    """
    Query GFW integrated alerts for the AOI + date range via POST with
    GeoJSON geometry + SQL filter.

    KEY FIX (v3): The SQL column name for the alert date is alert__date
    (double underscore), not alert_date.  GFW raster tile-set datasets
    expose their pixel_meaning values as SQL column names and those values
    always use double underscores as separators.  Using alert_date caused
    HTTP 500 on every request in v2.

    Returns list of alert dicts.  Results are cached locally.
    """
    cache_path = os.path.join(
        OUTPUT_DIR, "alerts_raw_false", f"{name}_{label}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # Reject stale empty caches (written by v2 with wrong column name)
        if len(cached) > 0:
            print(f"  Cache hit ({len(cached)} alerts): {os.path.basename(cache_path)}")
            return cached
        else:
            print(f"  Stale empty cache — re-querying: {os.path.basename(cache_path)}")
            os.remove(cache_path)

    url = f"{GFW_BASE}/dataset/{DATASET}/{version}/query/json"

    
    sql = (
        f"SELECT latitude, longitude, {date_col}, {conf_col} FROM results "
        f"WHERE {date_col} >= '{start_date}' "
        f"AND {date_col} <= '{end_date}'"
    )
    payload = {
        "geometry": _bbox_geojson(lon_min, lat_min, lon_max, lat_max),
        "sql": sql,
    }
    print(f"    SQL: {sql}")

    result = _post_query_with_retry(url, payload)

    if result is not None:
        data = result.get("data", [])
        print(f"  Got {len(data)} alerts from GFW API")
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    
    print("  Trying no-date-filter fallback (client-side date filtering)...")
    fallback_sql = f"SELECT latitude, longitude, {date_col}, {conf_col} FROM results"
    try:
        resp = requests.post(
            url,
            headers=_headers(),
            json={
                "geometry": _bbox_geojson(lon_min, lat_min, lon_max, lat_max),
                "sql": fallback_sql,
            },
            timeout=180,
        )
        if resp.status_code == 200:
            all_data = resp.json().get("data", [])
            # Filter by date client-side
            filtered = []
            for alert in all_data:
                d = str(alert.get(date_col) or alert.get("gfw_integrated_alerts__date") or "")
                if d and start_date <= d[:10] <= end_date:
                    filtered.append(alert)
            print(f"  Fallback: {len(all_data)} total → {len(filtered)} in date range")
            with open(cache_path, "w") as f:
                json.dump(filtered, f)
            return filtered
        print(f"  Fallback failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  Fallback error: {e}")

    # Return empty list so the rest of the pipeline continues gracefully
    print("  No alerts retrieved — treating as zero alerts for this period.")
    with open(cache_path, "w") as f:
        json.dump([], f)
    return []

def rasterize_glad(alerts, lon_min, lat_min, lon_max, lat_max,
                   H, W, confidence_filter=True, dilation_px=GLAD_DILATION_PX,
                   conf_col="gfw_integrated_alerts__confidence"):
    """
    Map GLAD alert lat/lon points onto the S2 pixel grid.
    confidence_filter: if True, keeps only high (≥2) confidence alerts.
    Returns binary uint8 mask (H, W).
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not alerts:
        return mask

    # API returns confidence as string: "nominal", "high", or "highest"
    CONF_MAP = {"nominal": 2, "high": 3, "highest": 4}
    min_confidence = 3 if confidence_filter else 2  # 3 = "high"
    count = 0

    for alert in alerts:
        try:
            lon = float(alert.get("longitude") or alert.get("lon") or 0)
            lat = float(alert.get("latitude")  or alert.get("lat") or 0)
            raw_conf = alert.get(conf_col) or alert.get("gfw_integrated_alerts__confidence") or 1
            # Handle both string ("high") and legacy integer (3) forms
            if isinstance(raw_conf, str):
                conf = CONF_MAP.get(raw_conf.lower(), 1)
            else:
                conf = int(raw_conf)
        except (TypeError, ValueError):
            continue

        if confidence_filter and conf < min_confidence:
            continue

        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue

        px = int((lon - lon_min) / (lon_max - lon_min) * W)
        py = int((lat_max - lat) / (lat_max - lat_min) * H)
        px = min(max(px, 0), W - 1)
        py = min(max(py, 0), H - 1)
        mask[py, px] = 1
        count += 1

    print(f"  Rasterized {count} high-confidence alerts onto {H}×{W} grid")

    if dilation_px > 0 and mask.any():
        d      = 2 * dilation_px + 1
        struct = np.ones((d, d), dtype=bool)
        mask   = binary_dilation(mask, structure=struct).astype(np.uint8)

    return mask

def safe_band(arr, idx):
    b = arr[:, :, idx].astype(np.float32)
    b[b == 0] = np.nan
    return b


def compute_ndvi(arr):
    b08   = safe_band(arr, 3)
    b04   = safe_band(arr, 2)
    denom = np.nansum([b08, b04], axis=0)
    denom[denom == 0] = 1e-6
    return (b08 - b04) / denom


def compute_dndvi_mask(before, after, threshold=NDVI_CLEAR_THRESH):
    dndvi = compute_ndvi(before) - compute_ndvi(after)
    return (dndvi > threshold).astype(np.uint8), dndvi

def load_npy_stacks(name):
    raw_dir = os.path.join(RAW_NPY_DIR, name)
    if not os.path.isdir(raw_dir):
        return None, None

    files        = sorted(f for f in os.listdir(raw_dir) if f.endswith(".npy"))
    before_files = [f for f in files if "_before_" in f]
    after_files  = [f for f in files if "_after_"  in f]

    if not before_files or not after_files:
        if len(files) < 2:
            return None, None
        before_files = files[:len(files) // 2]
        after_files  = files[len(files) // 2:]

    before = np.load(os.path.join(raw_dir, before_files[-1])).astype(np.float32)
    after  = np.load(os.path.join(raw_dir, after_files[-1])).astype(np.float32)

    # normalise to (H, W, C) if stored as (C, H, W)
    if before.ndim == 3 and before.shape[0] < before.shape[2]:
        before = np.moveaxis(before, 0, -1)
    if after.ndim == 3 and after.shape[0] < after.shape[2]:
        after = np.moveaxis(after, 0, -1)

    return before, after

def assign_confidence(ndvi_pos, glad_pos):
    """
    Four-way confidence for clearance patches.
      high        → dNDVI and GLAD both agree (either both signal or both quiet)
      uncertain   → dNDVI only (possible agriculture cycle, fire, or fresh clearing)
      glad_only   → GLAD only (likely cloud cover obscuring S2)
      negative    → neither detects anything (label=0, confidence=high)
    """
    if ndvi_pos and glad_pos:
        return 1, "high"
    elif ndvi_pos and not glad_pos:
        return 1, "uncertain"
    elif not ndvi_pos and glad_pos:
        return 1, "glad_only"
    else:
        return 0, "high"


def extract_patches(before, after, dndvi_mask, glad_mask,
                    hotspot, label, lon_min, lat_min, lon_max, lat_max):
    H, W = before.shape[:2]
    patch_dir = os.path.join(OUTPUT_DIR, "patches", f"{hotspot}_{label}")
    os.makedirs(patch_dir, exist_ok=True)
    records = []

    for row_idx, y in enumerate(range(0, H - PATCH_SIZE + 1, PATCH_STRIDE)):
        for col_idx, x in enumerate(range(0, W - PATCH_SIZE + 1, PATCH_STRIDE)):

            b_patch    = before[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            a_patch    = after [y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            ndvi_patch = dndvi_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            glad_patch = glad_mask [y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            valid_frac = float(np.mean(np.isfinite(b_patch) & (b_patch != 0)))
            if valid_frac < MIN_PATCH_VALID_FRAC:
                continue

            ndvi_frac = float(ndvi_patch.mean())
            glad_frac = float(glad_patch.mean())
            ndvi_pos  = ndvi_frac > NDVI_POS_FRAC
            glad_pos  = glad_frac > GLAD_POS_FRAC

            label_val, confidence = assign_confidence(ndvi_pos, glad_pos)
            combined_mask = np.clip(ndvi_patch + glad_patch, 0, 1).astype(np.uint8)

            prefix = f"{row_idx:04d}_{col_idx:04d}"
            np.save(os.path.join(patch_dir, f"{prefix}_before.npy"),
                    np.moveaxis(b_patch, -1, 0).astype(np.float32))
            np.save(os.path.join(patch_dir, f"{prefix}_after.npy"),
                    np.moveaxis(a_patch, -1, 0).astype(np.float32))
            np.save(os.path.join(patch_dir, f"{prefix}_mask.npy"),
                    combined_mask)

            meta = {
                "hotspot":     hotspot,
                "period":      label,
                "patch_id":    prefix,
                "row":         row_idx,
                "col":         col_idx,
                "y_px":        y,
                "x_px":        x,
                "label":       label_val,
                "confidence":  confidence,
                "ndvi_frac":   round(ndvi_frac, 4),
                "glad_frac":   round(glad_frac, 4),
                "valid_frac":  round(valid_frac, 4),
                "lon_min":     lon_min,
                "lat_min":     lat_min,
                "lon_max":     lon_max,
                "lat_max":     lat_max,
                "before_path": os.path.join(patch_dir, f"{prefix}_before.npy"),
                "after_path":  os.path.join(patch_dir, f"{prefix}_after.npy"),
                "mask_path":   os.path.join(patch_dir, f"{prefix}_mask.npy"),
            }
            with open(os.path.join(patch_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            records.append(meta)

    pos = sum(1 for r in records if r["label"] == 1)
    neg = len(records) - pos
    print(f"    Patches: {len(records)}  |  pos={pos}  neg={neg}")
    return records

def assign_splits(manifest, val_frac=0.15, test_frac=0.15, seed=42):
    """Hotspot-level split — all patches from one hotspot stay in one fold."""
    rng      = np.random.default_rng(seed)
    hotspots = list({r["hotspot"] for r in manifest})
    rng.shuffle(hotspots)
    n        = len(hotspots)
    n_val    = max(1, int(n * val_frac))
    n_test   = max(1, int(n * test_frac))
    val_set  = set(hotspots[:n_val])
    test_set = set(hotspots[n_val:n_val + n_test])
    for r in manifest:
        hs = r["hotspot"]
        r["split"] = "test" if hs in test_set else \
                     "val"  if hs in val_set  else "train"
    return manifest


def compute_dataset_stats(manifest):
    print("\n  Computing per-band normalization stats (train split)...")
    band_sums = np.zeros(6)
    band_sq   = np.zeros(6)
    n_total   = 0

    train_paths = [r["before_path"] for r in manifest
                   if r.get("split") == "train" and os.path.exists(r["before_path"])]
    if not train_paths:
        train_paths = [r["before_path"] for r in manifest
                       if os.path.exists(r["before_path"])]

    for path in train_paths:
        arr = np.load(path).astype(np.float32)
        for b in range(min(6, arr.shape[0])):
            vals = arr[b][np.isfinite(arr[b])]
            band_sums[b] += vals.sum()
            band_sq[b]   += (vals ** 2).sum()
        n_total += PATCH_SIZE * PATCH_SIZE

    n     = max(n_total, 1)
    means = band_sums / n
    stds  = np.sqrt(np.maximum(band_sq / n - means ** 2, 0))

    stats = {
        "bands":      ["B02", "B03", "B04", "B08", "B11", "B12"],
        "mean":       means.tolist(),
        "std":        stds.tolist(),
        "n_patches":  len(train_paths),
        "patch_size": PATCH_SIZE,
        "source":     "GLAD integrated alerts clearance validation",
    }
    path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats → {path}")
    print(f"  Band means: {np.round(means, 4)}")
    print(f"  Band stds : {np.round(stds,  4)}")
    return stats


def compute_scene_metrics(name, label, before, after, glad_mask, n_alerts):
    if before is None:
        return None
    dndvi_mask, _ = compute_dndvi_mask(before, after)
    ndvi_px       = int(np.sum(dndvi_mask))
    glad_px       = int(np.sum(glad_mask))
    ndvi_detected = ndvi_px > 0
    glad_detected = glad_px > 0

    tp = int(ndvi_detected and glad_detected)
    fp = int(ndvi_detected and not glad_detected)
    fn = int(not ndvi_detected and glad_detected)
    tn = int(not ndvi_detected and not glad_detected)
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec  = tp / (tp + fn) if (tp + fn) > 0 else None
    f1   = (2 * prec * rec / (prec + rec)) if (prec and rec) else None

    return {
        "hotspot":       name,
        "period":        label,
        "ndvi_detected": ndvi_detected,
        "ndvi_pixels":   ndvi_px,
        "glad_alerts":   n_alerts,
        "glad_pixels":   glad_px,
        "glad_detected": glad_detected,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 3) if prec is not None else "N/A",
        "recall":    round(rec,  3) if rec  is not None else "N/A",
        "f1":        round(f1,   3) if f1   is not None else "N/A",
        "agreement": "YES" if (ndvi_detected == glad_detected) else "NO",
    }

BG = "#0D1B2A"

def plot_summary(summary_df, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor(BG)

    # ── 1. GLAD alert counts ──
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    totals = summary_df.groupby("hotspot")["glad_alerts"].sum().sort_values()
    short  = [h.replace("hotspot_", "").replace("_", " ") for h in totals.index]
    colors = ["#2A9D8F" if v > 0 else "#457B9D" for v in totals.values]
    bars   = ax1.barh(short, totals.values, color=colors, height=0.6)
    ax1.set_xlabel("GLAD alert count (summed)", color="white")
    ax1.set_title("GLAD alerts per hotspot", color="white",
                  fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=7)
    ax1.spines[:].set_color("#444")
    for bar, val in zip(bars, totals.values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", color="white", fontsize=7)
    ax1.legend(handles=[
        mpatches.Patch(color="#2A9D8F", label="Alerts found"),
        mpatches.Patch(color="#457B9D", label="No alerts"),
    ], facecolor="#1B2838", labelcolor="white", fontsize=8)

    # ── 2. TP / FP / FN / TN ──
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    cats   = {"TP": "#2A9D8F", "FP": "#E9C46A", "FN": "#E76F51", "TN": "#457B9D"}
    bottom = np.zeros(len(summary_df))
    xlabels = [
        f"{r['hotspot'].replace('hotspot_','')[:14]}[{r['period'][:3]}]"
        for _, r in summary_df.iterrows()
    ]
    for cat, color in cats.items():
        vals = summary_df[cat].values.astype(float)
        ax2.bar(range(len(summary_df)), vals, bottom=bottom,
                color=color, label=cat, width=0.7)
        bottom += vals
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels(xlabels, rotation=60, ha="right",
                        color="white", fontsize=6)
    ax2.set_ylabel("Count", color="white")
    ax2.set_title("dNDVI vs GLAD agreement", color="white",
                  fontsize=11, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#1B2838", labelcolor="white", fontsize=8)

    # ── 3. Patch label distribution ──
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    if manifest:
        mdf = pd.DataFrame(manifest)
        conf_order = [
            (1, "high",       "#2A9D8F", "Pos / High (both agree)"),
            (1, "uncertain",  "#E9C46A", "Pos / Uncertain (NDVI only)"),
            (1, "glad_only",  "#F4A261", "Pos / GLAD only"),
            (0, "high",       "#457B9D", "Neg / High"),
        ]
        labels_list, vals, clrs = [], [], []
        for lbl, conf, clr, name_str in conf_order:
            ct = len(mdf[(mdf["label"] == lbl) & (mdf["confidence"] == conf)])
            labels_list.append(name_str)
            vals.append(ct)
            clrs.append(clr)
        bars3 = ax3.bar(labels_list, vals, color=clrs, width=0.5)
        ax3.set_ylabel("Patch count", color="white")
        ax3.set_title("Clearance patch label distribution", color="white",
                      fontsize=11, fontweight="bold")
        ax3.tick_params(colors="white", labelsize=7)
        ax3.spines[:].set_color("#444")
        for bar, val in zip(bars3, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, str(val),
                     ha="center", color="white", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "validation_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"\n  Chart → {path}")

def save_partial(all_patches, all_scene_results):
    print("  Saving partial manifest and scene results...")
    if all_patches:
        all_patches = assign_splits(all_patches)
        pd.DataFrame(all_patches).to_csv(
            os.path.join(OUTPUT_DIR, "manifest_partial.csv"), index=False)
        print(f"  Partial manifest → {OUTPUT_DIR}/manifest_partial.csv")
    if all_scene_results:
        pd.DataFrame(all_scene_results).to_csv(
            os.path.join(OUTPUT_DIR, "validation_summary_partial.csv"), index=False)
        print(f"  Partial scenes  → {OUTPUT_DIR}/validation_summary_partial.csv")

def main():
    global _partial_patches, _partial_scenes

    if GFW_API_KEY == "YOUR_GFW_API_KEY_HERE":
        print("=" * 65)
        print("  ERROR: Add your GFW API key to GFW_API_KEY")
        print("  Steps to get one (free):")
        print("  1. Sign up at https://www.globalforestwatch.org/my-gfw/")
        print("     (use email signup, NOT Google/Facebook)")
        print("  2. Get a token:")
        print("     curl -X POST https://data-api.globalforestwatch.org/auth/token \\")
        print("       -H 'Content-Type: application/x-www-form-urlencoded' \\")
        print("       --data-urlencode 'username=YOUR_EMAIL' \\")
        print("       --data-urlencode 'password=YOUR_PASSWORD'")
        print("  3. Get an API key:")
        print("     curl -X POST https://data-api.globalforestwatch.org/auth/apikey \\")
        print("       -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \\")
        print("       -H 'Content-Type: application/json' \\")
        print("       -d '{\"alias\":\"sentinel-watch\",\"email\":\"YOUR_EMAIL\",")
        print("            \"organization\":\"research\",\"domains\":[]}'")
        print("=" * 65)
        return

    print("=" * 65)
    print("  GLAD CLEARANCE VALIDATION — Siamese CNN edition  (v4)")
    print("  Using: GFW Integrated Alerts (GLAD-L + RADD)")
    print("  Coverage: May 2025 → May 2026")
    print("=" * 65)

    
    clear_bad_cache()

    
    version, date_col, conf_col = probe_version_and_fields()

    all_scene_results = []
    all_patches       = []

    try:
        for name, lon_min, lat_min, lon_max, lat_max in HOTSPOTS:
            if _interrupted:
                break

            print(f"\n{'─'*65}")
            print(f"  {name}")
            print(f"{'─'*65}")

            before, after = load_npy_stacks(name)
            if before is None:
                print("  No NPY stacks — GLAD mask will be built, patches skipped")

            H = before.shape[0] if before is not None else \
                max(1, int((lat_max - lat_min) * 110574 / 10))
            W = before.shape[1] if before is not None else \
                max(1, int((lon_max - lon_min) * 111320 *
                           np.cos(np.radians((lat_min + lat_max) / 2)) / 10))

            for bf_s, bf_e, af_s, af_e, label in DATE_PAIRS:
                if _interrupted:
                    break

                print(f"\n  [{label}]  GLAD window: {af_s} → {af_e}")

                alerts = fetch_glad_alerts(
                    name, lon_min, lat_min, lon_max, lat_max,
                    af_s, af_e, label, version, date_col, conf_col)
                time.sleep(0.3)   # be polite to the API

                glad_mask = rasterize_glad(
                    alerts, lon_min, lat_min, lon_max, lat_max, H, W,
                    conf_col=conf_col)

                scene = compute_scene_metrics(
                    name, label, before, after, glad_mask, len(alerts))
                if scene:
                    all_scene_results.append(scene)

                if before is not None:
                    dndvi_mask, _ = compute_dndvi_mask(before, after)
                    patches = extract_patches(
                        before, after, dndvi_mask, glad_mask,
                        name, label, lon_min, lat_min, lon_max, lat_max)
                    all_patches.extend(patches)

            # keep references for interrupt handler
            _partial_patches = all_patches
            _partial_scenes  = all_scene_results

    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        save_partial(all_patches, all_scene_results)
        raise

    if _interrupted:
        save_partial(all_patches, all_scene_results)
        sys.exit(0)

    
    all_patches = assign_splits(all_patches)

    manifest_df   = pd.DataFrame(all_patches)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  Manifest → {manifest_path}  ({len(manifest_df)} patches)")

   
    if all_patches:
        compute_dataset_stats(all_patches)

    # ── scene summary ──
    if all_scene_results:
        summary_df = pd.DataFrame(all_scene_results)
        csv_path   = os.path.join(OUTPUT_DIR, "validation_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        tp   = summary_df["TP"].sum()
        fp   = summary_df["FP"].sum()
        fn   = summary_df["FN"].sum()
        tn   = summary_df["TN"].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print("\n" + "=" * 65)
        print("  SCENE-LEVEL RESULTS")
        print("=" * 65)
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision : {prec:.3f}")
        print(f"  Recall    : {rec:.3f}")
        print(f"  F1 Score  : {f1:.3f}")
        agree = (summary_df["agreement"] == "YES").sum()
        print(f"  Agreement : {agree}/{len(summary_df)} scene-periods")

        plot_summary(summary_df, all_patches)

    # ── patch summary ──
    if all_patches:
        mdf = pd.DataFrame(all_patches)
        print("\n" + "=" * 65)
        print("  PATCH SUMMARY")
        print("=" * 65)
        for split in ["train", "val", "test"]:
            sub  = mdf[mdf["split"] == split]
            pos  = (sub["label"] == 1).sum()
            neg  = (sub["label"] == 0).sum()
            hc   = (sub["confidence"] == "high").sum()
            unc  = (sub["confidence"] == "uncertain").sum()
            glad = (sub["confidence"] == "glad_only").sum()
            print(f"  {split:6s}: {len(sub):5d} patches  |  "
                  f"pos={pos}  neg={neg}  "
                  f"high={hc}  uncertain={unc}  glad_only={glad}")

    print(f"\n  Done. Outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()