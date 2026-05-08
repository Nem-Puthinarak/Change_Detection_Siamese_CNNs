

import os, io, json, time, datetime
import numpy as np
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.ndimage import gaussian_filter
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "finetune_best.pt")
USE_MODEL  = os.path.exists(MODEL_PATH)

if USE_MODEL:
    import torch
    import torch.nn as nn
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("  [WARN] segmentation_models_pytorch not installed — falling back to dNDVI threshold")
        USE_MODEL = False

CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

if not all([CLIENT_ID, CLIENT_SECRET, CDSE_USERNAME, CDSE_PASSWORD]):
    raise ValueError("Missing credentials — check your .env file")

RESOLUTION     = 10
MAX_CLOUD_PCT  = 70        
CHANGE_THRESH  = 0.4       
DNBR_THRESH    = 0.27      
DNDVI_THRESH   = 0.15      
MIN_CHANGE_KM2 = 0.005     


FORCE_BASELINE_DATE    = None   
BASELINE_LOOKBACK_DAYS = 90     
BASELINE_TARGET_MONTH  = 2      

OUTPUT_DIR   = "sentinel_display"
STATE_FILE   = os.path.join(OUTPUT_DIR, "auto_state.json")
DASH_RGB_DIR = os.path.join(OUTPUT_DIR, "truecolor")
DASH_NPY_DIR = os.path.join(OUTPUT_DIR, "npy_flat")

for folder in ["auto/raw", "auto/results", "auto/alerts", "truecolor", "npy_flat"]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


HOTSPOTS = {
    "hotspot_01_osmach_west":        (103.630, 14.410, 103.680, 14.450),
    "hotspot_02_o3chean_village":    (102.720, 13.765, 102.765, 13.810),
    "hotspot_03_jorkjei_village":    (102.680, 13.748, 102.725, 13.783),
    "hotspot_04_poipet_thai":        (102.530, 13.653, 102.575, 13.688),
    "hotspot_05_poipet_road":        (102.690, 13.645, 102.735, 13.680),
    "hotspot_06_osmach_east":        (103.685, 14.408, 103.730, 14.445),
    "hotspot_07_choam_chom_sngan":   (104.040, 14.330, 104.085, 14.360),
    "hotspot_08_bos_sbov":           (104.730, 14.412, 104.775, 14.445),
    "hotspot_09_chok_chey":          (102.665, 13.745, 102.720, 13.780),
    "hotspot_10_osmach_jan2026":     (103.645, 14.400, 103.695, 14.445),
    "hotspot_11_an_ses":             (104.942, 14.371, 104.987, 14.406),
    "hotspot_12_ta_moan_thom":       (103.243, 14.330, 103.288, 14.365),
    "hotspot_13_ta_krabei":          (103.351, 14.335, 103.396, 14.380),
    "hotspot_14_preahvihear_thai":   (104.657, 14.381, 104.702, 14.416),
    "hotspot_15_preahvihear_temple": (104.657, 14.374, 104.702, 14.409),
    "hotspot_16_thmor_da":           (102.712, 12.114, 102.757, 12.150),
    "hotspot_17_prei_chan":          (102.714, 13.800, 102.759, 13.835),
}


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B08","B11","B12","SCL"],
              units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN"] }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  if ([9, 10].includes(s.SCL)) return [0, 0, 0, 0, 0, 0];
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12];
}
"""

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL    = "https://sh.dataspace.copernicus.eu/api/v1/process"
STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"

STAC_COLLECTIONS = [
    "SENTINEL-2-L2A",
    "sentinel-2-l2a",
    "SENTINEL-2",
]


def make_session():
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]
    ))
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = make_session()

if USE_MODEL:
    class SiameseUNet12(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=12,
                classes=1,
                activation="sigmoid",
            )
        def forward(self, img1, img2):
            return self.unet(torch.cat([img1, img2], dim=1))

def load_model():
    if not USE_MODEL:
        return None
    print(f"  Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SiameseUNet12().to(device)
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded on {device}")
    return model, device

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

_token_cache = {"token": None, "expires_at": 0}

def get_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    print("  Authenticating...")
    r = SESSION.post(TOKEN_URL, data={
        "grant_type":   "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }, timeout=30)

    if r.status_code == 200:
        token = r.json()["access_token"]
        print("  Token OK")
    else:
        r2 = SESSION.post(TOKEN_URL, data={
            "grant_type": "password",
            "client_id":  "cdse-public",
            "username":    CDSE_USERNAME,
            "password":    CDSE_PASSWORD,
        }, timeout=30)
        if r2.status_code == 200:
            token = r2.json()["access_token"]
            print("  Token OK (password fallback)")
        else:
            raise RuntimeError(f"Auth failed: {r2.text[:200]}")

    _token_cache["token"]      = token
    _token_cache["expires_at"] = now + 540
    return token

_working_collection = "NOT_PROBED"

def probe_stac_collections():
    print("\n  [DIAG] Probing STAC endpoint for working collection name...")
    test_bbox  = [103.630, 14.410, 103.680, 14.450]
    test_end   = datetime.date.today()
    test_start = test_end - datetime.timedelta(days=180)
    dt_str     = f"{test_start}T00:00:00Z/{test_end}T23:59:59Z"

    for col in STAC_COLLECTIONS:
        try:
            r = SESSION.post(STAC_URL, json={
                "bbox":        test_bbox,
                "datetime":    dt_str,
                "collections": [col],
                "limit":       3,
            }, timeout=30)
            n = len(r.json().get("features", []))
            print(f"  [DIAG]   {col!r:30s}  HTTP {r.status_code}  features={n}")
            if r.status_code == 200 and n > 0:
                print(f"  [DIAG] Using collection: {col!r}")
                return col
        except Exception as e:
            print(f"  [DIAG]   {col!r:30s}  ERROR: {e}")

    print("  [DIAG]   Trying without collection filter...")
    try:
        r = SESSION.post(STAC_URL, json={
            "bbox":     test_bbox,
            "datetime": dt_str,
            "limit":    3,
        }, timeout=30)
        feats = r.json().get("features", [])
        n = len(feats)
        print(f"  [DIAG]   (no filter)                    HTTP {r.status_code}  features={n}")
        if r.status_code == 200 and n > 0:
            found_cols = set()
            for f in feats:
                c = (f.get("collection")
                     or f.get("properties", {}).get("collection", "")
                     or "")
                if c:
                    found_cols.add(c)
            print(f"  [DIAG]   Collections in results: {found_cols}")
            if found_cols:
                col = list(found_cols)[0]
                print(f"  [DIAG] Auto-detected collection: {col!r}")
                return col
            print("  [DIAG] No collection field found. Will search without filter.")
            return None
    except Exception as e:
        print(f"  [DIAG]   (no filter) ERROR: {e}")

    print("  [DIAG] WARNING: STAC probe failed entirely.")
    return None

def get_working_collection():
    global _working_collection
    if _working_collection == "NOT_PROBED":
        _working_collection = probe_stac_collections()
    return _working_collection

def search_latest(aoi, days_back=60):
    """Find the most recent low-cloud scene within the last N days."""
    lon_min, lat_min, lon_max, lat_max = aoi
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days_back)
    print(f"  Searching last {days_back} days ({start} → {end})...")

    collection = get_working_collection()
    payload = {
        "bbox":     [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit":    50,
    }
    if collection:
        payload["collections"] = [collection]

    r = SESSION.post(STAC_URL, json=payload, timeout=60)
    if r.status_code != 200:
        print(f"  Catalog error {r.status_code}: {r.text[:200]}")
        return None

    all_features = r.json().get("features", [])
    print(f"  Catalog returned {len(all_features)} total scenes (before cloud filter)")

    if not all_features:
        if days_back < 90:
            new_days = min(days_back * 2, 90)
            print(f"  No scenes — retrying with {new_days} days...")
            return search_latest(aoi, days_back=new_days)
        print(f"  No scenes found in last {days_back} days")
        return None

    cleaned = [
        f for f in all_features
        if f.get("properties", {}).get("eo:cloud_cover") is None
        or f["properties"]["eo:cloud_cover"] <= MAX_CLOUD_PCT
    ]
    print(f"  After cloud filter (<= {MAX_CLOUD_PCT}%): {len(cleaned)} scenes")

    if not cleaned:
        if days_back < 90:
            new_days = min(days_back * 2, 90)
            print(f"  All cloudy — retrying with {new_days} days...")
            return search_latest(aoi, days_back=new_days)
        print(f"  No clean scenes in last {days_back} days")
        return None

    cleaned.sort(key=lambda f: f["properties"].get("datetime", ""), reverse=True)
    best  = cleaned[0]
    date  = best["properties"]["datetime"][:10]
    cloud = best["properties"].get("eo:cloud_cover", "?")
    print(f"  Best scene: {date}  cloud={cloud}%")
    return date

# ── CATALOG SEARCH (historical baseline) ─────────────────
def find_historical_baseline(aoi, target_date=None, lookback_days=90, prefer_month=None):
    """
    Find the cleanest historical scene to use as a before-image.

    Priority order:
      1. If target_date given  → search ±15 days around that date
      2. If prefer_month given → search last N years for that month
      3. Otherwise             → search [today - lookback_days] to [today - 30 days]

    Returns the date string of the best scene, or None.
    """
    lon_min, lat_min, lon_max, lat_max = aoi
    collection = get_working_collection()

    def _search_window(start, end, label):
        print(f"  Historical search: {label} ({start} → {end})")
        payload = {
            "bbox":     [lon_min, lat_min, lon_max, lat_max],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit":    100,
        }
        if collection:
            payload["collections"] = [collection]
        try:
            r = SESSION.post(STAC_URL, json=payload, timeout=60)
            if r.status_code != 200:
                return []
            return r.json().get("features", [])
        except Exception as e:
            print(f"  Historical search error: {e}")
            return []

    candidates = []

    # Strategy 1: pinned target date ±15 days
    if target_date:
        d     = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
        start = d - datetime.timedelta(days=15)
        end   = d + datetime.timedelta(days=15)
        candidates = _search_window(start, end, f"around {target_date}")

    # Strategy 2: preferred dry-season month across recent years
    elif prefer_month:
        today = datetime.date.today()
        for year_offset in range(3):           # try last 3 years
            year = today.year - year_offset
            # skip if the target month hasn't happened yet this year
            if year == today.year and prefer_month >= today.month:
                continue
            month_start = datetime.date(year, prefer_month, 1)
            # last day of the month
            if prefer_month == 12:
                month_end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                month_end = datetime.date(year, prefer_month + 1, 1) - datetime.timedelta(days=1)
            feats = _search_window(month_start, month_end, f"{year}-{prefer_month:02d}")
            candidates.extend(feats)
            if len(candidates) >= 10:          # enough to pick from
                break

    # Strategy 3: recent lookback window (today-30 to today-lookback_days)
    else:
        end   = datetime.date.today() - datetime.timedelta(days=30)
        start = end - datetime.timedelta(days=lookback_days)
        candidates = _search_window(start, end, f"last {lookback_days} days (offset 30)")

    if not candidates:
        print(f"  No historical scenes found")
        return None

    # Filter by cloud cover, then sort cleanest first
    today_str = datetime.date.today().isoformat()
    cleaned = [
        f for f in candidates
        if f["properties"].get("datetime", "")[:10] < today_str
        and f.get("properties", {}).get("eo:cloud_cover", 100) <= MAX_CLOUD_PCT
    ]

    if not cleaned:
        print(f"  No clean historical scenes (all > {MAX_CLOUD_PCT}% cloud)")
        return None

    cleaned.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    best  = cleaned[0]
    date  = best["properties"]["datetime"][:10]
    cloud = best["properties"].get("eo:cloud_cover", "?")
    print(f"  Best historical baseline: {date}  cloud={cloud}%")
    return date

# ── IMAGE DOWNLOAD ────────────────────────────────────────
def download_image(token, aoi, scene_date):
    import tifffile
    lon_min, lat_min, lon_max, lat_max = aoi
    W = int((lon_max - lon_min) * 111320 * np.cos(np.radians((lat_min + lat_max) / 2)) / RESOLUTION)
    H = int((lat_max - lat_min) * 110540 / RESOLUTION)
    W, H = max(W, 64), max(H, 64)

    payload = {
        "input": {
            "bounds": {
                "bbox": [lon_min, lat_min, lon_max, lat_max],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{scene_date}T00:00:00Z",
                        "to":   f"{scene_date}T23:59:59Z",
                    },
                    "maxCloudCoverage": MAX_CLOUD_PCT,
                },
            }],
        },
        "output": {
            "width":  W,
            "height": H,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    print(f"  Downloading {W}x{H}px for {scene_date}...")
    r = SESSION.post(
        SH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "image/tiff",
        },
        json=payload,
        timeout=(60, 600),
    )
    if r.status_code != 200:
        raise RuntimeError(f"Download failed ({r.status_code}): {r.text[:200]}")
    arr = tifffile.imread(io.BytesIO(r.content)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
    print(f"  Shape: {arr.shape}")
    return arr

# ── BAND HELPERS ──────────────────────────────────────────
def fill_masked(band):
    mask = band == 0
    if mask.all():     return np.full_like(band, 0.1)
    if not mask.any(): return band
    band = band.copy()
    band[mask] = np.nanmedian(band[~mask])
    return band

def compute_ndvi(arr):
    b08 = fill_masked(arr[:, :, 3])
    b04 = fill_masked(arr[:, :, 2])
    d   = b08 + b04
    d[d == 0] = 1e-6
    return (b08 - b04) / d

def compute_nbr(arr):
    b08 = fill_masked(arr[:, :, 3])
    b12 = fill_masked(arr[:, :, 5])
    d   = b08 + b12
    d[d == 0] = 1e-6
    return (b08 - b12) / d

def arr_to_tensor(arr, device):
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.clamp(t / 0.3, 0.0, 1.0)
    return t.to(device)

def resize_to_match(arr1, arr2):
    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]
    if h1 == h2 and w1 == w2:
        return arr1, arr2
    h, w = min(h1, h2), min(w1, w2)
    return arr1[:h, :w], arr2[:h, :w]

def pad_to_multiple(tensor, multiple=32):
    """Pad (B,C,H,W) tensor so H and W are divisible by multiple."""
    import torch.nn.functional as F
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), h, w

# ── INFERENCE ─────────────────────────────────────────────
def run_model(model, device, before_arr, after_arr):
    before_arr, after_arr = resize_to_match(before_arr, after_arr)
    with torch.no_grad():
        t1 = arr_to_tensor(before_arr, device)
        t2 = arr_to_tensor(after_arr,  device)
        t1_pad, orig_h, orig_w = pad_to_multiple(t1)
        t2_pad, _,      _      = pad_to_multiple(t2)
        prob_pad = model(t1_pad, t2_pad).squeeze().cpu().numpy()
        prob = prob_pad[:orig_h, :orig_w]   # crop back to real size
    return prob

def detect_change(before_arr, after_arr, model=None, device=None):
    before_arr, after_arr = resize_to_match(before_arr, after_arr)
    pixel_km2  = 100 / 1e6

    dndvi      = compute_ndvi(before_arr) - compute_ndvi(after_arr)
    dnbr       = compute_nbr(before_arr)  - compute_nbr(after_arr)
    dndvi_mask = dndvi > DNDVI_THRESH
    dnbr_mask  = dnbr  > DNBR_THRESH

    result = {
        "dndvi_mask": dndvi_mask,
        "dnbr_mask":  dnbr_mask,
        "dndvi_km2":  float(np.sum(dndvi_mask) * pixel_km2),
        "dnbr_km2":   float(np.sum(dnbr_mask)  * pixel_km2),
        "model_mask": None,
        "model_km2":  0.0,
        "prob_map":   None,
    }

    if model is not None:
        prob               = run_model(model, device, before_arr, after_arr)
        model_mask         = prob > CHANGE_THRESH
        result["prob_map"]   = prob
        result["model_mask"] = model_mask
        result["model_km2"]  = float(np.sum(model_mask) * pixel_km2)

    return result

# ── VISUALISATION ─────────────────────────────────────────
BG = "#0D1B2A"

def gaussian_filter_img(img, sigma=1.0, strength=1.5):
    blurred   = gaussian_filter(img.astype(np.float64), sigma=sigma)
    sharpened = img.astype(np.float64) + strength * (img.astype(np.float64) - blurred)
    return np.clip(sharpened, 0, 1).astype(np.float32)

def make_rgb(arr):
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    for i, b in enumerate([2, 1, 0]):
        band  = fill_masked(arr[:, :, b].copy())
        valid = band[band > 0.001]
        if len(valid) < 100:
            continue
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    return gaussian_filter_img(rgb)

def save_result_figure(name, before_arr, after_arr, result, before_date, after_date):
    n_panels = 5 if result["model_mask"] is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 5))
    fig.patch.set_facecolor(BG)

    panels = [
        (make_rgb(before_arr),               f"Before\n{before_date}",                          None,       None, None),
        (make_rgb(after_arr),                f"After\n{after_date}",                             None,       None, None),
        (result["dndvi_mask"].astype(float), f"dNDVI Clearance\n{result['dndvi_km2']:.3f} km²",  "YlOrRd",  0,    1),
        (result["dnbr_mask"].astype(float),  f"dNBR Burn\n{result['dnbr_km2']:.3f} km²",         "YlOrRd",  0,    1),
    ]
    if result["model_mask"] is not None:
        panels.append(
            (result["prob_map"], f"AI Model\n{result['model_km2']:.3f} km²", "RdYlGn_r", 0, 1)
        )

    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        ax.set_facecolor(BG)
        if cmap:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(data)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.axis("off")

    label = name.replace("hotspot_", "").replace("_", " ").title()
    method_tag = f"AI model + dNDVI/dNBR" if USE_MODEL else "dNDVI/dNBR only"
    plt.suptitle(f"Sentinel Watch — {label}  [{method_tag}]",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "auto", "results", f"{name}_{after_date}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  Figure saved: {path}")

# ── DASHBOARD-COMPATIBLE SAVERS ───────────────────────────
def date_to_period_label(date_str):
    try:
        d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%b%Y_auto")
    except Exception:
        return "auto"

def save_rgb_for_dashboard(before_arr, after_arr, name, period_label,
                           before_date=None, after_date=None,
                           flagged=False, method="unknown"):
    """
    Save side-by-side before/after truecolor PNG and a metadata JSON
    so the dashboard knows which detection method was used.
    """
    before_rgb = make_rgb(before_arr)
    after_rgb  = make_rgb(after_arr)
    h = min(before_rgb.shape[0], after_rgb.shape[0])
    w = min(before_rgb.shape[1], after_rgb.shape[1])
    combined = np.concatenate([before_rgb[:h, :w], after_rgb[:h, :w]], axis=1)

    out = os.path.join(DASH_RGB_DIR, f"{name}_{period_label}_rgb.png")
    try:
        from PIL import Image as PILImage
        PILImage.fromarray((combined * 255).astype(np.uint8)).save(out)
    except ImportError:
        fig, ax = plt.subplots(figsize=(combined.shape[1] / 100, combined.shape[0] / 100))
        ax.imshow(combined)
        ax.axis("off")
        plt.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    # Metadata JSON — tells dashboard which method ran and whether change was flagged
    meta = {
        "name":         name,
        "period":       period_label,
        "before_date":  before_date,
        "after_date":   after_date,
        "model_used":   USE_MODEL,
        "model_path":   MODEL_PATH if USE_MODEL else None,
        "method":       method,       # "AI model" | "dNDVI/dNBR" | "baseline_only"
        "flagged":      flagged,
        "generated_at": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(DASH_RGB_DIR, f"{name}_{period_label}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Dashboard RGB  saved: {out}")
    print(f"  Dashboard meta saved: {meta_path}")

def save_npy_for_dashboard(arr, name, date_str):
    date_clean = date_str.replace("-", "")
    out = os.path.join(DASH_NPY_DIR, f"{name}_{date_clean}_stack.npy")
    np.save(out, arr)
    print(f"  Dashboard NPY  saved: {out}")

# ── MAIN LOOP ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SENTINEL WATCH — AUTOMATED RUN")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model : {'ON (' + MODEL_PATH + ')' if USE_MODEL else 'OFF (dNDVI/dNBR fallback)'}")
    if FORCE_BASELINE_DATE:
        print(f"  Baseline pinned to : {FORCE_BASELINE_DATE} ±15 days")
    elif BASELINE_TARGET_MONTH:
        print(f"  Baseline preference: month {BASELINE_TARGET_MONTH:02d} (dry season)")
    else:
        print(f"  Baseline lookback  : {BASELINE_LOOKBACK_DAYS} days")
    print("=" * 60)

    state     = load_state()
    model_obj = load_model() if USE_MODEL else None
    model     = model_obj[0] if model_obj else None
    device    = model_obj[1] if model_obj else None

    working_col = get_working_collection()
    if working_col is None:
        print("\n  [WARN] No working STAC collection detected.")
        print("  Searches will run without a collection filter.\n")

    token = get_token()

    alerts  = []
    skipped = []

    for name, bbox in HOTSPOTS.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        try:
            token = get_token()

            # ── Step 1: find + download latest scene ──────
            latest_date = search_latest(bbox, days_back=60)
            if not latest_date:
                print("  No recent imagery — skipping")
                skipped.append(name)
                continue

            latest_arr  = download_image(token, bbox, latest_date)
            latest_path = os.path.join(OUTPUT_DIR, "auto", "raw",
                                       f"{name}_{latest_date}_stack.npy")
            np.save(latest_path, latest_arr)
            save_npy_for_dashboard(latest_arr, name, latest_date)

            hotspot_state = state.get(name, {})
            baseline_path = hotspot_state.get("baseline_npy")
            baseline_date = hotspot_state.get("baseline_date")

            # ── Step 2: need historical baseline? ─────────
            # Triggers when:
            #   (a) no baseline at all, OR
            #   (b) baseline exists but equals latest (first-run trap) — we
            #       need an older scene to actually compare against
            need_historical = (
                not baseline_path
                or not os.path.exists(baseline_path)
                or baseline_date == latest_date
            )

            if need_historical:
                reason = "no baseline" if not baseline_path else f"baseline ({baseline_date}) == latest"
                print(f"  {reason} — searching for clean historical scene...")

                hist_date = find_historical_baseline(
                    bbox,
                    target_date=FORCE_BASELINE_DATE,
                    lookback_days=BASELINE_LOOKBACK_DAYS,
                    prefer_month=BASELINE_TARGET_MONTH,
                )

                if not hist_date or hist_date == latest_date:
                    # Truly nothing older available
                    print(f"  No usable historical scene — saving {latest_date} as baseline only")
                    state[name] = {"baseline_npy": latest_path, "baseline_date": latest_date}
                    save_state(state)
                    save_rgb_for_dashboard(
                        latest_arr, latest_arr, name,
                        date_to_period_label(latest_date),
                        before_date=latest_date, after_date=None,
                        flagged=False, method="baseline_only",
                    )
                    skipped.append(name)
                    continue

                # Download the historical baseline image
                print(f"  Downloading historical baseline ({hist_date})...")
                hist_arr  = download_image(token, bbox, hist_date)
                hist_path = os.path.join(OUTPUT_DIR, "auto", "raw",
                                         f"{name}_{hist_date}_stack.npy")
                np.save(hist_path, hist_arr)
                save_npy_for_dashboard(hist_arr, name, hist_date)

                # Permanently replace baseline with the historical scene
                state[name] = {"baseline_npy": hist_path, "baseline_date": hist_date}
                save_state(state)

                baseline_path = hist_path
                baseline_date = hist_date
                print(f"  Baseline set: {baseline_date}  →  comparing with latest: {latest_date}")

            # ── Step 4: load baseline + detect change ─────
            print(f"  Comparing: baseline={baseline_date}  latest={latest_date}")
            before_arr = np.load(baseline_path).astype(np.float32)
            save_npy_for_dashboard(before_arr, name, baseline_date)

            result = detect_change(before_arr, latest_arr, model, device)

            if result["model_mask"] is not None:
                primary_km2  = result["model_km2"]
                primary_name = "AI model"
                confirmed    = (result["dndvi_km2"] > MIN_CHANGE_KM2 or
                                result["dnbr_km2"]  > MIN_CHANGE_KM2)
            else:
                primary_km2  = max(result["dndvi_km2"], result["dnbr_km2"])
                primary_name = "dNDVI/dNBR"
                confirmed    = True

            flagged = primary_km2 > MIN_CHANGE_KM2 and confirmed

            print(f"  {primary_name}: {primary_km2:.4f} km²  |  "
                  f"dNDVI: {result['dndvi_km2']:.4f} km²  |  "
                  f"dNBR: {result['dnbr_km2']:.4f} km²")
            print(f"  Status: {'*** CHANGE DETECTED ***' if flagged else 'No significant change'}")

            # ── Step 5: save outputs ──────────────────────
            save_result_figure(name, before_arr, latest_arr, result, baseline_date, latest_date)
            save_rgb_for_dashboard(
                before_arr, latest_arr, name,
                date_to_period_label(latest_date),
                before_date=baseline_date,
                after_date=latest_date,
                flagged=flagged,
                method=primary_name,
            )

            if flagged:
                alerts.append({
                    "hotspot":     name,
                    "baseline":    baseline_date,
                    "latest":      latest_date,
                    "model_km2":   result["model_km2"],
                    "dndvi_km2":   result["dndvi_km2"],
                    "dnbr_km2":    result["dnbr_km2"],
                    "primary":     primary_name,
                    "primary_km2": primary_km2,
                })

                import shutil
                result_png  = os.path.join(OUTPUT_DIR, "auto", "results",
                                           f"{name}_{latest_date}.png")
                alert_path  = os.path.join(OUTPUT_DIR, "auto", "alerts",
                                           f"ALERT_{name}_{latest_date}.png")
                shutil.copy(result_png, alert_path)
                print(f"  Alert saved: {alert_path}")

                # Roll baseline forward on detected change
                state[name] = {"baseline_npy": latest_path, "baseline_date": latest_date}
                save_state(state)

        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] Saving state before exit...")
            save_state(state)
            raise

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            skipped.append(name)
            continue

        time.sleep(1)

    # ── SUMMARY ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if alerts:
        print(f"\n  *** {len(alerts)} HOTSPOT(S) WITH NEW ACTIVITY ***\n")
        for a in alerts:
            label = a["hotspot"].replace("hotspot_", "").replace("_", " ").title()
            print(f"  🔴  {label}")
            print(f"       Period  : {a['baseline']} → {a['latest']}")
            print(f"       {a['primary']}: {a['primary_km2']:.4f} km²")
            if a["model_km2"] > 0:
                print(f"       AI Model : {a['model_km2']:.4f} km²")
            print(f"       dNDVI    : {a['dndvi_km2']:.4f} km²")
            print(f"       dNBR     : {a['dnbr_km2']:.4f} km²")
            print()
    else:
        print("\n  No significant change detected across all hotspots.")

    print(f"  Skipped : {len(skipped)} hotspots (no imagery or error)")
    print(f"  Outputs : {OUTPUT_DIR}/auto/")
    print(f"  Alerts  : {OUTPUT_DIR}/auto/alerts/  ({len(alerts)} files)")
    print(f"  Dash RGB: {DASH_RGB_DIR}/")
    print(f"  Dash NPY: {DASH_NPY_DIR}/")

    log = {
        "run_time":        datetime.datetime.now().isoformat(),
        "alerts":          alerts,
        "skipped":         skipped,
        "model_used":      USE_MODEL,
        "stac_collection": _working_collection,
        "baseline_config": {
            "force_date":    FORCE_BASELINE_DATE,
            "lookback_days": BASELINE_LOOKBACK_DAYS,
            "target_month":  BASELINE_TARGET_MONTH,
        },
    }
    log_path = os.path.join(OUTPUT_DIR, "auto", f"run_{datetime.date.today()}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log     : {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()