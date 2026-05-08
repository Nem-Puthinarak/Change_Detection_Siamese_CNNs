import os
import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.ndimage import gaussian_filter

class HTTP1Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = []
        super().init_poolmanager(*args, **kwargs)

def make_session():
    session = requests.Session()
    adapter = HTTP1Adapter(max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = make_session()

CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

RESOLUTION       = 10
MAX_CLOUD_PCT    = 70
MAX_SCENES       = 10     
MIN_VALID_PCT    = 75
MAX_VALID_DELTA  = 30     
MAX_HAZE_INDEX   = 0.42   
OUTPUT_DIR       = "sentinel-outputs-neg"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL    = "https://sh.dataspace.copernicus.eu/api/v1/process"
STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"

for folder in ["nbr", "ndvi", "truecolor", "raw"]:
    os.makedirs(f"{OUTPUT_DIR}/{folder}", exist_ok=True)


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

DATE_PAIRS = [
    ("2025-04-01", "2025-04-30", "2025-05-01", "2025-05-31", "May2025_event"),
    ("2025-05-01", "2025-05-31", "2025-06-01", "2025-06-30", "Jun2025_event"),
    ("2025-06-01", "2025-06-30", "2025-07-01", "2025-07-31", "Jul2025_event"),
    ("2025-07-01", "2025-07-31", "2025-08-01", "2025-08-31", "Aug2025_event"),
    ("2025-08-01", "2025-08-31", "2025-09-01", "2025-09-30", "Sep2025_event"),
    ("2025-09-01", "2025-09-30", "2025-10-01", "2025-10-31", "Oct2025_event"),
    ("2025-10-01", "2025-10-31", "2025-11-01", "2025-11-30", "Nov2025_event"),
    ("2025-11-01", "2025-11-30", "2025-12-01", "2025-12-31", "Dec2025_event"),
    ("2025-12-01", "2025-12-31", "2026-01-01", "2026-01-31", "Jan2026_event"),
    ("2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28", "Feb2026_event"),
    ("2026-02-01", "2026-02-28", "2026-03-01", "2026-03-31", "Mar2026_event"),
    ("2026-03-01", "2026-03-31", "2026-04-01", "2026-04-30", "Apr2026_event"),
    ("2026-04-01", "2026-04-30", "2026-05-01", "2026-05-31", "May2026_event"),
]


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B08","B11","B12","SCL"],
      units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN"]
    }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  // Expanded: 3=cloud shadow, 8=cloud med prob, 9=cirrus, 10=cloud high prob, 11=snow
  if ([3, 8, 9, 10, 11].includes(s.SCL)) {
    return [0, 0, 0, 0, 0, 0];
  }
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12];
}
"""


def get_access_token():
    print("  Authenticating...")
    resp = SESSION.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }, timeout=30)
    if resp.status_code == 200:
        print("  Token OK")
        return resp.json()["access_token"]
    resp2 = SESSION.post(TOKEN_URL, data={
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": CDSE_USERNAME,
        "password": CDSE_PASSWORD,
    }, timeout=30)
    if resp2.status_code == 200:
        print("  Token OK (password fallback)")
        return resp2.json()["access_token"]
    raise SystemExit(f"Auth failed: {resp2.text[:200]}")

def search_catalog(start, end, aoi):
    lon_min, lat_min, lon_max, lat_max = aoi
    print(f"  Searching {start} to {end}...")
    resp = SESSION.post(STAC_URL, json={
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "collections": ["SENTINEL-2"],
        "limit": 100,
    }, timeout=60)
    if resp.status_code == 200:
        features = resp.json().get("features", [])
        features = [f for f in features if f.get("properties", {}).get("eo:cloud_cover", 999) <= MAX_CLOUD_PCT]
        features.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 999))
        if features:
            print(f"  Found {len(features)} scene(s)")
            return features
    print("  Trying OData fallback...")
    fp = (f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
          f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))")
    resp2 = SESSION.get(
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
        params={
            "$filter": (
                "Collection/Name eq 'SENTINEL-2' "
                f"and ContentDate/Start gt {start}T00:00:00.000Z "
                f"and ContentDate/Start lt {end}T23:59:59.000Z "
                f"and OData.CSC.Intersects(area=geography'SRID=4326;{fp}')"
            ),
            "$orderby": "ContentDate/Start desc",
            "$top": 100,
            "$expand": "Attributes",
        }, timeout=60)
    if resp2.status_code != 200:
        print(f"  OData failed ({resp2.status_code})")
        return []
    products = resp2.json().get("value", [])
    def get_cloud(p):
        return next((float(a["Value"]) for a in p.get("Attributes", []) if a["Name"] == "cloudCover"), 999.0)
    products = [p for p in products if get_cloud(p) <= MAX_CLOUD_PCT]
    products.sort(key=get_cloud)
    print(f"  Found {len(products)} scene(s) via OData")
    return products

def get_scene_date(item):
    if "properties" in item:
        return item["properties"].get("datetime", "")[:10]
    return item["ContentDate"]["Start"][:10]

def download_image(token, aoi, scene_date):
    import tifffile
    lon_min, lat_min, lon_max, lat_max = aoi
    km_x = (lon_max - lon_min) * 111.32 * np.cos(np.radians((lat_min + lat_max) / 2))
    km_y = (lat_max - lat_min) * 110.574
    W = max(1, int(km_x * 1000 / RESOLUTION))
    H = max(1, int(km_y * 1000 / RESOLUTION))
    if W > 2500 or H > 2500:
        scale = 2500 / max(W, H)
        W, H = int(W * scale), int(H * scale)
    d = datetime.datetime.strptime(scene_date, "%Y-%m-%d")
    t_from = (d - datetime.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    t_to   = (d + datetime.timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")
    payload = {
        "input": {
            "bounds": {"bbox": list(aoi), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "dataFilter": {
                    "timeRange": {"from": t_from, "to": t_to},
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": MAX_CLOUD_PCT,
                },
                "processing": {
                    "upsampling": "BICUBIC",
                    "downsampling": "BILINEAR",
                },
                "type": "sentinel-2-l2a",
            }],
        },
        "output": {"width": W, "height": H, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    print(f"  Downloading {W}x{H} px for {scene_date}...")
    resp = SESSION.post(SH_URL, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }, json=payload, timeout=(60, 600))
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed ({resp.status_code}): {resp.text[:200]}")
    arr = tifffile.imread(io.BytesIO(resp.content)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
    print(f"  Downloaded shape: {arr.shape}")
    return arr

def build_composite(arrays):
    """
    Temporal median composite from multiple scenes.
    Zeros (SCL-masked pixels) are treated as NaN so they don't
    pull the median toward zero in partially-cloudy areas.
    With MAX_SCENES=10, the median has far more cloud-free
    observations to draw from per pixel.
    """
    if len(arrays) == 1:
        return arrays[0]

    target_h, target_w = arrays[0].shape[:2]
    aligned = []
    for arr in arrays:
        if arr.shape[:2] != (target_h, target_w):
            rh = np.linspace(0, arr.shape[0] - 1, target_h).astype(int)
            rw = np.linspace(0, arr.shape[1] - 1, target_w).astype(int)
            arr = arr[np.ix_(rh, rw)]
        aligned.append(arr)

    stack  = np.stack(aligned, axis=0).astype(np.float32)   # (N, H, W, 6)
    masked = np.where(stack < 0.001, np.nan, stack)         # mask zeros → NaN
    comp   = np.nanmedian(masked, axis=0)                   # (H, W, 6)
    comp   = np.nan_to_num(comp, nan=0.0)                   # remaining NaN → 0
    print(f"  Composite built from {len(aligned)} scene(s) → shape {comp.shape}")
    return comp.astype(np.float32)


def download_composite(token, aoi, scenes):
    """
    Download up to MAX_SCENES scenes (sorted by cloud cover)
    and return their temporal median composite array.
    FIX 1: MAX_SCENES raised to 10 — more scenes = better
    cloud-free median for difficult months.
    """
    scenes = scenes[:MAX_SCENES]
    arrays = []
    for scene in scenes:
        scene_date = get_scene_date(scene)
        try:
            arr = download_image(token, aoi, scene_date)
            arrays.append(arr)
        except Exception as e:
            print(f"  ⚠️  Scene {scene_date} failed: {e} — skipping")
    if not arrays:
        raise RuntimeError("All scene downloads failed for this period")
    return build_composite(arrays)


def valid_pixel_percentage(arr):
    valid = np.any(arr[:, :, :6] > 0.01, axis=2)
    return float(valid.mean() * 100)


def haze_score(arr):
    """
    Haze index = median(blue) / median(NIR).
    Elevated blue relative to NIR indicates haze or thin cloud.
    FIX 2: threshold raised to 0.42 to account for dry-season
    smoke common in Cambodia (previously 0.35 was too strict).
    """
    blue = arr[:, :, 0]
    nir  = arr[:, :, 3]
    valid = (blue > 0.001) & (nir > 0.001)
    if valid.sum() < 100:
        return 0.0
    return float(np.median(blue[valid]) / (np.median(nir[valid]) + 1e-6))


def quality_check(before_arr, after_arr):
    """
    Returns (passed: bool, reason: str).
    Three checks:
      1. Both composites >= MIN_VALID_PCT valid pixels
      2. Valid-pixel gap <= MAX_VALID_DELTA
         FIX 3: raised from 20 → 30 so pairs like 77%/100%
         are no longer rejected (both are individually clean)
      3. Neither composite has haze index > MAX_HAZE_INDEX
    """
    b_valid = valid_pixel_percentage(before_arr)
    a_valid = valid_pixel_percentage(after_arr)
    print(f"  Valid pixels → Before: {b_valid:.1f}% | After: {a_valid:.1f}%")

    if b_valid < MIN_VALID_PCT:
        return False, f"Before valid {b_valid:.1f}% < {MIN_VALID_PCT}%"
    if a_valid < MIN_VALID_PCT:
        return False, f"After valid {a_valid:.1f}% < {MIN_VALID_PCT}%"
    if abs(b_valid - a_valid) > MAX_VALID_DELTA:
        return False, f"Valid-pixel gap {abs(b_valid - a_valid):.1f}% > {MAX_VALID_DELTA}%"

    b_haze = haze_score(before_arr)
    a_haze = haze_score(after_arr)
    print(f"  Haze index   → Before: {b_haze:.3f} | After: {a_haze:.3f}")

    if b_haze > MAX_HAZE_INDEX:
        return False, f"Before haze {b_haze:.3f} > {MAX_HAZE_INDEX}"
    if a_haze > MAX_HAZE_INDEX:
        return False, f"After haze {a_haze:.3f} > {MAX_HAZE_INDEX}"

    return True, "OK"


def fill_masked(band):
    mask = band == 0
    if mask.all():
        return np.full_like(band, 0.1)
    band = band.copy()
    band[mask] = np.nanmedian(band[~mask])
    return band

def compute_nbr(arr):
    b08 = fill_masked(arr[:, :, 3].astype(np.float32))
    b12 = fill_masked(arr[:, :, 5].astype(np.float32))
    denom = b08 + b12
    denom[denom == 0] = 1e-6
    return (b08 - b12) / denom

def compute_ndvi(arr):
    b08 = fill_masked(arr[:, :, 3].astype(np.float32))
    b04 = fill_masked(arr[:, :, 2].astype(np.float32))
    denom = b08 + b04
    denom[denom == 0] = 1e-6
    return (b08 - b04) / denom

def unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(img.astype(np.float64), sigma=sigma)
    sharpened = img.astype(np.float64) + strength * (img.astype(np.float64) - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

def make_rgb(arr):
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    for i, b_idx in enumerate([2, 1, 0]):
        band = fill_masked(arr[:, :, b_idx].copy())
        valid = band[band > 0.001]
        if len(valid) < 100:
            rgb[:, :, i] = 0.3
            continue
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    return unsharp_mask(rgb)

BG = "#0D1B2A"

def colorbar(fig, ax, im):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

def save_burn_scar(before, after, name, label):
    nbr_b = compute_nbr(before)
    nbr_a = compute_nbr(after)
    dnbr  = nbr_b - nbr_a
    mask  = dnbr > 0.27
    area  = np.sum(mask) * 100 / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, [
        (nbr_b, "NBR — BEFORE",     "RdYlGn",   -1,   1),
        (nbr_a, "NBR — AFTER",      "RdYlGn",   -1,   1),
        (dnbr,  "dNBR — BURN SCAR", "RdYlGn_r", -0.5, 0.5),
    ]):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        colorbar(fig, ax, im)
    axes[2].contour(mask, levels=[0.5], colors=["#F4A261"], linewidths=2)
    axes[2].text(0.02, 0.02, f"Burned pixels: {np.sum(mask):,}\nEst. area: {area:.3f} km²\ndNBR threshold: 0.27",
        transform=axes[2].transAxes, color="white", fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1B4332", alpha=0.85))
    plt.suptitle(f"Burn Scar — {name.replace('_',' ').title()}  [{label}]", color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/nbr/{name}_{label}_burn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  Burn scar saved: {path}  ({area:.3f} km²)")
    return area

def save_forest_clearance(before, after, name, label):
    ndvi_b = compute_ndvi(before)
    ndvi_a = compute_ndvi(after)
    dndvi  = ndvi_b - ndvi_a
    mask   = dndvi > 0.15
    area   = np.sum(mask) * 100 / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, [
        (ndvi_b, "NDVI — BEFORE",     "RdYlGn",   -1,   1),
        (ndvi_a, "NDVI — AFTER",      "RdYlGn",   -1,   1),
        (dndvi,  "dNDVI — CLEARANCE", "RdYlGn_r", -0.3, 0.3),
    ]):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        colorbar(fig, ax, im)
    axes[2].contour(mask, levels=[0.5], colors=["#F4A261"], linewidths=2)
    axes[2].text(0.02, 0.02, f"Cleared pixels: {np.sum(mask):,}\nEst. area: {area:.3f} km²\ndNDVI threshold: 0.15",
        transform=axes[2].transAxes, color="white", fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1B4332", alpha=0.85))
    plt.suptitle(f"Forest Clearance — {name.replace('_',' ').title()}  [{label}]", color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/ndvi/{name}_{label}_clearance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  Clearance saved: {path}  ({area:.3f} km²)")
    return area

def save_true_color(before, after, name, label):
    rgb_before = make_rgb(before)
    rgb_after  = make_rgb(after)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax1.imshow(rgb_before)
    ax1.set_title("TRUE COLOR — BEFORE", color="white", fontsize=13, fontweight="bold")
    ax1.axis("off")
    ax2.imshow(rgb_after)
    ax2.set_title("TRUE COLOR — AFTER", color="white", fontsize=13, fontweight="bold")
    ax2.axis("off")
    plt.suptitle(f"Visual — {name.replace('_',' ').title()}  [{label}]",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/truecolor/{name}_{label}_rgb.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  True color saved: {path}")

def save_raw(arr, name, label, suffix):
    folder = f"{OUTPUT_DIR}/raw/{name}"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{label}_{suffix}_composite.npy"
    np.save(path, arr)
    print(f"  Raw NPY saved: {path}")

def process_hotspot(hotspot, date_pair):
    token = get_access_token()
    name, lon_min, lat_min, lon_max, lat_max = hotspot
    d_bf, d_bt, d_af, d_at, label = date_pair
    aoi = [lon_min, lat_min, lon_max, lat_max]

    # Skip pairs already downloaded (resume support)
    before_path = f"{OUTPUT_DIR}/raw/{name}/{label}_before_composite.npy"
    after_path  = f"{OUTPUT_DIR}/raw/{name}/{label}_after_composite.npy"
    if os.path.exists(before_path) and os.path.exists(after_path):
        print(f"  ⏭️  Already exists — skipping {name} [{label}]")
        return None

    print(f"\n{'='*80}")
    print(f"  {name}  [{label}]")
    print(f"  Before: {d_bf}-{d_bt}  |  After: {d_af}-{d_at}")
    print(f"{'='*80}")

    before_scenes = search_catalog(d_bf, d_bt, aoi)
    after_scenes  = search_catalog(d_af, d_at, aoi)

    if not before_scenes or not after_scenes:
        print("  Skipping — missing scenes")
        return None

    print(f"  Building BEFORE composite ({min(len(before_scenes), MAX_SCENES)} scene(s))...")
    before_arr = download_composite(token, aoi, before_scenes)

    print(f"  Building AFTER composite ({min(len(after_scenes), MAX_SCENES)} scene(s))...")
    after_arr = download_composite(token, aoi, after_scenes)

    passed, reason = quality_check(before_arr, after_arr)
    if not passed:
        print(f"  QUALITY FAIL -> {reason}")
        return None

    print("  Quality OK — saving pair")

    save_raw(before_arr, name, label, "before")
    save_raw(after_arr,  name, label, "after")

    burn_area  = save_burn_scar(before_arr, after_arr, name, label)
    clear_area = save_forest_clearance(before_arr, after_arr, name, label)
    save_true_color(before_arr, after_arr, name, label)

    return {
        "hotspot":          name,
        "label":            label,
        "burn_km2":         burn_area,
        "clear_km2":        clear_area,
        "before_valid_pct": valid_pixel_percentage(before_arr),
        "after_valid_pct":  valid_pixel_percentage(after_arr),
    }

# ====================== RUN ======================
if __name__ == "__main__":
    import pandas as pd

    ANALYSIS_CSV = r"D:\MCL\sentinel_outputs\analysis\priority_sorted.csv"
    df_csv = pd.read_csv(ANALYSIS_CSV)

    top55_keys = set(
        df_csv.nlargest(55, ["glad_high", "glad_total"])["hotspot_period"]
    )
    df_sorted     = df_csv.sort_values(["glad_high", "glad_total"], ascending=True)
    bottom70_keys = set(
        df_sorted[~df_sorted["hotspot_period"].isin(top55_keys)].head(70)["hotspot_period"]
    )

    print(f"  NEGATIVE mode — bottom 70 pairs (excl. top 55 positives)")
    print(f"  Selected {len(bottom70_keys)} pairs\n")
    neg_lookup = {}
    for key in bottom70_keys:
       
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            h_name, mon, evt = parts
            period = f"{mon}_{evt}"
            neg_lookup.setdefault(h_name, []).append(period)

    results = []
    hotspot_map = {h[0]: h for h in HOTSPOTS}

    for h_name, periods in sorted(neg_lookup.items()):
        hotspot = hotspot_map.get(h_name)
        if hotspot is None:
            print(f"  WARNING: {h_name} not in HOTSPOTS list")
            continue
        for period in sorted(periods):
            # Find matching DATE_PAIRS entry by label
            date_pair = next((dp for dp in DATE_PAIRS if dp[4] == period), None)
            if date_pair is None:
                print(f"  WARNING: no DATE_PAIRS entry for period {period}")
                continue
            try:
                result = process_hotspot(hotspot, date_pair)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {h_name} {period}: {e}")
                continue

    print(f"\n{'='*80}")
    print(f"  DONE — {len(results)} pairs saved to {OUTPUT_DIR}/")
    print(f"{'='*80}")