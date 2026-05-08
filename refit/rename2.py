import os
import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI, no tkinter, safe for long background runs
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

CLIENT_ID     = "sh-287d79d1-8827-4147-959e-5d717b4179ee"
CLIENT_SECRET = "ZA9AioQj47UeTwaF0ZG8UKL1bKgZPM3f"
CDSE_USERNAME = "nrraiyujin@gmail.com"
CDSE_PASSWORD = "2WCt2fUFsV_yReQ"

RESOLUTION    = 10
MAX_CLOUD_PCT = 70
OUTPUT_DIR    = "sentinel_outputs"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL    = "https://sh.dataspace.copernicus.eu/api/v1/process"
STAC_URL  = "https://catalogue.dataspace.copernicus.eu/stac/search"

for folder in ["nbr", "ndvi", "truecolor", "truecolor/separate", "raw"]:
    os.makedirs(f"{OUTPUT_DIR}/{folder}", exist_ok=True)

HOTSPOTS = [
    ("hotspot_01_osmach_west",         103.630, 14.410, 103.680, 14.450),
    ("hotspot_02_o3chean_village",     102.720, 13.765, 102.765, 13.810),
    ("hotspot_03_phnom_proek",         102.339, 13.302, 102.376, 13.334),
    ("hotspot_04_poipet_thai",         102.530, 13.653, 102.575, 13.688),
    ("hotspot_05_poipet_road",         102.690, 13.645, 102.735, 13.680),
    ("hotspot_06_osmach_east",         103.685, 14.408, 103.730, 14.445),
    ("hotspot_07_choam_chom_sngan",    104.040, 14.330, 104.085, 14.360),
    ("hotspot_08_bos_sbov",            104.730, 14.412, 104.775, 14.445),
    ("hotspot_09_chok_chey",           102.665, 13.745, 102.720, 13.780),
    ("hotspot_10_osmach_jan2026",      103.645, 14.400, 103.695, 14.445),
    ("hotspot_11_an_ses",              104.942, 14.371, 104.987, 14.406),
    ("hotspot_12_ta_moan_thom",        103.243, 14.330, 103.288, 14.365),
    ("hotspot_13_ta_krabei",           103.351, 14.335, 103.396, 14.380),
    ("hotspot_14_preahvihear_thai",    104.657, 14.381, 104.702, 14.416),
    ("hotspot_15_preahvihear_temple",  104.657, 14.374, 104.702, 14.409),
    ("hotspot_16_thmor_da",            102.712, 12.114, 102.757, 12.150),
    ("hotspot_17_prei_chan",           102.714, 13.800, 102.759, 13.835),
    ("hotspot_18_beoung_trakoun",      102.845, 13.980, 102.862, 13.992),
    ("hotspot_19_thmor_doun",          103.304, 14.344, 103.348, 14.364),
    ("hotspot_20_emerald_triangle",    105.160, 14.299, 105.205, 14.350),
]

RUN_HOTSPOTS = HOTSPOTS

DATE_PAIRS = [
    ("2025-06-01", "2025-11-30", "2025-12-01", "2025-12-31", "Dec2025_conflict"),
    ("2025-12-01", "2025-12-15", "2026-01-20", "2026-03-15", "Jan2026_persistent"),
    ("2025-06-01", "2025-11-30", "2026-02-01", "2026-02-28", "Feb2026_baseline"),
    ("2025-06-01", "2025-11-30", "2026-03-01", "2026-03-31", "Mar2026_baseline"),
    ("2025-06-01", "2025-11-30", "2026-04-01", "2026-04-30", "Apr2026_baseline"),
    ("2025-03-01", "2025-04-20", "2025-04-21", "2025-05-25", "AprMay2025_early"),
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
  if ([9, 10].includes(s.SCL)) {
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
    resp = SESSION.post(SH_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "image/tiff"}, json=payload, timeout=(60, 600))
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed ({resp.status_code}): {resp.text[:200]}")
    arr = tifffile.imread(io.BytesIO(resp.content)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
    print(f"  Downloaded shape: {arr.shape}")
    return arr

def fill_masked(band):
    mask = band == 0
    if mask.all():
        return np.full_like(band, 0.1)
    if not mask.any():
        return band
    median_val = np.nanmedian(band[~mask])
    band = band.copy()
    band[mask] = median_val
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
    # Combined side-by-side (quick visual check)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax1.imshow(make_rgb(before))
    ax1.set_title("TRUE COLOR — BEFORE", color="white", fontsize=13, fontweight="bold")
    ax1.axis("off")
    ax2.imshow(make_rgb(after))
    ax2.set_title("TRUE COLOR — AFTER", color="white", fontsize=13, fontweight="bold")
    ax2.axis("off")
    plt.suptitle(f"Visual — {name.replace('_',' ').title()}  [{label}]", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/truecolor/{name}_{label}_rgb.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  True color saved: {path}")
def save_separate_rgb(before, after, name, label, before_date, after_date):
    """
    Save before and after as completely separate clean PNGs for manual annotation.
    No axes, no titles, no borders — just the raw image ready for Label Studio / CVAT / Roboflow.
    Saved to: truecolor/separate/{name}_{label}_before_{date}.png
                              {name}_{label}_after_{date}.png
    """
    before_rgb = make_rgb(before)
    after_rgb  = make_rgb(after)

    before_date_clean = before_date.replace("-", "")
    after_date_clean  = after_date.replace("-", "")

    from PIL import Image as PILImage, ImageDraw, ImageFont
    for rgb, tag, date_clean in [
        (before_rgb, "before", before_date_clean),
        (after_rgb,  "after",  after_date_clean),
    ]:
        out_path = f"{OUTPUT_DIR}/truecolor/separate/{name}_{label}_{tag}_{date_clean}.png"
        img_uint8 = (__import__('numpy').clip(rgb, 0, 1) * 255).astype(__import__('numpy').uint8)
        img = PILImage.fromarray(img_uint8)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, img.height // 20))
        except:
            font = ImageFont.load_default()
        color = (0, 220, 80) if tag == "before" else (255, 80, 80)
        lines = [
            tag.upper(),
            date_clean[:4] + "-" + date_clean[4:6] + "-" + date_clean[6:],
            name,
            label,
        ]
        x, y = 8, img.height - (len(lines) * (max(14, img.height // 20) + 4)) - 8
        for line in lines:
            draw.text((x+1, y+1), line, font=font, fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=color)
            y += max(14, img.height // 20) + 4
        img.save(out_path)
        print(f"  Separate RGB saved: {out_path}  ({rgb.width}x{rgb.height} px)")

def save_raw(arr, name, scene_date):
    folder = f"{OUTPUT_DIR}/raw/{name}"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{scene_date.replace('-','')}_stack.npy"
    np.save(path, arr)
    print(f"  Raw saved: {path}")

def process_hotspot(hotspot, date_pair):
    token = get_access_token()  # fresh token every hotspot/date pair
    name, lon_min, lat_min, lon_max, lat_max = hotspot
    d_bf, d_bt, d_af, d_at, label = date_pair
    aoi = [lon_min, lat_min, lon_max, lat_max]
    print(f"\n{'='*60}")
    print(f"  {name}  [{label}]")
    print(f"  Before: {d_bf} to {d_bt}  |  After: {d_af} to {d_at}")
    print(f"{'='*60}")
    before_scenes = search_catalog(d_bf, d_bt, aoi)
    after_scenes  = search_catalog(d_af, d_at, aoi)
    if not before_scenes:
        print("  No before scene found — skipping")
        return None
    if not after_scenes:
        print("  No after scene found — skipping")
        return None
    before_date = get_scene_date(before_scenes[0])
    after_date  = get_scene_date(after_scenes[0])
    print(f"  Best before: {before_date}  |  Best after: {after_date}")
    before_arr = download_image(token, aoi, before_date)
    after_arr  = download_image(token, aoi, after_date)
    save_raw(before_arr, name, before_date)
    save_raw(after_arr,  name, after_date)
    burn_area  = save_burn_scar(before_arr, after_arr, name, label)
    clear_area = save_forest_clearance(before_arr, after_arr, name, label)
    save_true_color(before_arr, after_arr, name, label)
    save_separate_rgb(before_arr, after_arr, name, label, before_date, after_date)  # <-- separate PNGs for annotation
    return {"hotspot": name, "label": label, "before": before_date, "after": after_date, "burn_km2": burn_area, "clear_km2": clear_area}

def main():
    print("=" * 60)
    print("  SENTINEL WATCH")
    print("=" * 60)
    results = []
    for hotspot in RUN_HOTSPOTS:
        for date_pair in DATE_PAIRS:
            try:
                result = process_hotspot(hotspot, date_pair)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {hotspot[0]}: {e}")
                continue
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['hotspot']}  [{r['label']}]")
        print(f"    Before: {r['before']}  After: {r['after']}")
        print(f"    Burn: {r['burn_km2']:.3f} km²  |  Cleared: {r['clear_km2']:.3f} km²")
    print(f"\n  Done. {len(results)} hotspot/date pairs processed.")
    print(f"  Outputs in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()