import os
import json
import time
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from scipy.ndimage import binary_dilation
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")


FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")
FIRMS_URL      = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
RAW_NPY_DIR    = "sentinel_outputs/raw"
OUTPUT_DIR     = "sentinel_outputs/firms_false"
PATCH_SIZE     = 224
PATCH_STRIDE   = 112
MIN_PATCH_VALID_FRAC = 0.70
NBR_BURN_THRESH      = 0.27
NBR_POS_FRAC         = 0.05
FIRMS_POS_FRAC       = 0.01

VIIRS_DILATION_PX = 19
MODIS_DILATION_PX = 50

SATELLITES = [
    ("VIIRS_SNPP_SP",   VIIRS_DILATION_PX),
    ("VIIRS_NOAA20_SP", VIIRS_DILATION_PX),
    ("MODIS_SP",        MODIS_DILATION_PX),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
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


def date_chunks(start_str, end_str, chunk_days=5):
    start  = datetime.strptime(start_str, "%Y-%m-%d")
    end    = datetime.strptime(end_str,   "%Y-%m-%d")
    chunks = []
    cur    = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), (chunk_end - cur).days + 1))
        cur = chunk_end + timedelta(days=1)
    return chunks


def fetch_one_satellite(satellite, bbox_str, start, end):
    frames = []
    for chunk_start, n_days in date_chunks(start, end):
        url = f"{FIRMS_URL}/{FIRMS_MAP_KEY}/{satellite}/{bbox_str}/{n_days}/{chunk_start}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                if len(text) > 60 and not text.startswith("<?xml"):
                    df = pd.read_csv(StringIO(text))
                    if len(df) > 0:
                        frames.append(df)
                        print(f"      {chunk_start} ({n_days}d) → {len(df)} alerts")
                    else:
                        print(f"      {chunk_start} ({n_days}d) → 0 alerts")
                else:
                    print(f"      {chunk_start} ({n_days}d) → empty")
            else:
                print(f"      {chunk_start} ({n_days}d) → HTTP {resp.status_code}")
        except Exception as e:
            print(f"      {chunk_start} ({n_days}d) → error: {e}")
        time.sleep(0.4)
    return pd.concat(frames, ignore_index=True) if frames else None


def fetch_firms(name, lon_min, lat_min, lon_max, lat_max, start, end, label):
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    combined = pd.DataFrame()
    used_dilation = VIIRS_DILATION_PX

    for satellite, dilation_px in SATELLITES:
        print(f"    Trying {satellite}...")
        df = fetch_one_satellite(satellite, bbox_str, start, end)
        if df is not None and len(df) > 0:
            df["satellite"]   = satellite
            df["dilation_px"] = dilation_px
            combined          = df
            used_dilation     = dilation_px
            print(f"    Got {len(combined)} alerts from {satellite}")
            break
        print(f"    No data from {satellite}, trying next...")

    combined["hotspot"] = name
    combined["period"]  = label
    out_path = os.path.join(OUTPUT_DIR, f"firms_raw_{name}_{label}.csv")
    combined.to_csv(out_path, index=False)
    return combined, used_dilation


def rasterize_firms(firms_df, lon_min, lat_min, lon_max, lat_max,
                    H, W, dilation_px, confidence_filter=True):
    mask = np.zeros((H, W), dtype=np.uint8)
    if firms_df.empty:
        return mask

    df = firms_df.copy()

    if confidence_filter and "confidence" in df.columns:
        conf = df["confidence"].astype(str).str.lower()
        high = conf.isin(["h", "high", "n", "nominal"] +
                         [str(v) for v in range(80, 101)])
        df = df[high]
        if df.empty:
            return mask

    lon_col = "longitude" if "longitude" in df.columns else "LONGITUDE"
    lat_col = "latitude"  if "latitude"  in df.columns else "LATITUDE"

    for _, row in df.iterrows():
        lon, lat = row[lon_col], row[lat_col]
        px = int((lon - lon_min) / (lon_max - lon_min) * W)
        py = int((lat_max - lat) / (lat_max - lat_min) * H)
        if 0 <= px < W and 0 <= py < H:
            mask[py, px] = 1

    if dilation_px > 0 and mask.any():
        r      = dilation_px
        d      = 2 * r + 1
        struct = np.ones((d, d), dtype=bool)
        mask   = binary_dilation(mask, structure=struct).astype(np.uint8)

    return mask


def safe_band(arr, idx):
    b = arr[:, :, idx].astype(np.float32)
    b[b == 0] = np.nan
    return b


def compute_nbr(arr):
    b08   = safe_band(arr, 3)
    b12   = safe_band(arr, 5)
    denom = np.nansum([b08, b12], axis=0)
    denom[denom == 0] = 1e-6
    return (b08 - b12) / denom


def compute_dnbr_mask(before, after, threshold=NBR_BURN_THRESH):
    dnbr = compute_nbr(before) - compute_nbr(after)
    return (dnbr > threshold).astype(np.uint8), dnbr


def load_npy_stacks(name):
    raw_dir = os.path.join(RAW_NPY_DIR, name)
    if not os.path.isdir(raw_dir):
        return None, None

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".npy"))
    before_files = [f for f in files if "_before_" in f]
    after_files  = [f for f in files if "_after_"  in f]

    if not before_files or not after_files:
        if len(files) < 2:
            return None, None
        before_files = files[:len(files)//2]
        after_files  = files[len(files)//2:]

    before = np.load(os.path.join(raw_dir, before_files[-1])).astype(np.float32)
    after  = np.load(os.path.join(raw_dir, after_files[-1])).astype(np.float32)

    for arr, tag in [(before, "before"), (after, "after")]:
        if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
            arr = np.moveaxis(arr, 0, -1)

    return before, after


def assign_confidence(nbr_pos, firms_pos):
    if nbr_pos and firms_pos:
        return 1, "high"
    elif nbr_pos and not firms_pos:
        return 1, "uncertain"
    elif not nbr_pos and firms_pos:
        return 1, "firms_only"
    else:
        return 0, "high"


def extract_patches(before, after, dnbr_mask, firms_mask,
                    hotspot, label, lon_min, lat_min, lon_max, lat_max):
    H, W = before.shape[:2]
    patch_dir = os.path.join(OUTPUT_DIR, "patches", f"{hotspot}_{label}")
    os.makedirs(patch_dir, exist_ok=True)

    records = []

    for row_idx, y in enumerate(range(0, H - PATCH_SIZE + 1, PATCH_STRIDE)):
        for col_idx, x in enumerate(range(0, W - PATCH_SIZE + 1, PATCH_STRIDE)):

            b_patch     = before[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            a_patch     = after [y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            nbr_patch   = dnbr_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            firms_patch = firms_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            valid_frac = float(np.mean(np.isfinite(b_patch) & (b_patch != 0)))
            if valid_frac < MIN_PATCH_VALID_FRAC:
                continue

            nbr_frac   = float(nbr_patch.mean())
            firms_frac  = float(firms_patch.mean())
            nbr_pos    = nbr_frac  > NBR_POS_FRAC
            firms_pos   = firms_frac > FIRMS_POS_FRAC

            label_val, confidence = assign_confidence(nbr_pos, firms_pos)

            combined_mask = np.clip(nbr_patch + firms_patch, 0, 1).astype(np.uint8)

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
                "nbr_frac":    round(nbr_frac, 4),
                "firms_frac":  round(firms_frac, 4),
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
    print(f"    Patches: {len(records)} total  |  {pos} positive  |  {neg} negative")
    return records


def compute_dataset_stats(manifest):
    print("\n  Computing dataset normalization stats...")
    band_sums   = np.zeros(6)
    band_sq     = np.zeros(6)
    pixel_count = 0

    train_paths = [r["before_path"] for r in manifest
                   if r.get("split") == "train" and os.path.exists(r["before_path"])]

    if not train_paths:
        print("  WARNING: no training patches found for stats — using all patches")
        train_paths = [r["before_path"] for r in manifest
                       if os.path.exists(r["before_path"])]

    for path in train_paths:
        arr    = np.load(path).astype(np.float32)
        finite = np.isfinite(arr)
        for b in range(6):
            vals = arr[b][finite[b]]
            band_sums[b] += vals.sum()
            band_sq[b]   += (vals ** 2).sum()
            pixel_count   = max(pixel_count, len(vals))

    n     = len(train_paths) * PATCH_SIZE * PATCH_SIZE
    means = band_sums / n
    stds  = np.sqrt(band_sq / n - means ** 2)

    stats = {
        "bands":      ["B02", "B03", "B04", "B08", "B11", "B12"],
        "mean":       means.tolist(),
        "std":        stds.tolist(),
        "n_patches":  len(train_paths),
        "patch_size": PATCH_SIZE,
    }
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved → {stats_path}")
    print(f"  Band means : {np.round(means, 4)}")
    print(f"  Band stds  : {np.round(stds,  4)}")
    return stats


def assign_splits(manifest, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)

    hotspots = list({r["hotspot"] for r in manifest})
    rng.shuffle(hotspots)
    n      = len(hotspots)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    val_set  = set(hotspots[:n_val])
    test_set = set(hotspots[n_val:n_val + n_test])

    for r in manifest:
        hs = r["hotspot"]
        if hs in test_set:
            r["split"] = "test"
        elif hs in val_set:
            r["split"] = "val"
        else:
            r["split"] = "train"

    return manifest


def compute_scene_metrics(name, label, firms_df, before, after):
    if before is None:
        return None

    _, dnbr_mask = compute_dnbr_mask(before, after)
    nbr_px       = int(np.sum(dnbr_mask > NBR_BURN_THRESH))
    nbr_detected = nbr_px > 0

    if firms_df.empty:
        firms_count    = 0
        firms_detected = False
    else:
        if "confidence" in firms_df.columns:
            conf = firms_df["confidence"].astype(str).str.lower()
            high = firms_df[conf.isin(["h", "high", "n", "nominal"] +
                                      [str(v) for v in range(80, 101)])]
            firms_count    = len(high)
            firms_detected = firms_count > 0
        else:
            firms_count    = len(firms_df)
            firms_detected = firms_count > 0

    tp   = int(nbr_detected and firms_detected)
    fp   = int(nbr_detected and not firms_detected)
    fn   = int(not nbr_detected and firms_detected)
    tn   = int(not nbr_detected and not firms_detected)
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec  = tp / (tp + fn) if (tp + fn) > 0 else None
    f1   = (2 * prec * rec / (prec + rec)) if (prec and rec) else None

    return {
        "hotspot":        name,
        "period":         label,
        "nbr_detected":   nbr_detected,
        "nbr_pixels":     nbr_px,
        "firms_alerts":   firms_count,
        "firms_detected": firms_detected,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 3) if prec is not None else "N/A",
        "recall":    round(rec,  3) if rec  is not None else "N/A",
        "f1":        round(f1,   3) if f1   is not None else "N/A",
        "agreement": "YES" if (nbr_detected == firms_detected) else "NO",
    }


BG = "#0D1B2A"

def plot_summary(summary_df, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    totals = summary_df.groupby("hotspot")["firms_alerts"].sum().sort_values()
    short  = [h.replace("hotspot_", "").replace("_", " ") for h in totals.index]
    colors = ["#E76F51" if v > 0 else "#457B9D" for v in totals.values]
    bars   = ax1.barh(short, totals.values, color=colors, height=0.6)
    ax1.set_xlabel("FIRMS alert count", color="white")
    ax1.set_title("FIRMS alerts per hotspot", color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=7)
    ax1.spines[:].set_color("#444")
    for bar, val in zip(bars, totals.values):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", color="white", fontsize=7)
    ax1.legend(handles=[
        mpatches.Patch(color="#E76F51", label="Alerts found"),
        mpatches.Patch(color="#457B9D", label="No alerts"),
    ], facecolor="#1B2838", labelcolor="white", fontsize=8)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    cats   = {"TP": "#2A9D8F", "FP": "#E9C46A", "FN": "#E76F51", "TN": "#457B9D"}
    bottom = np.zeros(len(summary_df))
    xlabels = [f"{r['hotspot'].replace('hotspot_','')[:14]}[{r['period'][:3]}]"
               for _, r in summary_df.iterrows()]
    for cat, color in cats.items():
        vals = summary_df[cat].values.astype(float)
        ax2.bar(range(len(summary_df)), vals, bottom=bottom, color=color, label=cat, width=0.7)
        bottom += vals
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels(xlabels, rotation=60, ha="right", color="white", fontsize=6)
    ax2.set_ylabel("Count", color="white")
    ax2.set_title("NBR vs FIRMS agreement", color="white", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")
    ax2.legend(facecolor="#1B2838", labelcolor="white", fontsize=8)

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    if manifest:
        mdf      = pd.DataFrame(manifest)
        conf_cts = mdf.groupby(["label", "confidence"]).size().reset_index(name="count")
        bar_labels, bar_vals, bar_colors = [], [], []
        for (lbl, conf), color, name_str in [
            ((1, "high"),       "#2A9D8F", "Pos / High"),
            ((1, "uncertain"),  "#E9C46A", "Pos / Uncertain"),
            ((1, "firms_only"), "#F4A261", "Pos / FIRMS only"),
            ((0, "high"),       "#457B9D", "Neg / High"),
        ]:
            row = conf_cts[(conf_cts["label"] == lbl) & (conf_cts["confidence"] == conf)]
            ct  = int(row["count"].values[0]) if len(row) else 0
            bar_labels.append(name_str)
            bar_vals.append(ct)
            bar_colors.append(color)

        bars3 = ax3.bar(bar_labels, bar_vals, color=bar_colors, width=0.5)
        ax3.set_ylabel("Patch count", color="white")
        ax3.set_title("Patch label distribution", color="white", fontsize=11, fontweight="bold")
        ax3.tick_params(colors="white", labelsize=8)
        ax3.spines[:].set_color("#444")
        for bar, val in zip(bars3, bar_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, str(val),
                     ha="center", color="white", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "validation_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"\n  Chart saved → {path}")


def main():
    print("=" * 65)
    print("  FIRMS VALIDATION v2 — Siamese CNN edition")
    print("=" * 65)

    all_scene_results = []
    all_patches       = []

    for name, lon_min, lat_min, lon_max, lat_max in HOTSPOTS:
        print(f"\n{'─'*65}")
        print(f"  {name}")
        print(f"{'─'*65}")

        before, after = load_npy_stacks(name)
        if before is None:
            print("  No NPY stacks found — skipping patch extraction")

        if before is not None:
            H, W = before.shape[:2]
        else:
            km_x = (lon_max - lon_min) * 111.32 * np.cos(np.radians((lat_min + lat_max) / 2))
            km_y = (lat_max - lat_min) * 110.574
            H    = max(1, int(km_y * 1000 / 10))
            W    = max(1, int(km_x * 1000 / 10))

        for bf_s, bf_e, start, end, label in DATE_PAIRS:
            print(f"\n  [{label}]  {start} → {end}")

            firms_df, dilation_px = fetch_firms(
                name, lon_min, lat_min, lon_max, lat_max, start, end, label)

            scene_result = compute_scene_metrics(name, label, firms_df, before, after)
            if scene_result:
                all_scene_results.append(scene_result)

            if before is not None:
                firms_mask   = rasterize_firms(
                    firms_df, lon_min, lat_min, lon_max, lat_max, H, W, dilation_px)
                dnbr_mask, _ = compute_dnbr_mask(before, after)

                patches = extract_patches(
                    before, after, dnbr_mask, firms_mask,
                    name, label, lon_min, lat_min, lon_max, lat_max)
                all_patches.extend(patches)

    all_patches = assign_splits(all_patches)

    manifest_df   = pd.DataFrame(all_patches)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  Manifest → {manifest_path}  ({len(manifest_df)} patches)")

    if all_patches:
        compute_dataset_stats(all_patches)

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

    if all_patches:
        mdf = pd.DataFrame(all_patches)
        print("\n" + "=" * 65)
        print("  PATCH SUMMARY")
        print("=" * 65)
        for split in ["train", "val", "test"]:
            sub = mdf[mdf["split"] == split]
            pos = (sub["label"] == 1).sum()
            neg = (sub["label"] == 0).sum()
            hc  = (sub["confidence"] == "high").sum()
            unc = (sub["confidence"] == "uncertain").sum()
            print(f"  {split:6s}: {len(sub):5d} patches  |  "
                  f"pos={pos}  neg={neg}  "
                  f"high-conf={hc}  uncertain={unc}")
        print(f"\n  Outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()