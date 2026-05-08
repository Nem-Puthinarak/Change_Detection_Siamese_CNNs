import os
import re
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIRS     = [
    r"D:\MCL\sentinel_outputs\raw",
    r"D:\MCL\sentinel-outputs-neg\raw",
]
GLAD_DIR     = r"D:\MCL\sentinel_outputs\glad\alerts_raw"
FIRMS_DIR    = r"D:\MCL\sentinel_outputs\firms"
PATCH_DIR    = r"D:\MCL\sentinel_outputs\patches"
ANALYSIS_CSV = r"D:\MCL\sentinel_outputs\analysis\priority_sorted.csv"

PATCH_SIZE     = 128
PATCH_STRIDE   = 64
MIN_VALID_FRAC = 0.75
GLAD_BUFFER_PX = 25
FIRMS_BUFFER_PX = 30

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(PATCH_DIR, exist_ok=True)
CLASSES = ["0_no_change", "1_clearance", "2_fire", "3_mixed"]
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{PATCH_DIR}/{split}/{cls}", exist_ok=True)

glad_alerts = {}
if os.path.isdir(GLAD_DIR):
    for file in os.listdir(GLAD_DIR):
        if not file.endswith(".json"):
            continue
        key = file.replace(".json", "")
        try:
            with open(os.path.join(GLAD_DIR, file)) as f:
                data = json.load(f)
            if data:
                glad_alerts[key] = pd.DataFrame(data)
        except Exception as e:
            print(f"  ⚠️  Could not load GLAD {file}: {e}")

print(f"Loaded GLAD alerts for {len(glad_alerts)} pairs")


firms_alerts = {}
if os.path.isdir(FIRMS_DIR):
    for file in os.listdir(FIRMS_DIR):
        if not file.endswith(".csv"):
            continue
        key = file.replace("firms_raw_", "").replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(FIRMS_DIR, file))
            if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
                firms_alerts[key] = df
        except Exception as e:
            print(f"  ⚠️  Could not load FIRMS {file}: {e}")

print(f"Loaded FIRMS alerts for {len(firms_alerts)} pairs")

HOTSPOT_BOUNDS = {
    "hotspot_01_osmach_west":            (103.630, 14.410, 103.680, 14.450),
    "hotspot_02_o3chean_village":        (102.720, 13.765, 102.765, 13.810),
    "hotspot_03_phnom_proek":            (102.313, 13.295, 102.403, 13.385),
    "hotspot_04_poipet_thai":            (102.530, 13.653, 102.575, 13.688),
    "hotspot_05_poipet_road":            (102.690, 13.645, 102.735, 13.680),
    "hotspot_06_osmach_east":            (103.685, 14.408, 103.730, 14.445),
    "hotspot_07_choam_chom_sngan":       (104.040, 14.330, 104.085, 14.360),
    "hotspot_08_bos_sbov":               (104.730, 14.412, 104.775, 14.445),
    "hotspot_09_chok_chey":              (102.665, 13.745, 102.720, 13.780),
    "hotspot_10_osmach_jan2026":         (103.645, 14.400, 103.695, 14.445),
    "hotspot_11_an_ses":                 (104.942, 14.371, 104.987, 14.406),
    "hotspot_12_ta_moan_thom":           (103.243, 14.330, 103.288, 14.365),
    "hotspot_13_ta_krabei":              (103.351, 14.335, 103.396, 14.380),
    "hotspot_14_preahvihear_thai":       (104.657, 14.381, 104.702, 14.416),
    "hotspot_15_preahvihear_temple":     (104.657, 14.374, 104.702, 14.409),
    "hotspot_16_thmor_da":               (102.712, 12.114, 102.757, 12.150),
    "hotspot_17_prei_chan":              (102.714, 13.800, 102.759, 13.835),
    "hotspot_18_beoung_trakoun":         (102.830, 13.972, 102.877, 14.004),
    "hotspot_19_thmor_doun":             (103.269, 14.315, 103.359, 14.405),
    "hotspot_20_emerald_triangle":       (105.160, 14.299, 105.205, 14.350),
    "hotspot_21_phnom_trob":             (104.622, 14.368, 104.687, 14.433),
    "hotspot_22_cham_yeam":              (102.893, 11.627, 102.948, 11.682),
    "hotspot_23_koh_kong":               (102.801, 11.830, 102.879, 11.896),
    "hotspot_24_botum_sakor":            (102.698, 12.213, 102.753, 12.273),
    "hotspot_25_Area_400":               (102.661, 12.512, 102.714, 12.565),
    "hotspot_26_South-wing-Area400":     (102.674, 12.503, 102.746, 12.560),
    "hotspot_27_Sompov_lun":             (102.315, 13.394, 102.402, 13.481),
    "hotspot_28_ta_moan_thom_mountain":  (103.250, 14.318, 103.318, 14.386),
    "hotspot_29_ta_krabei_mountain":     (103.352, 14.320, 103.420, 14.388),
    "hotspot_30_Anses_mountain":         (104.950, 14.363, 104.994, 14.407),
    "hotspot_31_Anses_mountain_south":   (104.920, 14.353, 104.982, 14.415),
}

def build_hotspot_splits(hotspot_names):
    names = sorted(hotspot_names)
    random.shuffle(names)
    n = len(names)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end   = train_end + int(n * SPLIT_RATIOS["val"])
    split_map = {}
    for i, name in enumerate(names):
        if i < train_end:
            split_map[name] = "train"
        elif i < val_end:
            split_map[name] = "val"
        else:
            split_map[name] = "test"
    return split_map

def latlon_to_pixel(lat, lon, bounds, img_h, img_w):
    lon_min, lat_min, lon_max, lat_max = bounds
    px = int((lon - lon_min) / (lon_max - lon_min) * img_w)
    py = int((lat_max - lat) / (lat_max - lat_min) * img_h)
    return px, py


def build_alert_mask(img_h, img_w, bounds, alert_df, buffer_px):
    mask = np.zeros((img_h, img_w), dtype=bool)
    if alert_df is None or alert_df.empty:
        return mask
    if "latitude" not in alert_df.columns or "longitude" not in alert_df.columns:
        return mask
    for _, row in alert_df.iterrows():
        px, py = latlon_to_pixel(row["latitude"], row["longitude"],
                                  bounds, img_h, img_w)
        x0 = max(0, px - buffer_px); x1 = min(img_w, px + buffer_px)
        y0 = max(0, py - buffer_px); y1 = min(img_h, py + buffer_px)
        mask[y0:y1, x0:x1] = True
    return mask


def is_valid_patch(patch):
    valid = np.isfinite(patch) & (patch != 0)
    return np.mean(valid) >= MIN_VALID_FRAC

def extract_patches(hotspot_name, label, period, split, raw_dir):
    raw_folder = os.path.join(raw_dir, hotspot_name)
    if not os.path.isdir(raw_folder):
        print(f"  ⚠️  No raw folder: {hotspot_name}")
        return {cls: 0 for cls in CLASSES}

    before_path = os.path.join(raw_folder, f"{period}_before_composite.npy")
    after_path  = os.path.join(raw_folder, f"{period}_after_composite.npy")

    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print(f"  ⚠️  Missing npy: {hotspot_name} {period}")
        return {cls: 0 for cls in CLASSES}

    before = np.load(before_path).astype(np.float32)
    after  = np.load(after_path).astype(np.float32)

    if before.ndim == 3 and before.shape[0] < before.shape[2]:
        before = np.moveaxis(before, 0, -1)
        after  = np.moveaxis(after,  0, -1)

    H, W   = before.shape[:2]
    bounds = HOTSPOT_BOUNDS.get(hotspot_name)


    glad_mask  = np.zeros((H, W), dtype=bool)
    firms_mask = np.zeros((H, W), dtype=bool)

    if label == "positive" and bounds is not None:
        alert_key = f"{hotspot_name}_{period}"
        glad_df   = glad_alerts.get(alert_key)
        firms_df  = firms_alerts.get(alert_key)

        glad_mask  = build_alert_mask(H, W, bounds, glad_df,  GLAD_BUFFER_PX)
        firms_mask = build_alert_mask(H, W, bounds, firms_df, FIRMS_BUFFER_PX)

        combined_mask = glad_mask & firms_mask
        glad_cov  = glad_mask.mean()  * 100
        firms_cov = firms_mask.mean() * 100
        combo_cov = combined_mask.mean() * 100
        print(f"  Masks: GLAD={glad_cov:.1f}%  FIRMS={firms_cov:.1f}%  combined={combo_cov:.1f}%")

    # Collect and shuffle valid patch positions
    valid_positions = []
    for y in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            if is_valid_patch(before[y:y+PATCH_SIZE, x:x+PATCH_SIZE]) and \
               is_valid_patch(after [y:y+PATCH_SIZE, x:x+PATCH_SIZE]):
                valid_positions.append((y, x))

    random.shuffle(valid_positions)

    class_counts = {cls: 0 for cls in CLASSES}
    safe_name = hotspot_name.replace("-", "_")

    for y, x in valid_positions:
        b_patch = before[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        a_patch = after [y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        g = glad_mask [y:y+PATCH_SIZE, x:x+PATCH_SIZE].any()
        f = firms_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE].any()

        if g and f:
            patch_class = "3_mixed"
        elif g:
            patch_class = "1_clearance"
        elif f:
            patch_class = "2_fire"
        else:
            patch_class = "0_no_change"

        out_dir = os.path.join(PATCH_DIR, split, patch_class)
        fname   = f"{safe_name}__{period}__{y:04d}_{x:04d}"
        np.save(os.path.join(out_dir, f"{fname}__before.npy"),
                np.moveaxis(b_patch, -1, 0).astype(np.float32))
        np.save(os.path.join(out_dir, f"{fname}__after.npy"),
                np.moveaxis(a_patch, -1, 0).astype(np.float32))

        class_counts[patch_class] += 1

    counts_str = "  ".join(f"{c}={class_counts[c]}" for c in CLASSES)
    print(f"  {hotspot_name} [{period}] label={label} split={split} → {counts_str}  "
          f"total={len(valid_positions)}")
    return class_counts


def build_manifest_from_raw():
    manifest = []
    seen = set()
    for raw_dir in RAW_DIRS:
        if not os.path.isdir(raw_dir):
            print(f"  WARNING: RAW_DIR not found, skipping: {raw_dir}")
            continue
        for hotspot_name in os.listdir(raw_dir):
            folder = os.path.join(raw_dir, hotspot_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                m = re.match(r"(.+?)_before_composite\.npy", fname)
                if m:
                    period = m.group(1)
                    key = (hotspot_name, period)
                    if key not in seen:
                        seen.add(key)
                        manifest.append((hotspot_name, period, raw_dir))
    return manifest


if __name__ == "__main__":

    df_csv = pd.read_csv(ANALYSIS_CSV)

    top55_keys = set(
        df_csv.nlargest(55, ["glad_high", "glad_total"])["hotspot_period"]
    )
    df_sorted     = df_csv.sort_values(["glad_high", "glad_total"], ascending=True)
    bottom70_keys = set(
        df_sorted[~df_sorted["hotspot_period"].isin(top55_keys)].head(70)["hotspot_period"]
    )

    pair_labels = {}
    for k in top55_keys:    pair_labels[k] = "positive"
    for k in bottom70_keys: pair_labels[k] = "negative"

    print(f"Positive pairs: {len(top55_keys)}  |  Negative pairs: {len(bottom70_keys)}")

    manifest = build_manifest_from_raw()
    print(f"Found {len(manifest)} pairs across {len(RAW_DIRS)} raw directories")

    all_hotspots = sorted(set(h for h, _, _rd in manifest))
    hotspot_split_map = build_hotspot_splits(all_hotspots)
    print(f"\nHotspot split assignment:")
    for split in ["train", "val", "test"]:
        hs = [h for h, s in hotspot_split_map.items() if s == split]
        print(f"  {split}: {len(hs)} hotspots → {hs}")

    
    total_counts = {cls: 0 for cls in CLASSES}
    skipped = 0
    no_label_keys = []

    for hotspot_name, period, raw_dir in sorted(manifest):
        csv_key = f"{hotspot_name}_{period}"
        label   = pair_labels.get(csv_key)
        if label is None:
            no_label_keys.append(csv_key)
            skipped += 1
            continue
        split  = hotspot_split_map[hotspot_name]
        counts = extract_patches(hotspot_name, label, period, split, raw_dir)
        for cls in CLASSES:
            total_counts[cls] += counts[cls]

    print("\n" + "=" * 65)
    print("  PATCH EXTRACTION COMPLETE")
    for cls in CLASSES:
        print(f"  {cls:20s}: {total_counts[cls]}")
    print(f"  Skipped (no CSV label) : {skipped}")
    if no_label_keys:
        print(f"  Unmatched keys (first 10):")
        for k in no_label_keys[:10]:
            print(f"    {k}")
    print(f"  Output: {PATCH_DIR}")
    print("=" * 65)

    print("\nPer-split patch counts:")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            d = os.path.join(PATCH_DIR, split, cls)
            n = len([f for f in os.listdir(d) if f.endswith("__before.npy")])
            print(f"  {split}/{cls}: {n} patches")