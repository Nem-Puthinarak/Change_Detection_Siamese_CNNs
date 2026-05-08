"""
sync_outputs.py
---------------
Reads the truecolor folder to get the list of passed pairs,
then deletes any raw .npy files whose pair is NOT represented
in truecolor. Also cleans up orphaned nbr/ndvi PNGs.

Usage:
    python sync_outputs.py
    python sync_outputs.py --dry-run        # preview only, no deletion
    python sync_outputs.py --output-dir D:\MCL\sentinel_outputs
"""

import os
import re
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="sentinel_outputs",
                   help="Root sentinel_outputs directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without deleting")
    return p.parse_args()


def get_passed_keys(truecolor_dir):
    """
    Scan truecolor folder for files like:
        hotspot_06_osmach_east_Nov2025_event_rgb.png
    and extract (hotspot_name, label) keys.
    """
    keys = set()
    if not os.path.isdir(truecolor_dir):
        print(f"❌ Truecolor directory not found: {truecolor_dir}")
        return keys

    for fname in os.listdir(truecolor_dir):
        if not fname.endswith("_rgb.png"):
            continue
        # Strip suffix _rgb.png → "hotspot_06_osmach_east_Nov2025_event"
        stem = fname[: -len("_rgb.png")]
        # Split off the label (last two underscore-joined tokens: e.g. Nov2025_event)
        # Label format is always <Month><Year>_event
        m = re.search(r"_([A-Za-z]+\d{4}_event)$", stem)
        if not m:
            print(f"  ⚠️  Could not parse label from: {fname}")
            continue
        label = m.group(1)
        hotspot = stem[: m.start()]   # everything before _<label>
        keys.add((hotspot, label))
        print(f"  ✅ Passed: {hotspot}  [{label}]")

    return keys


def build_npy_index(raw_dir):
    """
    Walk raw/<hotspot>/ subfolders and collect all .npy paths
    keyed by (hotspot, label).
    Returns dict: (hotspot, label) -> [path, ...]
    """
    index = {}
    if not os.path.isdir(raw_dir):
        print(f"❌ Raw directory not found: {raw_dir}")
        return index

    for hotspot_name in os.listdir(raw_dir):
        hotspot_path = os.path.join(raw_dir, hotspot_name)
        if not os.path.isdir(hotspot_path):
            continue
        for fname in os.listdir(hotspot_path):
            if not fname.endswith(".npy"):
                continue
            # Filename format: <label>_before_composite.npy or <label>_after_composite.npy
            m = re.match(r"([A-Za-z]+\d{4}_event)_(before|after)_composite\.npy", fname)
            if not m:
                print(f"  ⚠️  Unexpected npy filename: {fname}")
                continue
            label = m.group(1)
            key = (hotspot_name, label)
            index.setdefault(key, []).append(os.path.join(hotspot_path, fname))

    return index


def build_png_index(folder, suffix):
    """
    Collect nbr/ndvi PNG paths keyed by (hotspot, label).
    Filename format: <hotspot>_<label>_burn.png or _clearance.png
    """
    index = {}
    if not os.path.isdir(folder):
        return index
    for fname in os.listdir(folder):
        if not fname.endswith(suffix):
            continue
        stem = fname[: -len(suffix)]
        m = re.search(r"_([A-Za-z]+\d{4}_event)$", stem)
        if not m:
            continue
        label = m.group(1)
        hotspot = stem[: m.start()]
        key = (hotspot, label)
        index.setdefault(key, []).append(os.path.join(folder, fname))
    return index


def delete_files(paths, dry_run):
    for path in paths:
        if dry_run:
            print(f"    [DRY-RUN] Would delete: {path}")
        else:
            try:
                os.remove(path)
                print(f"    🗑️  Deleted: {path}")
            except Exception as e:
                print(f"    ❌ Failed to delete {path}: {e}")


def main():
    args = parse_args()
    output_dir  = args.output_dir
    dry_run     = args.dry_run

    truecolor_dir = os.path.join(output_dir, "truecolor")
    raw_dir       = os.path.join(output_dir, "raw")
    nbr_dir       = os.path.join(output_dir, "nbr")
    ndvi_dir      = os.path.join(output_dir, "ndvi")

    print("=" * 70)
    print("  SYNC OUTPUTS — remove rejected pairs from raw/nbr/ndvi")
    print(f"  Output dir : {output_dir}")
    print(f"  Dry run    : {dry_run}")
    print("=" * 70)

    # 1. Get passed keys from truecolor
    print("\n📂 Scanning truecolor (passed pairs)...")
    passed = get_passed_keys(truecolor_dir)
    print(f"\n  → {len(passed)} passed pair(s) found in truecolor\n")

    # 2. Build indexes for raw npy, nbr, ndvi
    npy_index  = build_npy_index(raw_dir)
    nbr_index  = build_png_index(nbr_dir,  "_burn.png")
    ndvi_index = build_png_index(ndvi_dir, "_clearance.png")

    all_keys = set(npy_index) | set(nbr_index) | set(ndvi_index)
    rejected_keys = all_keys - passed

    print(f"  Total unique pairs in raw/nbr/ndvi : {len(all_keys)}")
    print(f"  Passed (kept)                      : {len(passed)}")
    print(f"  Rejected (to delete)               : {len(rejected_keys)}")

    if not rejected_keys:
        print("\n✅ Nothing to delete — all pairs are in sync.")
        return

    # 3. Delete rejected files
    print("\n🗑️  Deleting rejected pairs...\n")
    total_deleted = 0

    for key in sorted(rejected_keys):
        hotspot, label = key
        print(f"  ❌ Rejected: {hotspot}  [{label}]")

        targets = (
            npy_index.get(key, []) +
            nbr_index.get(key, []) +
            ndvi_index.get(key, [])
        )
        delete_files(targets, dry_run)
        total_deleted += len(targets)

        # Remove raw hotspot subfolder if now empty
        hotspot_folder = os.path.join(raw_dir, hotspot)
        if os.path.isdir(hotspot_folder) and not os.listdir(hotspot_folder):
            if dry_run:
                print(f"    [DRY-RUN] Would remove empty folder: {hotspot_folder}")
            else:
                try:
                    os.rmdir(hotspot_folder)
                    print(f"    📁 Removed empty folder: {hotspot_folder}")
                except Exception as e:
                    print(f"    ⚠️  Could not remove folder {hotspot_folder}: {e}")

    print(f"\n{'='*70}")
    action = "Would delete" if dry_run else "Deleted"
    print(f"  {action} {total_deleted} file(s) across {len(rejected_keys)} rejected pair(s)")
    print(f"  Kept {len(passed)} clean pair(s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()