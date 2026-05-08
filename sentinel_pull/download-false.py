import os
import pandas as pd
from sentinel_pull.sentinel_false import *

ANALYSIS_CSV = r"D:\MCL\sentinel_outputs\analysis\priority_sorted.csv"


MODE = "NEGATIVE"

POSITIVE_COUNT = 55
NEGATIVE_COUNT = 70


if not os.path.exists(ANALYSIS_CSV):
    print(f"❌ Error: Could not find {ANALYSIS_CSV}")
    exit()

df = pd.read_csv(ANALYSIS_CSV)


if MODE == "POSITIVE":
    selected_df = df.nlargest(POSITIVE_COUNT, ["glad_high", "glad_total"])
    print(f"  POSITIVE mode — top {POSITIVE_COUNT} pairs")
    print(f"  glad_high range: {selected_df['glad_high'].min()} – {selected_df['glad_high'].max()}")

elif MODE == "NEGATIVE":
    # Exclude whatever is in the positive set
    top_keys = set(df.nlargest(POSITIVE_COUNT, ["glad_high", "glad_total"])["hotspot_period"])

    # Sort ascending by glad_high then glad_total — lowest signal first
    df_sorted = df.sort_values(["glad_high", "glad_total"], ascending=True)

    # Remove positives and take bottom N
    neg_pool = df_sorted[~df_sorted["hotspot_period"].isin(top_keys)]
    selected_df = neg_pool.head(NEGATIVE_COUNT)

    print(f"  NEGATIVE mode — bottom {NEGATIVE_COUNT} pairs (excl. top {POSITIVE_COUNT} positives)")
    print(f"  glad_high range: {selected_df['glad_high'].min()} – {selected_df['glad_high'].max()}")
    print(f"  firms_high > 0: {(selected_df['firms_high'] > 0).sum()} rows")
    print(f"  Priority breakdown: {dict(selected_df['priority'].value_counts())}")

else:
    raise ValueError("MODE must be 'POSITIVE' or 'NEGATIVE'")

print(f"\n  Selected {len(selected_df)} pairs")
print(f"\n  Sample (first 10):")
print(selected_df[["hotspot_period", "priority", "glad_high", "glad_total", "firms_high"]].head(10).to_string(index=False))


RUN_HOTSPOTS = []
DATE_PAIRS_SELECTED = []
seen = set()
unmatched = []

print(f"\n🔍 Matching to HOTSPOTS and DATE_PAIRS...")

for _, row in selected_df.iterrows():
    full_hp = row["hotspot_period"]

    matched_dp = None
    for dp in DATE_PAIRS:
        if full_hp.endswith("_" + dp[4]):
            matched_dp = dp
            break

    if not matched_dp:
        unmatched.append(full_hp)
        continue

    hotspot_name = full_hp[:-(len(matched_dp[4]) + 1)]

    for h in HOTSPOTS:
        if h[0] == hotspot_name:
            key = (h[0], matched_dp[4])
            if key not in seen:
                seen.add(key)
                RUN_HOTSPOTS.append(h)
                DATE_PAIRS_SELECTED.append(matched_dp)
                print(f"   ✅ {h[0]} → {matched_dp[4]}")
            break
    else:
        unmatched.append(full_hp)

if unmatched:
    print(f"\n  ⚠️  {len(unmatched)} unmatched:")
    for u in unmatched:
        print(f"     - {u}")

print(f"\n  → {len(RUN_HOTSPOTS)} pairs ready to download")

if __name__ == "__main__":

    if len(RUN_HOTSPOTS) == 0:
        print("\n❌ Cannot start — no matched pairs found.")
        exit()

    print("\n" + "=" * 80)
    print(f"  🚀 STARTING DOWNLOAD — {len(RUN_HOTSPOTS)} PAIRS — MODE={MODE}")
    print("=" * 80)

    results = []
    failed  = 0

    for i, (hotspot, date_pair) in enumerate(zip(RUN_HOTSPOTS, DATE_PAIRS_SELECTED)):
        print(f"\n[{i+1:2d}/{len(RUN_HOTSPOTS)}] {hotspot[0]} → {date_pair[4]}")
        try:
            result = process_hotspot(hotspot, date_pair)
            if result:
                results.append(result)
            else:
                print("    ⚠️  Skipped (quality gate or missing scenes)")
                failed += 1
        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"  ✅ DONE | MODE={MODE}")
    print(f"     Successfully processed : {len(results)}")
    print(f"     Failed / skipped       : {failed}")
    print("=" * 80)