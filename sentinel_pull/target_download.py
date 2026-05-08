import os
import pandas as pd
from sentinel_pull.testsentinel import *   


ANALYSIS_CSV = r"D:\MCL\sentinel_outputs\analysis\priority_sorted.csv"

if not os.path.exists(ANALYSIS_CSV):
    print(f"❌ Error: Could not find {ANALYSIS_CSV}")
    exit()

df = pd.read_csv(ANALYSIS_CSV)


priority_order = {"VERY_HIGH": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
df["priority_score"] = df["priority"].map(priority_order).fillna(0)


top90 = df.nlargest(90, ["priority_score", "glad_high"])

print(f"Selected Top {len(top90)} pairs from CSV")
print("\nTop 10 from CSV:")
print(top90[["hotspot_period", "priority", "glad_high"]].head(10))

# ====================== FIXED MATCHING LOGIC ======================
RUN_HOTSPOTS = []
DATE_PAIRS_SELECTED = []
seen = set()

print("\n🔍 Matching CSV entries to internal HOTSPOTS and DATE_PAIRS...")

for _, row in top90.iterrows():
    full_hp = row["hotspot_period"]
    
   
    matched_dp = None
    for dp in DATE_PAIRS:
        
        if full_hp.endswith("_" + dp[4]):
            matched_dp = dp
            break
    
    if not matched_dp:
        continue

    
    name_to_match = full_hp[:-(len(matched_dp[4]) + 1)]

    
    for h in HOTSPOTS:
        if h[0] == name_to_match:
            key = (h[0], matched_dp[4])
            if key not in seen:
                seen.add(key)
                RUN_HOTSPOTS.append(h)
                DATE_PAIRS_SELECTED.append(matched_dp)
                print(f"   MATCHED: {h[0]} → {matched_dp[4]}")
            break

print(f"\n✅ Final unique pairs ready to download: **{len(RUN_HOTSPOTS)}**")

if __name__ == "__main__":
    if len(RUN_HOTSPOTS) == 0:
        print("\n❌ Cannot start download - no matches found between CSV and script constants.")
        print("Debug Info:")
        print(f"Sample Hotspot Names: {[h[0] for h in HOTSPOTS[:5]]}")
        print(f"Sample Periods: {[dp[4] for dp in DATE_PAIRS[:5]]}")
    else:
        print("=" * 90)
        print(f"🚀 STARTING DOWNLOAD — {len(RUN_HOTSPOTS)} PAIRS")
        print("=" * 90)
        
        results = []
        failed = 0

        for i, (hotspot, date_pair) in enumerate(zip(RUN_HOTSPOTS, DATE_PAIRS_SELECTED)):
            print(f"\n[{i+1:2d}/{len(RUN_HOTSPOTS)}] Processing: {hotspot[0]} for {date_pair[4]}")
            try:
                
                result = process_hotspot(hotspot, date_pair)
                if result:
                    results.append(result)
                else:
                    print("    ⚠️ Skipped (Return value was None)")
            except Exception as e:
                print(f"    ❌ Error processing {hotspot[0]}: {e}")
                failed += 1

        print("\n" + "="*90)
        print(f"✅ DOWNLOAD PROCESS COMPLETE")
        print(f"   Successfully processed : {len(results)} pairs")
        print(f"   Failed / Errors        : {failed}")
        print("="*90)