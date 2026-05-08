import os
import json
import pandas as pd
from pathlib import Path

FIRMS_DIR = r"D:\MCL\sentinel_outputs\firms_false"
GLAD_DIR  = r"D:\MCL\sentinel_outputs\glad\alerts_raw_false"
OUTPUT_DIR = r"D:\MCL\sentinel_outputs\analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

glad_data = []
for file in os.listdir(GLAD_DIR):
    if file.endswith(".json"):
        path = os.path.join(GLAD_DIR, file)
        try:
            with open(path, encoding='utf-8') as f:
                alerts = json.load(f)
            if isinstance(alerts, list) and alerts:
                name_period = file.replace(".json", "")
                for a in alerts:
                    a_copy = a.copy()
                    a_copy["hotspot_period"] = name_period
                    a_copy["source"] = "GLAD"
                    glad_data.append(a_copy)
        except Exception as e:
            print(f"Error reading {file}: {e}")

glad_df = pd.DataFrame(glad_data)
print(f"Loaded {len(glad_df)} GLAD alerts from {len(os.listdir(GLAD_DIR))} files")

firms_list = []
for file in os.listdir(FIRMS_DIR):
    if file.endswith(".csv"):
        path = os.path.join(FIRMS_DIR, file)
        try:
            df = pd.read_csv(path)
            if not df.empty and len(df.columns) > 5:
                name_period = file.replace("firms_raw_", "").replace(".csv", "")
                df["hotspot_period"] = name_period
                df["source"] = "FIRMS"
                firms_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

firms_df = pd.concat(firms_list, ignore_index=True) if firms_list else pd.DataFrame()
print(f"Loaded {len(firms_df)} FIRMS records")

summary = []

all_hotspot_periods = sorted(set(glad_df["hotspot_period"]) | set(firms_df["hotspot_period"] if not firms_df.empty else []))

for hp in all_hotspot_periods:
    g = glad_df[glad_df["hotspot_period"] == hp] if not glad_df.empty else pd.DataFrame()
    f = firms_df[firms_df["hotspot_period"] == hp] if not firms_df.empty else pd.DataFrame()
    
    glad_high = len(g[g.get("gfw_integrated_alerts__confidence", pd.Series([])) == "high"])
    firms_high = len(f[f.get("confidence", pd.Series([])).astype(str).str.lower().str.contains("h", na=False)])
    
    total_glad = len(g)
    total_firms = len(f)
    
    if glad_high >= 5:
        priority = "VERY_HIGH"
    elif glad_high >= 2:
        priority = "HIGH"
    elif glad_high >= 1:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    summary.append({
        "hotspot_period": hp,
        "priority": priority,
        "glad_high": glad_high,
        "glad_total": total_glad,
        "firms_high": firms_high,
        "firms_total": total_firms,
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(["priority", "glad_high"], ascending=[False, False])

summary_df.to_csv(os.path.join(OUTPUT_DIR, "download_priority_false.csv"), index=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "priority_sorted_false.csv"), index=False)

print("\n" + "="*60)
print("DOWNLOAD PRIORITY SUMMARY")
print("="*60)
print(summary_df.groupby("priority").size())
print("\nTop 20 Highest Priority Pairs:")
print(summary_df.head(20)[["hotspot_period", "priority", "glad_high", "firms_high"]])

print(f"\n✅ Analysis complete! Files saved to:\n{OUTPUT_DIR}")