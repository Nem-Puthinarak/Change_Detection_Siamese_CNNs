import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

FIRMS_MAP_KEY = "fab785ea5e883e3a5dd71164f00998c0"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


HOTSPOTS = [
    ("hotspot_01_osmach_west",         103.630, 14.410, 103.680, 14.450),
    ("hotspot_02_o3chean_village",     102.720, 13.765, 102.765, 13.810),
    ("hotspot_03_phnom_proek",         102.349, 13.310, 102.363, 13.325),
    ("hotspot_04_poipet_thai",         102.530, 13.653, 102.575, 13.688),
    ("hotspot_05_poipet_road",         102.690, 13.645, 102.735, 13.680),
    ("hotspot_06_osmach_east",         103.685, 14.408, 103.730, 14.445),
    ("hotspot_07_choam_chom_sngan",    104.040, 14.330, 104.085, 14.360),
    ("hotspot_08_bos_sbov",            104.730, 14.412, 104.775, 14.445),  # shifted west to avoid tile seam
    ("hotspot_09_chok_chey",           102.665, 13.745, 102.720, 13.780),
    ("hotspot_10_osmach_jan2026",      103.645, 14.400, 103.695, 14.445),
    ("hotspot_11_an_ses",              104.942, 14.371, 104.987, 14.406),
    ("hotspot_12_ta_moan_thom",        103.243, 14.330, 103.288, 14.365),
    ("hotspot_13_ta_krabei",           103.351, 14.335, 103.396, 14.380),  # extended north to cover Thai side
    ("hotspot_14_preahvihear_thai",    104.657, 14.381, 104.702, 14.416),
    ("hotspot_15_preahvihear_temple",  104.657, 14.374, 104.702, 14.409),
    ("hotspot_16_thmor_da",            102.712, 12.114, 102.757, 12.150),
    ("hotspot_17_prei_chan",           102.714, 13.800, 102.759, 13.835),
    ("hotspot_18_beoung_trakoun", 102.845, 13.980, 102.862, 13.992),
    ("hotspot_19_thmor_doun", 103.309, 14.343, 103.333, 14.357),
    ("hotspot_20_emerald_triangle", 105.160, 14.299, 105.205, 14.350),
]

results = []

for name, lon_min, lat_min, lon_max, lat_max in HOTSPOTS:
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    start = datetime(2026, 2, 1)
    end = datetime(2026, 4, 25)
    cur = start
    print(f"\nChecking {name}...")
    while cur <= end:
        chunk_end = min(cur + timedelta(days=4), end)
        n_days = (chunk_end - cur).days + 1
        date_str = cur.strftime("%Y-%m-%d")
        url = f"{FIRMS_URL}/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{bbox}/{n_days}/{date_str}"
        try:
            resp = requests.get(url, timeout=30)
            print(f"  {date_str} status={resp.status_code} len={len(resp.text)}")
            if resp.status_code == 200 and len(resp.text) > 60:
                df = pd.read_csv(StringIO(resp.text))
                if len(df) > 0:
                    print(f"  FIRE FOUND: {len(df)} alerts!")
                    for _, row in df.iterrows():
                        results.append({
                            "hotspot": name,
                            "date": row.get("acq_date", ""),
                            "lat": row.get("latitude", ""),
                            "lon": row.get("longitude", ""),
                            "confidence": row.get("confidence", "")
                        })
        except Exception as e:
            print(f"  ERROR: {e}")
        cur = chunk_end + timedelta(days=1)

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    print("\n=== FIRES FOUND ===")
    print(df_results.groupby(["hotspot", "date"]).size().reset_index(name="alerts"))
else:
    print("\nNo fires detected Feb-April 2026")