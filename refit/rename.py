import os
import shutil

raw_dir = r"C:\Users\User\MCL\sentinel_outputs\raw"
output_dir = r"C:\Users\User\MCL\sentinel_outputs\npy_flat"
os.makedirs(output_dir, exist_ok=True)

for hotspot in sorted(os.listdir(raw_dir)):
    hotspot_path = os.path.join(raw_dir, hotspot)
    if os.path.isdir(hotspot_path):
        for f in sorted(os.listdir(hotspot_path)):
            if f.endswith('.npy'):
                new_name = f"{hotspot}_{f}"
                src = os.path.join(hotspot_path, f)
                dst = os.path.join(output_dir, new_name)
                if not os.path.exists(dst):  # skip already copied
                    shutil.copy2(src, dst)
                    print(f"Copied: {new_name}")
                else:
                    print(f"Skipped (exists): {new_name}")

print(f"\nDone! Total: {len(os.listdir(output_dir))} files")