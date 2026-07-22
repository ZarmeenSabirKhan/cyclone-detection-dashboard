import numpy as np
import xarray as xr
import os
import gc

EXTRACTED_DIR = r"C:\Users\Tmart\Desktop\cy\extracted"
SAVE_PATH = r"C:\Users\Tmart\Desktop\cy\future_5channels.npy"

variables = ['tasmax', 'pr', 'hurs', 'rsds', 'sfcWind']

# Downsample factors: 691/5 ≈ 138, 886/7 ≈ 126
LAT_FACTOR = 5
LON_FACTOR = 7

print("📥 Processing 5 variables (this takes 3-5 min)...\n")
arrays = []
for var in variables:
    file_path = os.path.join(EXTRACTED_DIR, f"{var}_future.nc")
    if not os.path.exists(file_path):
        print(f"  ❌ {var}: file not found")
        continue
    
    print(f"  📦 {var}: opening...", end=" ", flush=True)
    ds = xr.open_dataset(file_path)
    actual_var = list(ds.data_vars)[0]
    
    print(f"coarsening...", end=" ", flush=True)
    coarsened = ds[actual_var].coarsen(
        lat=LAT_FACTOR, lon=LON_FACTOR, boundary='trim'
    ).mean()
    arr = coarsened.values.astype(np.float32)
    print(f"shape={arr.shape}")
    
    arrays.append(arr)
    ds.close()
    gc.collect()

# Stack into 5-channel array: (days, 5, H, W)
data_future = np.stack(arrays, axis=1)
print(f"\n✅ Final shape: {data_future.shape}")
print(f"   Memory: {data_future.nbytes / 1024**2:.1f} MB")

# Per-channel value ranges (sanity check)
print("\n📊 Channel value ranges:")
for i, var in enumerate(variables):
    ch = data_future[:, i, :, :]
    print(f"   {var}: {ch.min():.2f} to {ch.max():.2f}")

np.save(SAVE_PATH, data_future)
print(f"\n💾 Saved: {SAVE_PATH}")
print(f"   File size: {os.path.getsize(SAVE_PATH) / 1024**2:.1f} MB")