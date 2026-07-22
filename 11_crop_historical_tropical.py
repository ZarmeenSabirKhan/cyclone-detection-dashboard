"""
11_crop_historical_tropical.py
================================
Historical 5-channel data (30 years, 1985-2014) ko tropical Australia
region tak crop karta hai - exact same lat/lon logic jo 03_find_cyclones.py
mein future data ke liye use hui thi, taake dono datasets directly
compatible rahein (training mein combine karne ke liye).

Requirements: numpy
"""

import numpy as np
import os

DATA_PATH = r"C:\Users\Tmart\Desktop\cy\historical_5channels.npy"
DATES_PATH = r"C:\Users\Tmart\Desktop\cy\historical_dates.npy"
SAVE_PATH = r"C:\Users\Tmart\Desktop\cy\historical_tropical.npy"
SAVE_DATES_PATH = r"C:\Users\Tmart\Desktop\cy\historical_tropical_dates.npy"

print("📂 Loading historical data...")
data = np.load(DATA_PATH)  # (10957, 5, 138, 126)
dates = np.load(DATES_PATH, allow_pickle=True)
print(f"   Shape: {data.shape}")

# Same lat/lon range and coarsened grid as 03_find_cyclones.py
H, W = data.shape[2], data.shape[3]
lats = np.linspace(-44.5, -10.0, H)
lons = np.linspace(112.0, 156.25, W)

# Tropical Australia region: lat -25 to -10, lon 115 to 155 (same as future)
tropical_lat_mask = (lats >= -25) & (lats <= -10)
tropical_lon_mask = (lons >= 115) & (lons <= 155)

lat_idx = np.where(tropical_lat_mask)[0]
lon_idx = np.where(tropical_lon_mask)[0]

print(f"\n🌏 Tropical region:")
print(f"   Latitude indices: {lat_idx[0]} to {lat_idx[-1]}")
print(f"   Longitude indices: {lon_idx[0]} to {lon_idx[-1]}")

data_tropical = data[:, :, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
print(f"   Tropical data shape: {data_tropical.shape}")

# Sanity check: this should match future_tropical.npy's spatial dims exactly
print(f"\n   Expected to match future_tropical.npy spatial shape (should be same H,W as that file)")

np.save(SAVE_PATH, data_tropical)
np.save(SAVE_DATES_PATH, dates)

print(f"\n💾 Saved: {SAVE_PATH}")
print(f"   File size: {os.path.getsize(SAVE_PATH) / 1024**2:.1f} MB")
print(f"   Total historical tropical days: {data_tropical.shape[0]}")
