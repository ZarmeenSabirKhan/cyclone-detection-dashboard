import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

DATA_PATH = r"C:\Users\Tmart\Desktop\cy\future_5channels.npy"
TASMAX_NC = r"C:\Users\Tmart\Desktop\cy\extracted\tasmax_future.nc"

# Load data + original lat/lon (coarsened)
print("📂 Loading data...")
data = np.load(DATA_PATH)  # (365, 5, 138, 126)
print(f"   Shape: {data.shape}")

# Original lat/lon range thi -44.5 to -10 and 112 to 156.25
# Hum coarsen kiye, so new lat/lon array
H, W = data.shape[2], data.shape[3]
lats = np.linspace(-44.5, -10.0, H)  # H rows (south to north)
lons = np.linspace(112.0, 156.25, W)

# Tropical Australia region: lat -25 to -10, lon 115 to 155
tropical_lat_mask = (lats >= -25) & (lats <= -10)
tropical_lon_mask = (lons >= 115) & (lons <= 155)

print(f"\n🌏 Tropical region:")
print(f"   Latitude indices: {np.where(tropical_lat_mask)[0][0]} to {np.where(tropical_lat_mask)[0][-1]}")
print(f"   Longitude indices: {np.where(tropical_lon_mask)[0][0]} to {np.where(tropical_lon_mask)[0][-1]}")

# Crop data to tropical region
lat_idx = np.where(tropical_lat_mask)[0]
lon_idx = np.where(tropical_lon_mask)[0]
data_tropical = data[:, :, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
print(f"   Tropical data shape: {data_tropical.shape}")

# Cyclone season: Nov–April
# Days 0–119 (Jan–Apr) + Days 305–364 (Nov–Dec)
cyclone_season_days = list(range(0, 120)) + list(range(305, 365))
print(f"\n📅 Cyclone season: {len(cyclone_season_days)} days (Nov–April)")

# Compute "cyclone score" for each day in cyclone season
# Score = max_wind × max_pr (high wind + heavy rain = likely cyclone)
print("\n🔍 Searching for cyclone candidates...")
sfcwind_idx, pr_idx = 4, 1

cyclone_scores = []
for day in cyclone_season_days:
    max_wind = data_tropical[day, sfcwind_idx].max()
    max_pr = data_tropical[day, pr_idx].max()
    score = max_wind * max_pr  # Combined metric
    cyclone_scores.append((day, max_wind, max_pr, score))

# Sort by score
cyclone_scores.sort(key=lambda x: -x[3])

print("\n🌀 TOP 5 cyclone candidates (tropical region, cyclone season):")
print(f"{'Day':>5} {'Date':>12} {'Wind (m/s)':>12} {'Rain (mm)':>12} {'Score':>10}")
for day, wind, rain, score in cyclone_scores[:5]:
    # Convert day to approximate date
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m_idx in range(11, -1, -1):
        if day >= month_starts[m_idx]:
            month = months[m_idx]
            day_of_month = day - month_starts[m_idx] + 1
            break
    print(f"{day:>5} {f'{day_of_month} {month}':>12} {wind:>12.1f} {rain:>12.1f} {score:>10.0f}")

# Visualize the TOP cyclone candidate
top_day = cyclone_scores[0][0]
print(f"\n🎯 Visualizing Day {top_day} (top candidate)...")

variables = ['tasmax', 'pr', 'hurs', 'rsds', 'sfcWind']
units = ['°C', 'mm/day', '%', 'W/m²', 'm/s']
cmaps = ['hot', 'Blues', 'YlGnBu', 'plasma', 'viridis']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, (var, unit, cmap) in enumerate(zip(variables, units, cmaps)):
    im = axes[i].imshow(data_tropical[top_day, i], cmap=cmap, 
                        origin='lower', aspect='auto')
    axes[i].set_title(f"{var}\n({unit})", fontsize=11)
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, shrink=0.8)

plt.suptitle(f"Day {top_day} — TROPICAL region — Top cyclone candidate", 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\cyclone_candidate.png", 
            dpi=100, bbox_inches='tight')
plt.show()

# Save tropical data for next steps
np.save(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy", data_tropical)
print(f"\n💾 Saved tropical region data: future_tropical.npy")
print(f"   Shape: {data_tropical.shape}")