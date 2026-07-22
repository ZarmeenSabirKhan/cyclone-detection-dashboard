import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = r"C:\Users\Tmart\Desktop\cy\future_5channels.npy"
NORM_PATH = r"C:\Users\Tmart\Desktop\cy\future_normalized.npy"
STATS_PATH = r"C:\Users\Tmart\Desktop\cy\norm_stats.npz"

# Load
print("📂 Loading data...")
data = np.load(DATA_PATH)
print(f"   Shape: {data.shape}")

# Normalize each channel to [0, 1]
print("\n🔧 Normalizing each channel to [0, 1]...")
ch_min = data.min(axis=(0, 2, 3), keepdims=True)
ch_max = data.max(axis=(0, 2, 3), keepdims=True)
data_norm = (data - ch_min) / (ch_max - ch_min)

# Save normalized data + stats (stats baseline pe bhi use karenge)
np.save(NORM_PATH, data_norm)
np.savez(STATS_PATH, ch_min=ch_min.squeeze(), ch_max=ch_max.squeeze())
print(f"   Normalized range: {data_norm.min():.3f} to {data_norm.max():.3f}")
print(f"   💾 Saved normalized data: {NORM_PATH}")

# Find day with highest wind speed (likely cyclone candidate)
sfcwind_channel = 4  # index of sfcWind in variables list
max_wind_per_day = data[:, sfcwind_channel, :, :].max(axis=(1, 2))
peak_day = int(np.argmax(max_wind_per_day))
print(f"\n🌀 Strongest wind day: Day {peak_day} (max wind = {max_wind_per_day[peak_day]:.1f} m/s)")

# Visualize 5 channels for peak wind day (likely cyclone)
variables = ['tasmax', 'pr', 'hurs', 'rsds', 'sfcWind']
units = ['°C', 'mm/day', '%', 'W/m²', 'm/s']
cmaps = ['hot', 'Blues', 'YlGnBu', 'plasma', 'viridis']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, (var, unit, cmap) in enumerate(zip(variables, units, cmaps)):
    im = axes[i].imshow(data[peak_day, i], cmap=cmap, origin='lower', aspect='auto')
    axes[i].set_title(f"{var}\n({unit})", fontsize=11)
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, shrink=0.8)

plt.suptitle(f"Day {peak_day} — High wind day (probable cyclone) 🌀", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\peak_day_5channels.png", dpi=100, bbox_inches='tight')
plt.show()
print(f"   💾 Saved visualization: peak_day_5channels.png")