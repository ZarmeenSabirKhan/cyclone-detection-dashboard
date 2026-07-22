import numpy as np
import matplotlib.pyplot as plt

# Use the main 5-channel file (which definitely exists)
DATA_PATH = r"C:\Users\Tmart\Desktop\cy\future_5channels.npy"
LABELS_PATH = r"C:\Users\Tmart\Desktop\cy\labels.npy"
SCORES_PATH = r"C:\Users\Tmart\Desktop\cy\cyclone_scores.npy"
TROPICAL_PATH = r"C:\Users\Tmart\Desktop\cy\future_tropical.npy"

# Load full data
print("📂 Loading full data...")
data = np.load(DATA_PATH)  # (365, 5, 138, 126)
print(f"   Full shape: {data.shape}")

# Crop to tropical region (same as 03 step)
H, W = data.shape[2], data.shape[3]
lats = np.linspace(-44.5, -10.0, H)
lons = np.linspace(112.0, 156.25, W)

lat_idx = np.where((lats >= -25) & (lats <= -10))[0]
lon_idx = np.where((lons >= 115) & (lons <= 155))[0]

data_tropical = data[:, :, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
print(f"   Tropical shape: {data_tropical.shape}")

# Save tropical data (so next time it's available)
np.save(TROPICAL_PATH, data_tropical)
print(f"   💾 Saved: future_tropical.npy")

# Compute cyclone scores for all 365 days
pr_idx, sfcwind_idx = 1, 4

print("\n🔍 Computing cyclone score for all 365 days...")
scores = np.zeros(365)
for day in range(365):
    max_wind = data_tropical[day, sfcwind_idx].max()
    max_pr = data_tropical[day, pr_idx].max()
    scores[day] = max_wind * max_pr

np.save(SCORES_PATH, scores)

# Statistics
print(f"\n📊 Score statistics:")
print(f"   Min:    {scores.min():.0f}")
print(f"   Max:    {scores.max():.0f}")
print(f"   Mean:   {scores.mean():.0f}")
print(f"   Median: {np.median(scores):.0f}")

# Generate labels — top 15% as cyclones
THRESHOLD_PERCENTILE = 85
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
labels = (scores >= threshold).astype(np.int32)

print(f"\n🌀 Threshold (85th percentile): {threshold:.0f}")
print(f"   Cyclone days:  {labels.sum()} ({labels.sum()/365*100:.1f}%)")
print(f"   Normal days:   {(1-labels).sum()} ({(1-labels).sum()/365*100:.1f}%)")

np.save(LABELS_PATH, labels)
print(f"\n💾 Saved labels: {LABELS_PATH}")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
day_axis = np.arange(365)

# Time series
axes[0].plot(day_axis, scores, color='steelblue', alpha=0.7, linewidth=1)
axes[0].axhline(y=threshold, color='red', linestyle='--',
                label=f'Threshold ({threshold:.0f})')
axes[0].scatter(day_axis[labels == 1], scores[labels == 1],
                color='red', s=30, label=f'Cyclone days ({labels.sum()})')
axes[0].axvspan(0, 120, alpha=0.1, color='orange', label='Cyclone season')
axes[0].axvspan(305, 365, alpha=0.1, color='orange')
axes[0].set_xlabel('Day of Year 2064')
axes[0].set_ylabel('Cyclone Score (wind x rain)')
axes[0].set_title('Cyclone Score over the Year')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Histogram
axes[1].hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=threshold, color='red', linestyle='--',
                label=f'Threshold ({threshold:.0f})')
axes[1].set_xlabel('Cyclone Score')
axes[1].set_ylabel('Number of Days')
axes[1].set_title('Distribution of Cyclone Scores')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\labels_visualization.png",
            dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Labels generated and visualized!")