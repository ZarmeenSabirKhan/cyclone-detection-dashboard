"""
12_autoencoder_expanded.py
============================
Autoencoder ko EXPANDED training pool pe train karta hai:
  - Sab 10,957 historical days (1985-2014, "normal" baseline weather pool)
  - + future 2064 ke 232 normal-labeled training days (jo pehle bhi use hue the)
  = ~11,189 training days total (pehle sirf 232 the)

Test set WAHI rehta hai jo 05/06/07/07b mein tha (future ke 73 held-out din,
seed=42 stratified split) - taake purane methods (KNN, HOG+KNN, CNN,
original Autoencoder) ke saath FAIR comparison ho sake.

NOTE: Historical days ko "normal" treat kiya ja raha hai kyunki humein
unke labels nahi hain (unsupervised setting). Real historical cyclones
bhi is pool mein maujood hain lekin woh rare hain (~1-2% of days), isliye
autoencoder anomaly-detection assumption (majority = normal pattern)
still valid rehta hai.

Requirements: numpy, torch, sklearn, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import time
import os

torch.manual_seed(42)
np.random.seed(42)

BASE = r"C:\Users\Tmart\Desktop\cy"

# ============================================================
# 1. Load future data (for the labeled train/test split, same as before)
# ============================================================
print("📂 Loading future data (for test set + labels)...")
future_data = np.load(os.path.join(BASE, "future_tropical.npy"))  # (365, 5, 60, 113)
labels = np.load(os.path.join(BASE, "labels.npy"))
print(f"   Future data: {future_data.shape}, Cyclones: {labels.sum()}")

# SAME split as 05/06/07/07b (seed=42, stratify) -> test set is identical
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    future_data, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"   Future train: {len(X_train_f)} days, Future test (held-out): {len(X_test_f)} days")

# Only the NORMAL days from future's training split (same logic as 07_autoencoder.py)
X_train_f_normal = X_train_f[y_train_f == 0]
print(f"   Future normal training days: {len(X_train_f_normal)}")

# ============================================================
# 2. Load historical data (treated entirely as "normal" pool)
# ============================================================
print("\n📂 Loading historical data (expanded training pool)...")
historical_data = np.load(os.path.join(BASE, "historical_tropical.npy"))  # (10957, 5, 60, 113)
print(f"   Historical data: {historical_data.shape}")

# ============================================================
# 3. Combine training pool: historical (all) + future normal train days
# ============================================================
print("\n🔧 Combining training pool...")
X_train_combined = np.concatenate([historical_data, X_train_f_normal], axis=0).astype(np.float32)
print(f"   Combined training pool: {X_train_combined.shape[0]} days "
      f"({historical_data.shape[0]} historical + {len(X_train_f_normal)} future-normal)")

del historical_data  # free memory, no longer needed as separate array
import gc
gc.collect()

# ============================================================
# 4. Normalize per channel to [0,1] using stats from the COMBINED training pool
#    (must reuse the SAME stats on the test set, for consistency)
# ============================================================
print("\n🔧 Computing normalization stats from combined training pool...")
ch_min = X_train_combined.min(axis=(0, 2, 3), keepdims=True)
ch_max = X_train_combined.max(axis=(0, 2, 3), keepdims=True)

X_train_norm = (X_train_combined - ch_min) / (ch_max - ch_min + 1e-8)
X_test_norm = (X_test_f.astype(np.float32) - ch_min) / (ch_max - ch_min + 1e-8)
# clip test set in case future extremes exceed the historical+future-train range
X_test_norm = np.clip(X_test_norm, 0.0, 1.0)

print(f"   Train normalized range: {X_train_norm.min():.3f} to {X_train_norm.max():.3f}")
print(f"   Test normalized range:  {X_test_norm.min():.3f} to {X_test_norm.max():.3f}")

del X_train_combined
gc.collect()

# ============================================================
# 5. Resize to 64x64 (same as original autoencoder pipeline)
# ============================================================
print("\n📐 Resizing 60x113 -> 64x64...")
X_train_t = torch.from_numpy(X_train_norm).float()
X_train_resized = F.interpolate(X_train_t, size=(64, 64), mode='bilinear', align_corners=False)
del X_train_t, X_train_norm
gc.collect()

X_test_t_raw = torch.from_numpy(X_test_norm).float()
X_test_resized = F.interpolate(X_test_t_raw, size=(64, 64), mode='bilinear', align_corners=False)
print(f"   Train tensor: {X_train_resized.shape}")
print(f"   Test tensor:  {X_test_resized.shape}")

train_loader = DataLoader(TensorDataset(X_train_resized), batch_size=16, shuffle=True)

# ============================================================
# 6. Autoencoder architecture - IDENTICAL to 07_autoencoder.py / 08_train_save.py
# ============================================================
class CycloneAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 5, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = CycloneAutoencoder()
print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 50
losses = []
batches_per_epoch = len(train_loader)
print(f"\n🏋️ Training for {EPOCHS} epochs, {batches_per_epoch} batches/epoch "
      f"({batches_per_epoch * EPOCHS:,} total batches)...")
print("   (This will take a while on CPU with the full dataset - progress prints every epoch)")

t_start = time.time()
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    t_epoch = time.time()
    for (batch,) in train_loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / batches_per_epoch
    losses.append(avg_loss)
    epoch_time = time.time() - t_epoch
    elapsed_total = time.time() - t_start
    eta = epoch_time * (EPOCHS - epoch - 1)

    print(f"   Epoch {epoch+1:>3}/{EPOCHS}  loss={avg_loss:.5f}  "
          f"epoch_time={epoch_time:.0f}s  elapsed={elapsed_total/60:.1f}min  ETA={eta/60:.1f}min")

total_time = time.time() - t_start
print(f"\n✅ Training done in {total_time/60:.1f} min")

# ============================================================
# 7. Compute reconstruction errors and threshold
# ============================================================
model.eval()
with torch.no_grad():
    # Compute training errors in batches to avoid memory spike
    train_errors_list = []
    eval_loader = DataLoader(TensorDataset(X_train_resized), batch_size=64, shuffle=False)
    for (batch,) in eval_loader:
        recon = model(batch)
        err = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
        train_errors_list.append(err.numpy())
    train_errors = np.concatenate(train_errors_list)

    recon_test = model(X_test_resized)
    test_errors = ((recon_test - X_test_resized) ** 2).mean(dim=[1, 2, 3]).numpy()

threshold = float(train_errors.mean() + 2 * train_errors.std())
print(f"\n🎯 Threshold (train mean + 2*std): {threshold:.5f}")

y_pred = (test_errors > threshold).astype(int)

# ============================================================
# 8. Evaluate
# ============================================================
acc = accuracy_score(y_test_f, y_pred)
prec = precision_score(y_test_f, y_pred, zero_division=0)
rec = recall_score(y_test_f, y_pred, zero_division=0)
f1 = f1_score(y_test_f, y_pred, zero_division=0)
cm = confusion_matrix(y_test_f, y_pred)

print("\n" + "=" * 60)
print("📊 EXPANDED AUTOENCODER RESULTS (trained on historical+future)")
print("=" * 60)
print(f"Training days used: {len(train_errors)} (vs 232 in original)")
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Pred Normal | Pred Cyclone")
print(f"  Actual Normal   {cm[0,0]:>11} | {cm[0,1]:>12}")
print(f"  Actual Cyclone  {cm[1,0]:>11} | {cm[1,1]:>12}")

# ============================================================
# 9. Save model, artifacts, metrics (as v2, doesn't overwrite original)
# ============================================================
torch.save(model.state_dict(), os.path.join(BASE, "autoencoder_model_v2_expanded.pt"))
np.savez(os.path.join(BASE, "autoencoder_artifacts_v2_expanded.npz"),
         threshold=threshold, ch_min=ch_min, ch_max=ch_max)
np.save(os.path.join(BASE, "autoencoder_v2_metrics.npy"),
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
         'n_training_days': len(train_errors)})
np.save(os.path.join(BASE, "autoencoder_v2_errors.npy"), test_errors)
print(f"\n💾 Saved: autoencoder_model_v2_expanded.pt, artifacts, metrics")

# ============================================================
# 10. Comparison with ORIGINAL autoencoder (232 days), if available
# ============================================================
orig_metrics_path = os.path.join(BASE, "autoencoder_metrics.npy")
print("\n" + "=" * 70)
print("🏆 COMPARISON: Original (232 days) vs Expanded (historical+future)")
print("=" * 70)
if os.path.exists(orig_metrics_path):
    orig = np.load(orig_metrics_path, allow_pickle=True).item()
    print(f"{'Version':<25} {'Train days':<12} {'Accuracy':<12} {'F1':<10}")
    print("-" * 70)
    print(f"{'Original Autoencoder':<25} {'232':<12} {orig['accuracy']*100:.1f}%{'':<7} {orig['f1']:.3f}")
    print(f"{'Expanded Autoencoder':<25} {len(train_errors):<12} {acc*100:.1f}%{'':<7} {f1:.3f}")
else:
    print("(original autoencoder_metrics.npy not found - skipping side-by-side)")
    print(f"{'Expanded Autoencoder':<25} {len(train_errors):<12} {acc*100:.1f}%{'':<7} {f1:.3f}")
print("=" * 70)

# ============================================================
# 11. Visualizations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(losses, color='steelblue', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Reconstruction Loss')
axes[0, 0].set_title(f'Training Loss (Expanded, {len(train_errors)} days)')
axes[0, 0].grid(alpha=0.3)

colors = ['steelblue' if l == 0 else 'red' for l in y_test_f]
axes[0, 1].scatter(range(len(test_errors)), test_errors, c=colors, alpha=0.7)
axes[0, 1].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
axes[0, 1].set_xlabel('Test sample index')
axes[0, 1].set_ylabel('Reconstruction error')
axes[0, 1].set_title('Reconstruction Error (red = actual cyclones)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Normal', 'Cyclone'])
axes[1, 0].set_yticklabels(['Normal', 'Cyclone'])
axes[1, 0].set_xlabel('Predicted'); axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix - Expanded Autoencoder')
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=22, fontweight='bold',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im, ax=axes[1, 0])

if os.path.exists(orig_metrics_path):
    orig = np.load(orig_metrics_path, allow_pickle=True).item()
    methods = ['Original\n(232 days)', 'Expanded\n(historical+future)']
    accs = [orig['accuracy'] * 100, acc * 100]
    f1s = [orig['f1'] * 100, f1 * 100]
    x = np.arange(len(methods))
    width = 0.35
    axes[1, 1].bar(x - width/2, accs, width, label='Accuracy (%)', color='steelblue')
    axes[1, 1].bar(x + width/2, f1s, width, label='F1 x 100', color='coral')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Original vs Expanded Autoencoder')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 110)
else:
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Original metrics not found\n(run 07_autoencoder.py first for comparison)',
                     ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(BASE, "autoencoder_v2_expanded_results.png"), dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Expanded autoencoder training complete!")
