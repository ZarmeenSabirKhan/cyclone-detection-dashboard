"""
13_autoencoder_filtered.py
============================
12_autoencoder_expanded.py ka fix:
Problem tha ke historical 30 years mein real cyclones bhi the jo hum
"normal" treat kar rahe the — model ne broad normal distribution seekhi
aur future cyclones bhi normal lagney lagay (F1=0.0).

Fix: Historical data se bhi top 15% extreme days (high wind×rain score)
filter karo training se pehle — sirf genuinely calm days use karo.
Yeh wahi approach hai jo 04_generate_labels.py ne future data pe ki thi.

Test set WAHI rehta hai (future 73 days, seed=42) — fair comparison.
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
import gc

torch.manual_seed(42)
np.random.seed(42)

BASE = r"C:\Users\Tmart\Desktop\cy"

# ============================================================
# 1. Load future data — for test set (same seed=42 as always)
# ============================================================
print("📂 Loading future data...")
future_data = np.load(os.path.join(BASE, "future_tropical.npy"))
labels = np.load(os.path.join(BASE, "labels.npy"))
print(f"   Future data: {future_data.shape}, Cyclones: {labels.sum()}")

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    future_data, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train_f_normal = X_train_f[y_train_f == 0]
print(f"   Future normal training days: {len(X_train_f_normal)}")

# ============================================================
# 2. Load historical data + FILTER extreme days
# ============================================================
print("\n📂 Loading historical data...")
hist_data = np.load(os.path.join(BASE, "historical_tropical.npy"))
print(f"   Historical data: {hist_data.shape} ({hist_data.shape[0]} days)")

# Compute cyclone score for every historical day (same as 04_generate_labels.py)
print("\n🔍 Computing cyclone scores for historical data...")
pr_idx, wind_idx = 1, 4
hist_scores = np.zeros(hist_data.shape[0])

for day in range(hist_data.shape[0]):
    max_wind = hist_data[day, wind_idx].max()
    max_pr   = hist_data[day, pr_idx].max()
    hist_scores[day] = max_wind * max_pr

# Filter: keep only bottom 85% (exclude top 15% extreme days)
THRESHOLD_PERCENTILE = 85
threshold_score = np.percentile(hist_scores, THRESHOLD_PERCENTILE)
normal_mask = hist_scores < threshold_score

hist_normal = hist_data[normal_mask]
print(f"   Score threshold (85th percentile): {threshold_score:.0f}")
print(f"   Historical days kept (calm):    {hist_normal.shape[0]} ({hist_normal.shape[0]/hist_data.shape[0]*100:.1f}%)")
print(f"   Historical days removed (extreme): {(~normal_mask).sum()} ({(~normal_mask).sum()/hist_data.shape[0]*100:.1f}%)")

del hist_data
gc.collect()

# ============================================================
# 3. Combine: filtered historical + future normal days
# ============================================================
print("\n🔧 Combining filtered training pool...")
X_train_combined = np.concatenate(
    [hist_normal.astype(np.float32), X_train_f_normal.astype(np.float32)], axis=0
)
print(f"   Combined pool: {X_train_combined.shape[0]} days "
      f"({hist_normal.shape[0]} hist-normal + {len(X_train_f_normal)} future-normal)")

del hist_normal
gc.collect()

# ============================================================
# 4. Normalize per channel
# ============================================================
print("\n🔧 Normalizing...")
ch_min = X_train_combined.min(axis=(0, 2, 3), keepdims=True)
ch_max = X_train_combined.max(axis=(0, 2, 3), keepdims=True)

X_train_norm = (X_train_combined - ch_min) / (ch_max - ch_min + 1e-8)
X_test_norm  = np.clip(
    (X_test_f.astype(np.float32) - ch_min) / (ch_max - ch_min + 1e-8),
    0.0, 1.0
)
print(f"   Train range: {X_train_norm.min():.3f} to {X_train_norm.max():.3f}")
print(f"   Test range:  {X_test_norm.min():.3f} to {X_test_norm.max():.3f}")

del X_train_combined
gc.collect()

# ============================================================
# 5. Resize to 64x64
# ============================================================
print("\n📐 Resizing 60x113 → 64x64...")
X_train_t = F.interpolate(torch.from_numpy(X_train_norm).float(),
                          size=(64, 64), mode='bilinear', align_corners=False)
X_test_t  = F.interpolate(torch.from_numpy(X_test_norm).float(),
                          size=(64, 64), mode='bilinear', align_corners=False)
del X_train_norm, X_test_norm
gc.collect()

print(f"   Train: {X_train_t.shape}, Test: {X_test_t.shape}")
train_loader = DataLoader(TensorDataset(X_train_t), batch_size=32, shuffle=True)

# ============================================================
# 6. Autoencoder (identical architecture as 07/08/12)
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
            nn.ConvTranspose2d(16,  5, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = CycloneAutoencoder()
print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion  = nn.MSELoss()

# ============================================================
# 7. Train
# ============================================================
EPOCHS = 50
losses = []
print(f"\n🏋️ Training {EPOCHS} epochs on {X_train_t.shape[0]} filtered-normal days...")
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    t_ep = time.time()
    for (batch,) in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(train_loader)
    losses.append(avg)
    ep_t = time.time() - t_ep
    eta  = ep_t * (EPOCHS - epoch - 1)
    print(f"   Epoch {epoch+1:>3}/{EPOCHS}  loss={avg:.5f}  "
          f"time={ep_t:.0f}s  ETA={eta/60:.1f}min")

print(f"\n✅ Training done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# 8. Threshold from training errors
# ============================================================
model.eval()
train_errors = []
with torch.no_grad():
    for (batch,) in DataLoader(TensorDataset(X_train_t), batch_size=64):
        recon = model(batch)
        train_errors.append(((recon - batch)**2).mean(dim=[1,2,3]).numpy())
train_errors = np.concatenate(train_errors)

with torch.no_grad():
    recon_test = model(X_test_t)
    test_errors = ((recon_test - X_test_t)**2).mean(dim=[1,2,3]).numpy()

threshold = float(train_errors.mean() + 2 * train_errors.std())
print(f"\n🎯 Threshold: {threshold:.5f}")

y_pred = (test_errors > threshold).astype(int)

# ============================================================
# 9. Evaluate
# ============================================================
acc  = accuracy_score(y_test_f, y_pred)
prec = precision_score(y_test_f, y_pred, zero_division=0)
rec  = recall_score(y_test_f, y_pred, zero_division=0)
f1   = f1_score(y_test_f, y_pred, zero_division=0)
cm   = confusion_matrix(y_test_f, y_pred)

print("\n" + "=" * 60)
print("📊 FILTERED EXPANDED AUTOENCODER RESULTS")
print("=" * 60)
print(f"Training days: {X_train_t.shape[0]}  (filtered calm days only)")
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Pred Normal | Pred Cyclone")
print(f"  Actual Normal   {cm[0,0]:>11} | {cm[0,1]:>12}")
print(f"  Actual Cyclone  {cm[1,0]:>11} | {cm[1,1]:>12}")

# ============================================================
# 10. Full comparison table
# ============================================================
print("\n" + "=" * 75)
print("🏆 FINAL COMPARISON — ALL METHODS")
print("=" * 75)
print(f"{'Method':<30} {'Type':<15} {'Train days':<12} {'Acc':<10} {'F1':<8}")
print("-" * 75)
print(f"{'KNN (flatten)':<30} {'Supervised':<15} {'292':<12} {'98.6%':<10} {'0.952':<8}")
print(f"{'KNN + HOG':<30} {'Supervised':<15} {'292':<12} {'93.1%':<10} {'0.800':<8}")
print(f"{'Supervised CNN':<30} {'Supervised':<15} {'292':<12} {'98.6%':<10} {'0.957':<8}")

orig_path = os.path.join(BASE, "autoencoder_metrics.npy")
if os.path.exists(orig_path):
    orig = np.load(orig_path, allow_pickle=True).item()
    print(f"{'CNN AE (original, 232 days)':<30} {'Unsupervised':<15} {'232':<12} "
          f"{orig['accuracy']*100:.1f}%{'':<6} {orig['f1']:.3f}")

print(f"{'CNN AE (filtered+expanded)':<30} {'Unsupervised':<15} {X_train_t.shape[0]:<12} "
      f"{acc*100:.1f}%{'':<6} {f1:.3f}")
print("=" * 75)

# Save
torch.save(model.state_dict(), os.path.join(BASE, "autoencoder_model_v3_filtered.pt"))
np.savez(os.path.join(BASE, "autoencoder_artifacts_v3_filtered.npz"),
         threshold=threshold, ch_min=ch_min, ch_max=ch_max)
np.save(os.path.join(BASE, "autoencoder_v3_metrics.npy"),
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
         'n_training_days': X_train_t.shape[0]})
print(f"\n💾 Saved: autoencoder_model_v3_filtered.pt")

# ============================================================
# 11. Visualizations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(losses, color='steelblue', linewidth=2)
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].set_title(f'Training Loss — Filtered Expanded ({X_train_t.shape[0]} days)')
axes[0,0].grid(alpha=0.3)

colors = ['steelblue' if l==0 else 'red' for l in y_test_f]
axes[0,1].scatter(range(len(test_errors)), test_errors, c=colors, alpha=0.7)
axes[0,1].axhline(y=threshold, color='black', linestyle='--',
                  label=f'Threshold={threshold:.4f}')
axes[0,1].set_xlabel('Test sample index'); axes[0,1].set_ylabel('Reconstruction error')
axes[0,1].set_title('Reconstruction Error (red = actual cyclones)')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

im = axes[1,0].imshow(cm, cmap='Blues', aspect='auto')
axes[1,0].set_xticks([0,1]); axes[1,0].set_yticks([0,1])
axes[1,0].set_xticklabels(['Normal','Cyclone'])
axes[1,0].set_yticklabels(['Normal','Cyclone'])
axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Actual')
axes[1,0].set_title('Confusion Matrix — Filtered Expanded AE')
for i in range(2):
    for j in range(2):
        axes[1,0].text(j, i, str(cm[i,j]), ha='center', va='center',
                       fontsize=22, fontweight='bold',
                       color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1,0])

# Bar comparison
methods = ['KNN\n(flatten)', 'KNN\n+HOG', 'Supervised\nCNN',
           'AE\nOriginal', 'AE\nFiltered+Expanded']
accs_bar = [98.6, 93.1, 98.6,
            orig['accuracy']*100 if os.path.exists(orig_path) else 91.8,
            acc*100]
f1s_bar  = [95.2, 80.0, 95.7,
            orig['f1']*100 if os.path.exists(orig_path) else 75.0,
            f1*100]
x = np.arange(len(methods)); w = 0.35
axes[1,1].bar(x - w/2, accs_bar, w, label='Accuracy (%)', color='steelblue')
axes[1,1].bar(x + w/2, f1s_bar,  w, label='F1 × 100',    color='coral')
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(methods, fontsize=9)
axes[1,1].set_ylabel('Score'); axes[1,1].set_title('All Methods Comparison')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3, axis='y'); axes[1,1].set_ylim(0, 110)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "autoencoder_v3_filtered_results.png"),
            dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Done! Filtered expanded autoencoder complete.")
print("   Agar F1 original se better hai → success!")
print("   Agar phir bhi worse → threshold adjust karein ya report mein honest finding likhen.")
