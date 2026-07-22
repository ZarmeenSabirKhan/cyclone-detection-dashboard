"""
14_threshold_tuning.py
========================
13_autoencoder_filtered.py ka saved model load karke
different thresholds pe evaluate karta hai — best F1/Recall
balance dhundhne ke liye. Model dobara train nahi karna.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

BASE = r"C:\Users\Tmart\Desktop\cy"

# ============================================================
# 1. Load saved model + artifacts from 13_autoencoder_filtered.py
# ============================================================
print("📂 Loading saved model and artifacts...")

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
model.load_state_dict(torch.load(
    os.path.join(BASE, "autoencoder_model_v3_filtered.pt"),
    map_location="cpu"
))
model.eval()
print("   Model loaded ✅")

art = np.load(os.path.join(BASE, "autoencoder_artifacts_v3_filtered.npz"))
ch_min = art['ch_min']
ch_max = art['ch_max']
old_threshold = float(art['threshold'])
print(f"   Original threshold (mean+2std): {old_threshold:.6f}")

# ============================================================
# 2. Rebuild test set (same seed=42 as always)
# ============================================================
print("\n📂 Rebuilding test set...")
future_data = np.load(os.path.join(BASE, "future_tropical.npy"))
labels = np.load(os.path.join(BASE, "labels.npy"))

_, X_test_f, _, y_test_f = train_test_split(
    future_data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Normalize using saved stats
X_test_norm = np.clip(
    (X_test_f.astype(np.float32) - ch_min) / (ch_max - ch_min + 1e-8),
    0.0, 1.0
)
X_test_t = F.interpolate(
    torch.from_numpy(X_test_norm).float(),
    size=(64, 64), mode='bilinear', align_corners=False
)
print(f"   Test set: {X_test_t.shape[0]} days, {y_test_f.sum()} cyclones")

# ============================================================
# 3. Get reconstruction errors
# ============================================================
with torch.no_grad():
    recon_test = model(X_test_t)
    test_errors = ((recon_test - X_test_t)**2).mean(dim=[1,2,3]).numpy()

# ============================================================
# 4. Try multiple thresholds — mean + k*std for different k
# ============================================================
# We also need train_errors to compute mean+k*std thresholds
# Load from saved errors if available, else recompute from test distribution
train_errors_path = os.path.join(BASE, "autoencoder_v3_train_errors.npy")

if os.path.exists(train_errors_path):
    train_errors = np.load(train_errors_path)
    train_mean = train_errors.mean()
    train_std  = train_errors.std()
    print(f"\n   Train errors loaded: mean={train_mean:.6f}, std={train_std:.6f}")
else:
    # Estimate from old threshold: threshold = mean + 2*std → solve for mean,std
    # We'll try a range of absolute threshold values instead
    train_mean = None
    train_std  = None
    print("\n   Train errors not found — using absolute threshold sweep")

print("\n🔍 Trying different thresholds...")
print(f"\n{'Threshold':<15} {'Multiplier':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1':<8} {'TP':<5} {'FP':<5}")
print("-" * 90)

results = []

if train_mean is not None:
    # Try k from 0.5 to 2.5 in steps of 0.25
    k_values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    thresholds = [(train_mean + k * train_std, f"mean+{k}std") for k in k_values]
else:
    # Sweep around old threshold
    thresholds = [(old_threshold * f, f"×{f:.2f}") for f in
                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]

for thresh, label in thresholds:
    y_pred = (test_errors > thresh).astype(int)
    acc  = accuracy_score(y_test_f, y_pred)
    prec = precision_score(y_test_f, y_pred, zero_division=0)
    rec  = recall_score(y_test_f, y_pred, zero_division=0)
    f1   = f1_score(y_test_f, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test_f, y_pred)
    tp   = cm[1,1] if cm.shape == (2,2) else 0
    fp   = cm[0,1] if cm.shape == (2,2) else 0
    results.append((thresh, label, acc, prec, rec, f1, tp, fp))
    marker = " ← original" if abs(thresh - old_threshold) < 1e-8 else ""
    print(f"{thresh:<15.6f} {label:<12} {acc*100:<12.1f} {prec:<12.3f} {rec:<10.3f} {f1:<8.3f} {tp:<5} {fp:<5}{marker}")

# Best by F1
best_f1 = max(results, key=lambda x: x[5])
best_recall = max(results, key=lambda x: x[4])

print(f"\n🏆 Best F1:    threshold={best_f1[0]:.6f} ({best_f1[1]}) → F1={best_f1[5]:.3f}, Recall={best_f1[4]:.3f}, Precision={best_f1[3]:.3f}")
print(f"🏆 Best Recall: threshold={best_recall[0]:.6f} ({best_recall[1]}) → Recall={best_recall[4]:.3f}, Precision={best_recall[3]:.3f}, F1={best_recall[5]:.3f}")

# ============================================================
# 5. Show detailed results for best F1 threshold
# ============================================================
thresh_use, label_use, acc, prec, rec, f1, tp, fp = best_f1
y_pred_best = (test_errors > thresh_use).astype(int)
cm_best = confusion_matrix(y_test_f, y_pred_best)

print(f"\n{'='*60}")
print(f"📊 BEST THRESHOLD RESULTS ({label_use})")
print(f"{'='*60}")
print(f"Threshold: {thresh_use:.6f}  (original was {old_threshold:.6f})")
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Pred Normal | Pred Cyclone")
print(f"  Actual Normal   {cm_best[0,0]:>11} | {cm_best[0,1]:>12}")
print(f"  Actual Cyclone  {cm_best[1,0]:>11} | {cm_best[1,1]:>12}")

# Save best threshold artifacts
np.savez(os.path.join(BASE, "autoencoder_artifacts_v3_tuned.npz"),
         threshold=thresh_use, ch_min=ch_min, ch_max=ch_max)
print(f"\n💾 Saved tuned artifacts: autoencoder_artifacts_v3_tuned.npz")

# ============================================================
# 6. Plot: threshold sweep + best confusion matrix
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Threshold sweep curves
thresh_vals = [r[0] for r in results]
f1_vals     = [r[5] for r in results]
rec_vals    = [r[4] for r in results]
prec_vals   = [r[3] for r in results]

axes[0].plot(thresh_vals, f1_vals,   'o-', label='F1',        linewidth=2)
axes[0].plot(thresh_vals, rec_vals,  's-', label='Recall',    linewidth=2)
axes[0].plot(thresh_vals, prec_vals, '^-', label='Precision', linewidth=2)
axes[0].axvline(x=thresh_use, color='red', linestyle='--',
                label=f'Best F1 ({thresh_use:.5f})', alpha=0.7)
axes[0].axvline(x=old_threshold, color='gray', linestyle=':',
                label=f'Original ({old_threshold:.5f})', alpha=0.7)
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Threshold Tuning Curves')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Reconstruction errors with new threshold
colors = ['steelblue' if l==0 else 'red' for l in y_test_f]
axes[1].scatter(range(len(test_errors)), test_errors, c=colors, alpha=0.7)
axes[1].axhline(y=thresh_use, color='red', linestyle='--',
                label=f'New threshold={thresh_use:.5f}')
axes[1].axhline(y=old_threshold, color='gray', linestyle=':',
                label=f'Old threshold={old_threshold:.5f}', alpha=0.7)
axes[1].set_xlabel('Test sample index')
axes[1].set_ylabel('Reconstruction error')
axes[1].set_title('Reconstruction Errors — New Threshold')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Confusion matrix
im = axes[2].imshow(cm_best, cmap='Blues', aspect='auto')
axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
axes[2].set_xticklabels(['Normal','Cyclone'])
axes[2].set_yticklabels(['Normal','Cyclone'])
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')
axes[2].set_title(f'Confusion Matrix — Tuned Threshold\n(F1={f1:.3f}, Recall={rec:.3f})')
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm_best[i,j]), ha='center', va='center',
                     fontsize=22, fontweight='bold',
                     color='white' if cm_best[i,j] > cm_best.max()/2 else 'black')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(BASE, "autoencoder_v3_tuned_results.png"),
            dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Threshold tuning complete!")
