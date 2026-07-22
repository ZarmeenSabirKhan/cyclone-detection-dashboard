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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load data
# ============================================================
print("📂 Loading data...")
data = np.load(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy")  # (365, 5, 60, 113)
labels = np.load(r"C:\Users\Tmart\Desktop\cy\labels.npy")
print(f"   Data: {data.shape}, Cyclones: {labels.sum()}")

# Normalize per channel to [0, 1]
ch_min = data.min(axis=(0, 2, 3), keepdims=True)
ch_max = data.max(axis=(0, 2, 3), keepdims=True)
data_norm = (data - ch_min) / (ch_max - ch_min + 1e-8)

# Resize to 64x64 for cleaner conv math
print("📐 Resizing 60x113 → 64x64...")
data_tensor = torch.from_numpy(data_norm).float()
data_resized = F.interpolate(data_tensor, size=(64, 64), mode='bilinear', align_corners=False)
print(f"   New shape: {data_resized.shape}")

# ============================================================
# 2. Train/test split (SAME as KNN — seed 42)
# ============================================================
X = data_resized.numpy()
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} days, Test: {len(X_test)} days")

# Filter TRAIN to NORMAL days only (this is the key — model learns "normal")
X_train_normal = X_train[y_train == 0]
print(f"   Training on {len(X_train_normal)} normal days only")

# Convert to tensors
X_train_normal_t = torch.from_numpy(X_train_normal).float()
X_test_t = torch.from_numpy(X_test).float()

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_normal_t), batch_size=16, shuffle=True)

# ============================================================
# 3. Autoencoder Architecture
# ============================================================
class CycloneAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: (5, 64, 64) → (64, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        # Decoder: (64, 8, 8) → (5, 64, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8→16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 5, 3, stride=2, padding=1, output_padding=1),   # 32→64
            nn.Sigmoid(),  # output in [0, 1]
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = CycloneAutoencoder()
print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 4. Training
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 50
losses = []

print(f"\n🏋️ Training for {EPOCHS} epochs (CPU, takes ~3-5 min)...")
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1:>3}/{EPOCHS}  loss = {avg_loss:.5f}")

elapsed = time.time() - start
print(f"\n✅ Training done in {elapsed:.1f}s")

# ============================================================
# 5. Compute reconstruction errors on TEST set
# ============================================================
model.eval()
with torch.no_grad():
    recon_test = model(X_test_t)
    # Mean squared error per sample
    errors = ((recon_test - X_test_t) ** 2).mean(dim=[1, 2, 3]).numpy()

# Also compute on training normal days (for threshold)
with torch.no_grad():
    recon_train = model(X_train_normal_t)
    train_errors = ((recon_train - X_train_normal_t) ** 2).mean(dim=[1, 2, 3]).numpy()

# Threshold: mean + 2*std of training errors
threshold = train_errors.mean() + 2 * train_errors.std()
print(f"\n🎯 Threshold (train mean + 2*std): {threshold:.5f}")

# Predict cyclones (high reconstruction error)
y_pred = (errors > threshold).astype(int)

# ============================================================
# 6. Evaluate
# ============================================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 CNN AUTOENCODER RESULTS (Unsupervised)")
print("=" * 60)
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Pred Normal | Pred Cyclone")
print(f"  Actual Normal   {cm[0,0]:>11} | {cm[0,1]:>12}")
print(f"  Actual Cyclone  {cm[1,0]:>11} | {cm[1,1]:>12}")

# ============================================================
# 7. FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("🏆 FINAL COMPARISON — ALL METHODS")
print("=" * 70)
print(f"{'Method':<25} {'Type':<15} {'Accuracy':<12} {'F1':<10}")
print("-" * 70)
print(f"{'KNN (flatten)':<25} {'Supervised':<15} {'98.6%':<12} {'0.952':<10}")
print(f"{'KNN + HOG':<25} {'Supervised':<15} {'93.1%':<12} {'0.800':<10}")
print(f"{'CNN Autoencoder':<25} {'Unsupervised':<15} {f'{acc*100:.1f}%':<12} {f'{f1:.3f}':<10}")
print("=" * 70)

# Save
np.save(r"C:\Users\Tmart\Desktop\cy\autoencoder_errors.npy", errors)
np.save(r"C:\Users\Tmart\Desktop\cy\autoencoder_metrics.npy",
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

# ============================================================
# 8. Visualizations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Loss curve
axes[0,0].plot(losses, color='steelblue', linewidth=2)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Reconstruction Loss')
axes[0,0].set_title('Training Loss')
axes[0,0].grid(alpha=0.3)

# (b) Reconstruction errors per test sample (colored by true label)
colors = ['steelblue' if l == 0 else 'red' for l in y_test]
axes[0,1].scatter(range(len(errors)), errors, c=colors, alpha=0.7)
axes[0,1].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
axes[0,1].set_xlabel('Test sample index')
axes[0,1].set_ylabel('Reconstruction error')
axes[0,1].set_title('Reconstruction Error (red = actual cyclones)')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# (c) Confusion matrix
im = axes[1,0].imshow(cm, cmap='Blues', aspect='auto')
axes[1,0].set_xticks([0, 1]); axes[1,0].set_yticks([0, 1])
axes[1,0].set_xticklabels(['Normal', 'Cyclone'])
axes[1,0].set_yticklabels(['Normal', 'Cyclone'])
axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Actual')
axes[1,0].set_title(f'Confusion Matrix — Autoencoder')
for i in range(2):
    for j in range(2):
        axes[1,0].text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=22, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1,0])

# (d) Method comparison bar chart
methods = ['KNN\nflatten', 'KNN\n+ HOG', 'CNN\nAutoencoder']
accs = [98.6, 93.1, acc*100]
f1s = [0.952*100, 0.800*100, f1*100]
x = np.arange(len(methods))
width = 0.35
axes[1,1].bar(x - width/2, accs, width, label='Accuracy (%)', color='steelblue')
axes[1,1].bar(x + width/2, f1s, width, label='F1 × 100', color='coral')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(methods)
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Final Method Comparison')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3, axis='y')
axes[1,1].set_ylim(0, 110)

plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\autoencoder_results.png", dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ CNN Autoencoder complete!")
print("📊 Project ka main goal achieve ho gaya 🎉")