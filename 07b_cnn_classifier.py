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

# Reproducibility — SAME seed as KNN/HOG/Autoencoder, so the comparison is fair
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load data (identical to 07_autoencoder.py, for a fair comparison)
# ============================================================
print("📂 Loading data...")
data = np.load(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy")  # (365, 5, 60, 113)
labels = np.load(r"C:\Users\Tmart\Desktop\cy\labels.npy")
print(f"   Data: {data.shape}, Cyclones: {labels.sum()}")

# Normalize per channel to [0, 1] — same approach as autoencoder
ch_min = data.min(axis=(0, 2, 3), keepdims=True)
ch_max = data.max(axis=(0, 2, 3), keepdims=True)
data_norm = (data - ch_min) / (ch_max - ch_min + 1e-8)

# Resize to 64x64 — same as autoencoder, for clean conv math
print("📐 Resizing 60x113 → 64x64...")
data_tensor = torch.from_numpy(data_norm).float()
data_resized = F.interpolate(data_tensor, size=(64, 64), mode='bilinear', align_corners=False)
print(f"   New shape: {data_resized.shape}")

# ============================================================
# 2. Train/test split — SAME seed=42 as every other method
# ============================================================
X = data_resized.numpy()
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} days ({y_train.sum()} cyclones)")
print(f"   Test:  {len(X_test)} days ({y_test.sum()} cyclones)")

# IMPORTANT DIFFERENCE vs Autoencoder: here we train SUPERVISED,
# so we use ALL training days (both normal + cyclone), with their labels.
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

# ============================================================
# 3. CNN Classifier architecture
# ============================================================
class CycloneCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Same conv backbone shape as the autoencoder's encoder, for fair comparison
        self.features = nn.Sequential(
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
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (64, 8, 8) → (64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),  # single output: cyclone probability (logit)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier_head(x)

model = CycloneCNNClassifier()
print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 4. Training — class imbalance handled with pos_weight
# ============================================================
# Cyclones are rare (~15% of days), so we up-weight the cyclone class
# in the loss, otherwise the model will just learn to predict "normal" always.
n_normal = (y_train == 0).sum()
n_cyclone = (y_train == 1).sum()
pos_weight = torch.tensor([n_normal / max(n_cyclone, 1)], dtype=torch.float32)
print(f"\n⚖️  Class balance — Normal: {n_normal}, Cyclone: {n_cyclone}, pos_weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
losses = []

print(f"\n🏋️ Training for {EPOCHS} epochs (CPU, takes ~2-4 min)...")
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
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
# 5. Evaluate on TEST set
# ============================================================
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits).squeeze(1).numpy()

y_pred = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 SUPERVISED CNN CLASSIFIER RESULTS")
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
# 6. Save model + metrics (so the app / final report can use them)
# ============================================================
torch.save(model.state_dict(), r"C:\Users\Tmart\Desktop\cy\cnn_classifier_model.pt")
np.save(r"C:\Users\Tmart\Desktop\cy\cnn_classifier_metrics.npy",
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
np.save(r"C:\Users\Tmart\Desktop\cy\cnn_classifier_probs.npy", test_probs)
print(f"\n💾 Saved: cnn_classifier_model.pt, cnn_classifier_metrics.npy")

# ============================================================
# 7. Visualizations
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses, color='steelblue', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('BCE Loss')
axes[0].set_title('Supervised CNN — Training Loss')
axes[0].grid(alpha=0.3)

im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Normal', 'Cyclone'])
axes[1].set_yticklabels(['Normal', 'Cyclone'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix — Supervised CNN')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                      fontsize=20, fontweight='bold',
                      color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\cnn_classifier_results.png", dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 8. Updated comparison table — now ALL 4 methods
# ============================================================
print("\n" + "=" * 70)
print("🏆 FINAL COMPARISON — ALL 4 METHODS")
print("=" * 70)
print(f"{'Method':<25} {'Type':<15} {'Accuracy':<12} {'F1':<10}")
print("-" * 70)
print(f"{'KNN (flatten)':<25} {'Supervised':<15} {'98.6%':<12} {'0.952':<10}")
print(f"{'KNN + HOG':<25} {'Supervised':<15} {'93.1%':<12} {'0.800':<10}")
print(f"{'Supervised CNN':<25} {'Supervised':<15} {f'{acc*100:.1f}%':<12} {f'{f1:.3f}':<10}")
print(f"{'CNN Autoencoder':<25} {'Unsupervised':<15} {'(see 07_autoencoder.py)':<12}")
print("=" * 70)

print("\n✅ Supervised CNN baseline complete!")
print("📊 Aapka comparison table ab sir ki requirement ke according complete hai:")
print("   3 supervised methods (KNN, KNN+HOG, CNN) + 1 unsupervised (Autoencoder)")
