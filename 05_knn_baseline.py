import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, confusion_matrix, classification_report)
import os

# Load data + labels
print("📂 Loading tropical data + labels...")
data = np.load(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy")
labels = np.load(r"C:\Users\Tmart\Desktop\cy\labels.npy")
print(f"   Data shape: {data.shape}")
print(f"   Labels: {labels.shape}, Cyclones: {labels.sum()}")

# === Whiteboard Step 1: FLATTEN ===
# Each day (5, 60, 113) → 1D vector (33,900 features)
X = data.reshape(data.shape[0], -1)
y = labels
print(f"\n📐 Flattened shape: {X.shape}")
print(f"   Features per day: {X.shape[1]:,}")

# === Train/test split (80/20, stratified to preserve class balance) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} days ({y_train.sum()} cyclones)")
print(f"   Test:  {len(X_test)} days ({y_test.sum()} cyclones)")

# === Normalize (KNN distance ke liye essential) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === KNN ke liye different k values try karein ===
print("\n🔍 Trying different k values...")
k_values = [1, 3, 5, 7, 9, 11]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results.append((k, acc, f1))
    print(f"   k={k}: Accuracy={acc:.3f}, F1={f1:.3f}")

# Best k
best_k = max(results, key=lambda x: x[2])[0]
print(f"\n🏆 Best k = {best_k}")

# === Final model with best k ===
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

# === Detailed metrics ===
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 KNN BASELINE RESULTS")
print("=" * 60)
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Predicted Normal | Predicted Cyclone")
print(f"  Actual Normal   {cm[0,0]:>15} | {cm[0,1]:>17}")
print(f"  Actual Cyclone  {cm[1,0]:>15} | {cm[1,1]:>17}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Cyclone'], zero_division=0))

# Save results
np.save(r"C:\Users\Tmart\Desktop\cy\knn_baseline_metrics.npy",
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'k': best_k})
print(f"💾 Saved metrics to: knn_baseline_metrics.npy")

# === Visualize confusion matrix + k tuning ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: K tuning
ks = [r[0] for r in results]
accs = [r[1] for r in results]
f1s = [r[2] for r in results]
axes[0].plot(ks, accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
axes[0].plot(ks, f1s, 's-', label='F1 Score', linewidth=2, markersize=8)
axes[0].axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
axes[0].set_xlabel('k (number of neighbors)')
axes[0].set_ylabel('Score')
axes[0].set_title('KNN — k Tuning')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Confusion matrix
im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Normal', 'Cyclone'])
axes[1].set_yticklabels(['Normal', 'Cyclone'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix (k={best_k})')

# Add numbers
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=20, fontweight='bold',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\knn_results.png", dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ KNN baseline complete!")
print(f"\n👉 Sir ke whiteboard ka pehla step done — accuracy: {acc*100:.1f}%")
print("   Next step: feature extraction (HOG/CNN features) try karke improve karenge!")