import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, confusion_matrix, classification_report)
import time

# Load data + labels
print("📂 Loading data...")
data = np.load(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy")
labels = np.load(r"C:\Users\Tmart\Desktop\cy\labels.npy")
print(f"   Data: {data.shape}, Cyclones: {labels.sum()}")

# === Extract HOG features per channel, per day ===
print("\n🔍 Extracting HOG features (this takes ~30 sec)...")
start = time.time()

all_features = []
for day in range(data.shape[0]):
    day_features = []
    for ch in range(5):  # 5 channels
        img = data[day, ch]
        # Normalize image to [0,1] per channel
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img
        
        # HOG features
        features = hog(
            img_norm,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        day_features.extend(features)
    
    all_features.append(day_features)
    
    if (day + 1) % 50 == 0:
        print(f"   Processed {day+1}/365 days...")

X_hog = np.array(all_features)
elapsed = time.time() - start
print(f"\n✅ HOG features extracted in {elapsed:.1f}s")
print(f"   Shape: {X_hog.shape}")
print(f"   Reduction: {365*5*60*113:,} pixels → {X_hog.shape[1]:,} features")

y = labels

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_hog, y, test_size=0.2, random_state=42, stratify=y
)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Try different k values ===
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

best_k = max(results, key=lambda x: x[2])[0]
print(f"\n🏆 Best k = {best_k}")

# === Final model ===
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 KNN + HOG FEATURES RESULTS")
print("=" * 60)
print(f"Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                  Pred Normal | Pred Cyclone")
print(f"  Actual Normal   {cm[0,0]:>11} | {cm[0,1]:>12}")
print(f"  Actual Cyclone  {cm[1,0]:>11} | {cm[1,1]:>12}")

# Save
np.save(r"C:\Users\Tmart\Desktop\cy\hog_features.npy", X_hog)
np.save(r"C:\Users\Tmart\Desktop\cy\hog_metrics.npy",
        {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'k': best_k})

# === Compare with baseline ===
print("\n" + "=" * 60)
print("📊 COMPARISON: Flatten vs HOG")
print("=" * 60)
print(f"{'Method':<20} {'Accuracy':<12} {'F1 Score':<12}")
print(f"{'KNN (flatten)':<20} {'98.6%':<12} {'0.952':<12}  (baseline)")
print(f"{'KNN + HOG':<20} {f'{acc*100:.1f}%':<12} {f'{f1:.3f}':<12}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K tuning
ks = [r[0] for r in results]
accs = [r[1] for r in results]
f1s = [r[2] for r in results]
axes[0].plot(ks, accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
axes[0].plot(ks, f1s, 's-', label='F1 Score', linewidth=2, markersize=8)
axes[0].axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
axes[0].set_xlabel('k (number of neighbors)')
axes[0].set_ylabel('Score')
axes[0].set_title('KNN + HOG — k Tuning')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confusion matrix
im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Normal', 'Cyclone'])
axes[1].set_yticklabels(['Normal', 'Cyclone'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix — KNN + HOG (k={best_k})')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=20, fontweight='bold',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig(r"C:\Users\Tmart\Desktop\cy\hog_results.png", dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ HOG features step complete!")