import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

print("📂 Loading data...")
data = np.load(r"C:\Users\Tmart\Desktop\cy\future_tropical.npy")
labels = np.load(r"C:\Users\Tmart\Desktop\cy\labels.npy")

# Normalize
ch_min = data.min(axis=(0, 2, 3), keepdims=True)
ch_max = data.max(axis=(0, 2, 3), keepdims=True)
data_norm = (data - ch_min) / (ch_max - ch_min + 1e-8)

# Resize
data_tensor = torch.from_numpy(data_norm).float()
data_resized = F.interpolate(data_tensor, size=(64, 64), mode='bilinear', align_corners=False)

X = data_resized.numpy()
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_normal = X_train[y_train == 0]
X_train_normal_t = torch.from_numpy(X_train_normal).float()
train_loader = DataLoader(TensorDataset(X_train_normal_t), batch_size=16, shuffle=True)

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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("🏋️ Training (3-5 min)...")
for epoch in range(50):
    model.train()
    total = 0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/50  loss={total/len(train_loader):.5f}")

# Compute threshold
model.eval()
with torch.no_grad():
    recon = model(X_train_normal_t)
    train_errors = ((recon - X_train_normal_t) ** 2).mean(dim=[1,2,3]).numpy()

threshold = float(train_errors.mean() + 2 * train_errors.std())

# Save everything
torch.save(model.state_dict(), r"C:\Users\Tmart\Desktop\cy\autoencoder_model.pt")
np.savez(r"C:\Users\Tmart\Desktop\cy\autoencoder_artifacts.npz",
         threshold=threshold, ch_min=ch_min, ch_max=ch_max)

print(f"\n✅ Saved!")
print(f"   Threshold: {threshold:.5f}")
print(f"   Model file: autoencoder_model.pt")