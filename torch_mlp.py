import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

def z_normalize_images(images):
    mean = images.mean()
    std  = images.std()
    eps  = 1e-8
    return (images - mean) / (std + eps)

# -------------------------------------------------------------------
# 1) Load & prepare your data FROM the split files!
# -------------------------------------------------------------------
train_x = np.load('train_mlp_x.npy')   # shape (N_train, D)
train_y = np.load('train_mlp_y.npy')
test_x  = np.load('test_mlp_x.npy')    # shape (N_test, D)
test_y  = np.load('test_mlp_y.npy')

perm = np.random.permutation(len(train_x))
train_x = train_x[perm]
train_y = train_y[perm]

print("Train class counts:", Counter(train_y))
print("Test  class counts:", Counter(test_y))

# Convert to torch tensors
X_train = torch.from_numpy(train_x).float()
y_train = torch.from_numpy(train_y).long()
X_test  = torch.from_numpy(test_x).float()
y_test  = torch.from_numpy(test_y).long()

# z_normalize using the training set stats!
X_train = z_normalize_images(X_train)  
X_test  = z_normalize_images(X_test)  # (optionally: use train mean/std on test_x for true ML practice)

# Datasets & DataLoaders
batch_size = 64
train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

# -------------------------------------------------------------------
# 2) Define your shallow MLP (update num_classes if needed)
# -------------------------------------------------------------------
class ShallowMLP(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512, num_classes=7):
        print (f"CREATING MLP with hidden_dim = {hidden_dim}")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        return self.fc9(x)  # raw logits

# Adjust input_dim and num_classes if needed!
input_dim = X_train.shape[1]
num_classes = len(np.unique(train_y))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = ShallowMLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

# -------------------------------------------------------------------
# 4) Train & eval functions
# -------------------------------------------------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        _, preds = logits.max(1)
        correct  += (preds == yb).sum().item()
        total    += xb.size(0)
    avg_loss = running_loss / total
    acc      = correct / total
    return avg_loss, acc

def evaluate():
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            _, preds = logits.max(1)
            correct  += (preds == yb).sum().item()
            total    += xb.size(0)
    avg_loss = running_loss / total
    acc      = correct / total
    return avg_loss, acc

# -------------------------------------------------------------------
# 5) Training loop
# -------------------------------------------------------------------
n_epochs = 5
for epoch in range(1, n_epochs+1):
    tr_loss, tr_acc = train_one_epoch()
    tst_loss, tst_acc = evaluate()
    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
          f" Test Loss: {tst_loss:.4f},  Test Acc: {tst_acc:.4f}")