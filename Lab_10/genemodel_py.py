import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init


X_tr = np.load('1000G_X_train.npy')
Y_tr = np.load('1000G_Y_train.npy')
X_va = np.load('1000G_X_val.npy')
Y_va = np.load('1000G_Y_val.npy')
X_te = np.load('1000G_X_test.npy')
Y_te = np.load('1000G_Y_test.npy')
half_idx = Y_tr.shape[1] // 2

Y_tr_half = Y_tr[:, :half_idx]
Y_va_half = Y_va[:, :half_idx]
Y_te_half = Y_te[:, :half_idx]

class GeneDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = GeneDataset(X_tr, Y_tr_half)
val_ds   = GeneDataset(X_va, Y_va_half)
test_ds  = GeneDataset(X_te, Y_te_half)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=93, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=47, shuffle=False)

# class GeneModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

class GeneModel(nn.Module):
    def __init__(
            self,
            input_dim: int,
            layer_dims: list,
            output_dim: int,
            use_batchnorm: bool = False,
            dropout_prob: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in layer_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = GeneModel(X_tr.shape[1], Y_tr.shape[1]).to(device)

def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

# out_shape=Y_tr.shape[1]
# out_dim = out_shape // 2

model = GeneModel(
    input_dim=X_tr.shape[1],
    layer_dims=[2048, 2048, 2048, 1024, 1024, 1024, 512],
    output_dim=half_idx,
    use_batchnorm=True,
    dropout_prob=0.5).to(device)

model.apply(weights_init_xavier)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)

epochs = 512
for epoch in range(epochs):
    train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
    val_loss = eval_epoch(val_loader, model, loss_fn)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

test_loss = eval_epoch(test_loader, model, loss_fn)
print(f"Test Loss: {test_loss:.4f}")

