# import numpy as np
#
# def main():
#     # Load data
#     data = np.load('1000G_reqnorm_float64.npy')
#
#     # Print shape info
#     num_genes, num_samples = data.shape
#     print(f"Number of genes (rows): {num_genes}")
#     print(f"Number of samples (columns): {num_samples}")
#
#     # Normalize each row (gene)
#     data_means = data.mean(axis=1, keepdims=True)
#     data_stds = data.std(axis=1, keepdims=True) + 1e-3
#     data = (data - data_means) / data_stds
#
#     # Number of landmark genes
#     num_lm = 943  # Change this if needed
#     X = data[:num_lm, :].T
#     Y = data[num_lm:, :].T
#
#     # Save full X and Y
#     np.save('1000G_X_float64.npy', X)
#     np.save('1000G_Y_float64.npy', Y)
#
#     # Save Y in two parts
#     mid = Y.shape[1] // 2
#     np.save(f'1000G_Y_0-{mid}_float64.npy', Y[:, :mid])
#     np.save(f'1000G_Y_{mid}-{Y.shape[1]}_float64.npy', Y[:, mid:])
#
# if __name__ == '__main__':
#     main()


# import numpy as np
#
# data = np.load('1000G_reqnorm_float64.npy', allow_pickle=False)
# print("Shape:", data.shape)
# print("First few values:\n", data[:5, :5])

# import numpy as np
#
# # Check the raw file before quantile normalization
# data_raw = np.load('1000G_float64.npy')
# print("Raw file shape:", data_raw.shape)
#
# # Check the file after quantile normalization
# data_norm = np.load('1000G_reqnorm_float64.npy')
# print("Normalized file shape:", data_norm.shape)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def main():
    data = np.load('1000G_reqnorm_float64.npy')

    num_genes, num_samples = data.shape
    print(f"Number of genes (rows): {num_genes}")
    print(f"Number of samples (columns): {num_samples}")

    data_means = data.mean(axis=1, keepdims=True)
    data_stds = data.std(axis=1, keepdims=True) + 1e-3
    data = (data - data_means) / data_stds

    num_lm = 943  # Change this if needed
    X = data[:num_lm, :].T   # (samples × genes_lm)
    Y = data[num_lm:, :].T   # (samples × genes_other)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    train_end = int(0.7 * num_samples)
    val_end = int(0.9 * num_samples)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    # np.save('1000G_X_train.npy', X_train)
    # np.save('1000G_Y_train.npy', Y_train)
    #
    # np.save('1000G_X_val.npy', X_val)
    # np.save('1000G_Y_val.npy', Y_val)
    #
    # np.save('1000G_X_test.npy', X_test)
    # np.save('1000G_Y_test.npy', Y_test)

    print(f"Train: {X_train.shape}, Y_train: {Y_train.shape} Val: {X_val.shape}, Test: {X_test.shape}")


# X_tr = np.load('1000G_X_train.npy')
# Y_tr = np.load('1000G_Y_train.npy')
# X_va = np.load('1000G_X_val.npy')
# Y_va = np.load('1000G_Y_val.npy')
# X_te = np.load('1000G_X_test.npy')
# Y_te = np.load('1000G_Y_test.npy')
#
# class GeneDataset(Dataset):
#     def __init__(self, X, Y):
#         self.x = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(Y, dtype=torch.float32)
#     def __len__(self):
#         return self.x.shape[0]
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
# train_ds = GeneDataset(X_tr, Y_tr)
# val_ds   = GeneDataset(X_va, Y_va)
# test_ds  = GeneDataset(X_te, Y_te)
#
# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
# val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
# test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)
#
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
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = GeneModel(X_tr.shape[1], Y_tr.shape[1]).to(device)
#
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
# def train_epoch(loader, model, loss_fn, optimizer):
#     model.train()
#     total_loss = 0
#     for X, y in loader:
#         X, y = X.to(device), y.to(device)
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#     return total_loss / len(loader)
#
# def eval_epoch(loader, model, loss_fn):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for X, y in loader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             loss = loss_fn(pred, y)
#             total_loss += loss.item()
#     return total_loss / len(loader)
#
# epochs = 20
# for epoch in range(epochs):
#     train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
#     val_loss = eval_epoch(val_loader, model, loss_fn)
#     print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
# test_loss = eval_epoch(test_loader, model, loss_fn)
# print(f"Test Loss: {test_loss:.4f}")



if __name__ == '__main__':
    main()
