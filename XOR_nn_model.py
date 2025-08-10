import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class XORDataset(Dataset):
    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)
        self.x = self.data[['x1','x2']].values.astype('float32')
        self.y = self.data['y'].values.astype('int64')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

dataset = XORDataset("XOR_data.csv")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,2),
            nn.ReLU(),
            nn.Linear(2,2)
        )

    def forward(self,x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = XORModel().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")


def test(dataloader, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted_labels = pred.argmax(1)
            correct += (predicted_labels == y).sum().item()
    accuracy = correct / len(dataloader.dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    train(dataloader, model, loss_fn, optimizer)
    test(dataloader, model)
    print('---------------------')

