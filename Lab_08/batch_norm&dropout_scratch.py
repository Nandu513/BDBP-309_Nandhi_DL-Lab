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

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable params
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Normalize
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # Use running stats at inference
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

        return out


class LayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return mask * x / (1 - self.p)  # scale
        else:
            return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.bn1 = BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.drop1 = Dropout(0.3)

        self.fc2 = nn.Linear(16, 16)
        self.ln2 = LayerNorm1d(16)
        self.relu2 = nn.ReLU()
        self.drop2 = Dropout(0.3)

        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
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
