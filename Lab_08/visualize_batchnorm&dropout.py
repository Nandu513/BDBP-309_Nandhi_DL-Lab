import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class XORDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.x = self.data[['x1', 'x2']].values.astype('float32')
        self.y = self.data['y'].values.astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


dataset = XORDataset("XOR_data.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # bigger batch for better histograms


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
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
        self.last_mask = None

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            self.last_mask = mask
            return mask * x / (1 - self.p)
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


# activations = {}
#
# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = output.detach().cpu()
#     return hook
#
# model = NeuralNetwork()
# model.bn1.register_forward_hook(get_activation("BatchNorm1"))
# model.ln2.register_forward_hook(get_activation("LayerNorm2"))
# model.drop1.register_forward_hook(get_activation("Dropout1"))
# model.drop2.register_forward_hook(get_activation("Dropout2"))


activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

model = NeuralNetwork()
model.fc1.register_forward_hook(get_activation("FC1 (before BN)"))
model.bn1.register_forward_hook(get_activation("BatchNorm1 (after)"))

model.fc2.register_forward_hook(get_activation("FC2 (before LN)"))
model.ln2.register_forward_hook(get_activation("LayerNorm2 (after)"))

model.drop1.register_forward_hook(get_activation("Dropout1"))
model.drop2.register_forward_hook(get_activation("Dropout2"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:  # visualize first batch
            plot_activations(epoch, batch_idx)


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


# def plot_activations(epoch, batch_idx):
#     for name, act in activations.items():
#         plt.figure()
#         plt.hist(act.view(-1).numpy(), bins=20, color="skyblue", edgecolor="black")
#         plt.title(f"{name} - Epoch {epoch+1}, Batch {batch_idx+1}")
#         plt.xlabel("Activation Value")
#         plt.ylabel("Frequency")
#         plt.show()

def plot_activations(epoch, batch_idx):
    for pair in [
        ("FC1 (before BN)", "BatchNorm1 (after)"),
        ("FC2 (before LN)", "LayerNorm2 (after)")
    ]:
        if pair[0] in activations and pair[1] in activations:
            plt.figure(figsize=(10,4))

            # Before normalization
            plt.subplot(1, 2, 1)
            plt.hist(activations[pair[0]].view(-1).numpy(), bins=20, color="salmon", edgecolor="black")
            plt.title(f"{pair[0]} - Epoch {epoch+1}, Batch {batch_idx+1}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")

            # After normalization
            plt.subplot(1, 2, 2)
            plt.hist(activations[pair[1]].view(-1).numpy(), bins=20, color="skyblue", edgecolor="black")
            plt.title(f"{pair[1]} - Epoch {epoch+1}, Batch {batch_idx+1}")
            plt.xlabel("Activation Value")

            plt.tight_layout()
            plt.show()

    for name in ["Dropout1", "Dropout2"]:
        if name in activations:
            plt.figure()
            plt.hist(activations[name].view(-1).numpy(), bins=20, color="green", edgecolor="black")
            plt.title(f"{name} - Epoch {epoch+1}, Batch {batch_idx+1}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.show()

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train(dataloader, model, loss_fn, optimizer, epoch)
    test(dataloader, model)
    print("---------------------")
