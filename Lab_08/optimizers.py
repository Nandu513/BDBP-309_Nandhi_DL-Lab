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
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_model(optimizer_name, optimizer_class, optimizer_params, epochs=50):
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate accuracy
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                predicted = pred.argmax(1)
                correct += (predicted == y).sum().item()
        acc = correct / len(dataset)

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        acc_history.append(acc)

    return loss_history, acc_history


optimizers = {
    "SGD": (torch.optim.SGD, {"lr": 0.1}),
    "SGD+Momentum": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
    "NAG": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
    "AdaGrad": (torch.optim.Adagrad, {"lr": 0.1}),
    "RMSProp": (torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9}),
    "Adam": (torch.optim.Adam, {"lr": 0.01})
}

results = {}

for name, (opt_class, opt_params) in optimizers.items():
    print(f"\nTraining with {name}...")
    loss_hist, acc_hist = train_model(name, opt_class, opt_params, epochs=50)
    results[name] = {"loss": loss_hist, "acc": acc_hist}
    print(f"Final Accuracy ({name}): {acc_hist[-1]*100:.2f}%")


plt.figure(figsize=(10,5))
for name, vals in results.items():
    plt.plot(vals["loss"], label=name)
plt.title("Loss Curves for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
for name, vals in results.items():
    plt.plot(vals["acc"], label=name)
plt.title("Accuracy Curves for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
