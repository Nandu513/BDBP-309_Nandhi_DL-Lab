import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# Vision Transformer (ViT) Model Definition
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                 dim=128, depth=6, heads=8, mlp_dim=256, channels=3, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, channels, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.n_patches, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)


# Training and Evaluation Loop
def train_model(model, train_loader, test_loader, epochs=5, lr=3e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%\n")

    return model


# Example: CIFAR-10 dataset (change here for others)
if __name__ == "__main__":
    dataset_name = "CIFAR10"  # change to "MNIST", "FashionMNIST", etc.
    image_size = 32
    num_classes = 10
    channels = 3  # change to 1 for grayscale datasets like MNIST

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load Dataset
    if dataset_name == "MNIST":
        channels = 1
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        image_size = 28

    elif dataset_name == "FashionMNIST":
        channels = 1
        trainset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
        num_classes = 10
        image_size = 28

    elif dataset_name == "CIFAR10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        image_size = 32

    # (For Flowers102, FER2013, Caltech101, GTSRB, TinyImageNet, etc.,
    # replace with corresponding torchvision.datasets or custom Dataset)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # Initialize and Train
    model = VisionTransformer(img_size=image_size,
                              patch_size=4,
                              num_classes=num_classes,
                              channels=channels,
                              dim=128, depth=6, heads=8, mlp_dim=256)
    trained_model = train_model(model, trainloader, testloader, epochs=5)


# | Dataset      | `dataset_name`   | `image_size` | `channels` | `num_classes` |
# | ------------ | ---------------- | ------------ | ---------- | ------------- |
# | MNIST        | `"MNIST"`        | 28           | 1          | 10            |
# | FashionMNIST | `"FashionMNIST"` | 28           | 1          | 10            |
# | CIFAR-10     | `"CIFAR10"`      | 32           | 3          | 10            |
# | GTSRB        | custom loader    | 32           | 3          | 43            |
# | FER2013      | custom loader    | 48           | 1          | 7             |
# | Flowers-102  | custom loader    | 64           | 3          | 102           |
# | TinyImageNet | custom loader    | 64           | 3          | 200           |
# | Caltech-101  | custom loader    | 224          | 3          | 101           |


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CONFIGURATION
dataset_name = "CIFAR10"  # Change this to any dataset name
image_size = 32  # 28 for MNIST, 32 for CIFAR10, 48 for FER2013, etc.
num_classes = 10  # Change based on dataset
rnn_type = "LSTM"  # Options: "RNN", "LSTM", "GRU"

# DATASET LOADING
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1) if dataset_name in ["MNIST", "FashionMNIST",
                                                                    "EMNIST"] else transforms.Lambda(lambda x: x),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_name == "MNIST":
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
elif dataset_name == "FashionMNIST":
    train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
elif dataset_name == "CIFAR10":
    train_data = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
else:
    raise ValueError(f"Dataset {dataset_name} not configured here — please add your dataset loader.")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# MODEL DEFINITION
class RNNImageClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_type="LSTM"):
        super(RNNImageClassifier, self).__init__()
        self.model_type = model_type

        if model_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be 'RNN', 'LSTM', or 'GRU'")

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, h, w = x.size()
        x = x.view(b, h, w * c)  # treat each row as one time step (sequence length = height)

        if self.model_type == "LSTM":
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)

        out = self.fc(out[:, -1, :])  # use last hidden state
        return out


# TRAINING SETUP
input_size = image_size * (1 if dataset_name in ["MNIST", "FashionMNIST", "EMNIST"] else 3)
hidden_size = 128
num_layers = 2

model = RNNImageClassifier(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, num_classes=num_classes, model_type=rnn_type)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TRAINING LOOP
for epoch in range(3):  # You can increase epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/3], Loss: {running_loss / len(train_loader):.4f}")

# EVALUATION
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
