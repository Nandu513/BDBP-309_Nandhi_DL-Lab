# Full Fine-tuning of ResNet18 on CIFAR-10 (all layers trainable)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transforms & loaders (same as yours)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Build model for FULL fine-tuning
def get_model_full_finetune(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)          # pretrained backbone
    model.fc = nn.Linear(model.fc.in_features, num_classes)    # replace head
    # Ensure ALL params are trainable:
    for param in model.parameters():
        param.requires_grad = True
    return model.to(device)

# Training loop (optimizer uses all params)
def train_full_finetune(model, trainloader, testloader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        # ----- train -----
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects / len(trainloader.dataset)
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # ----- val -----
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(testloader.dataset)
        epoch_acc = running_corrects / len(testloader.dataset)
        val_loss_hist.append(epoch_loss)
        val_acc_hist.append(epoch_acc)
        print(f"Val   Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return (train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist)

# Create model and run full fine-tuning
model_full = get_model_full_finetune(num_classes=10)
train_loss, train_acc, val_loss, val_acc = train_full_finetune(model_full, trainloader, testloader, epochs=5, lr=1e-4)

# Quick plot
plt.figure(figsize=(8,4))
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Full Fine-tuning (all layers trainable)")
plt.show()



# Epoch 1/5
# Train Loss: 0.3192, Acc: 0.8952
# Val   Loss: 0.2026, Acc: 0.9313

# Epoch 2/5
# Train Loss: 0.0955, Acc: 0.9699
# Val   Loss: 0.1940, Acc: 0.9338

