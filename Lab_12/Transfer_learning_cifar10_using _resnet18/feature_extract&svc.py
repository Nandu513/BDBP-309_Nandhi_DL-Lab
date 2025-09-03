# CIFAR10 Feature Extraction + SVM Classifier
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Device (only used for feature extraction)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------- Data -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ----------- Feature Extractor -----------
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # take all layers except the last fc
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # remove last FC

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)   # flatten to (batch, 512)

# Load pretrained ResNet18
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False

feature_extractor = FeatureExtractor(resnet).to(device)

# ----------- Extract Features -----------
def extract_features(dataloader):
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = feature_extractor(images)   # (batch, 512)
            feats.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

print("Extracting training features...")
X_train, y_train = extract_features(trainloader)
print("Extracting test features...")
X_test, y_test = extract_features(testloader)

print("Train features:", X_train.shape, " Test features:", X_test.shape)

# ----------- Train SVM -----------
print("Training SVM classifier...")
svc = SVC(kernel='rbf', C=10, gamma='scale')  # you can tune hyperparameters
svc.fit(X_train, y_train)

# ----------- Evaluate -----------
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))


# Using device: cpu
# Extracting training features...
# Extracting test features...
# Train features: (50000, 512)  Test features: (10000, 512)
# Training SVM classifier...
#
# Test Accuracy: 0.8692

# Classification Report:
#               precision    recall  f1-score   support
#
#        plane       0.88      0.92      0.90      1000
#          car       0.92      0.92      0.92      1000
#         bird       0.83      0.82      0.83      1000
#          cat       0.76      0.77      0.76      1000
#         deer       0.82      0.86      0.84      1000
#          dog       0.83      0.80      0.82      1000
#         frog       0.89      0.91      0.90      1000
#        horse       0.92      0.86      0.89      1000
#         ship       0.92      0.91      0.92      1000
#        truck       0.93      0.92      0.92      1000
#
#     accuracy                           0.87     10000
#    macro avg       0.87      0.87      0.87     10000
# weighted avg       0.87      0.87      0.87     10000

