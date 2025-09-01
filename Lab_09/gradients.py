import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DeepNet(nn.Module):
    def __init__(self, n_layers=50, n_neurons=64, activation="tanh", init="default"):
        super().__init__()
        layers = []
        act = nn.Tanh() if activation == "tanh" else nn.ReLU()

        for _ in range(n_layers):
            lin = nn.Linear(n_neurons, n_neurons)

            # Apply initialization
            if init == "xavier":
                nn.init.xavier_uniform_(lin.weight)
            elif init == "he":
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")

            layers.append(lin)
            layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Function to record gradient magnitudes
def get_gradients(activation="tanh", init="default", n_layers=50, n_neurons=64):
    net = DeepNet(n_layers=n_layers, n_neurons=n_neurons, activation=activation, init=init)
    x = torch.randn(1, n_neurons)
    y = net(x).sum()  # scalar loss
    y.backward()

    grads = []
    for name, p in net.named_parameters():
        if "weight" in name:
            grads.append(p.grad.std().item())
    return grads


# Plot side by side
layers = 50
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Tanh plot
for init in ["default", "xavier"]:
    grads = get_gradients(activation="tanh", init=init, n_layers=layers)
    axes[0].plot(range(1, len(grads) + 1), grads, label=f"Tanh-{init}")
axes[0].set_title("Tanh: Vanishing Gradient")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Gradient Std (log scale)")
axes[0].set_yscale("log")
axes[0].legend()

# ReLU plot
for init in ["default", "he"]:
    grads = get_gradients(activation="relu", init=init, n_layers=layers)
    axes[1].plot(range(1, len(grads) + 1), grads, label=f"ReLU-{init}")
axes[1].set_title("ReLU: Exploding Gradient")
axes[1].set_xlabel("Layer")
axes[1].set_yscale("log")
axes[1].legend()

plt.suptitle("Vanishing vs Exploding Gradients\nDefault vs Proper Initialization", fontsize=14)
plt.tight_layout()
plt.show()
