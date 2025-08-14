import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

#data points as input
x = np.linspace(-10, 10, 1000)

# Compute sigmoid and its derivative
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

# Plotting
plt.figure(figsize=(10, 5))

# Plot sigmoid
plt.subplot(1, 2, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True)
plt.legend()

# Plot derivative
plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label="Derivative", color='red')
plt.title("Derivative of Sigmoid")
plt.xlabel("x")
plt.ylabel("g'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
