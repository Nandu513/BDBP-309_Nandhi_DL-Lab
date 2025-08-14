import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Derivative of sigmoid function
def tanh_derivative(x):
    s = tanh(x)
    return 1 - np.square(s)

#data points as input
x = np.linspace(-10, 10, 1000)

# Compute sigmoid and its derivative
y_tanh = tanh(x)
y_derivative = tanh_derivative(x)


plt.figure(figsize=(10, 5))

# Plot sigmoid
plt.subplot(1, 2, 1)
plt.plot(x, y_tanh, label="tanh", color='blue')
plt.title("tanh Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True)
plt.legend()

# Plot derivative
plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label="Derivative", color='red')
plt.title("Derivative of tanh")
plt.xlabel("x")
plt.ylabel("g'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
