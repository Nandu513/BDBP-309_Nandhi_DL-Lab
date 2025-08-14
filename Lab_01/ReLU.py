import numpy as np
import matplotlib.pyplot as plt
import random

def relu(x):
    x=[i if i>0 else 0 for i in x]
    return x

x=sorted([random.randint(-10, 10) for _ in range(15)])

y_relu=relu(x)


def relu_derivative(x):
    r=[1 if i>0 else 0 for i in x]
    return r

y_derivative=relu_derivative(x)

plt.figure(figsize=(10, 5))

# Plot sigmoid
plt.subplot(1, 2, 1)
plt.plot(x, y_relu, label="relu", color='blue')
plt.title("relu Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label="Derivative", color='red')
plt.title("Derivative of relu")
plt.xlabel("x")
plt.ylabel("g'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
