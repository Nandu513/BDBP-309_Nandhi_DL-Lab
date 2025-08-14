import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x):
    a = 0.1
    x=[i if i>0 else a*i for i in x]
    return x

x=np.linspace(-10,10)

print(x)
y_relu=leaky_relu(x)


def lrelu_derivative(x):
    r=[1 if i>0 else 0.1 for i in x]
    return r

y_derivative=lrelu_derivative(x)

plt.figure(figsize=(10, 5))

# Plot sigmoid
plt.subplot(1, 2, 1)
plt.plot(x, y_relu, label="leaky relu", color='blue')
plt.title("leaky relu Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label="Derivative", color='red')
plt.title("Derivative of leaky relu")
plt.xlabel("x")
plt.ylabel("g'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
