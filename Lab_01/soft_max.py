import numpy as np
import matplotlib.pyplot as plt


def soft_max(x):
    deno=sum(np.exp(i) for i in x)
    s=[np.exp(i)/deno for i in x]
    return s

x=np.linspace(2,16)
y_soft=soft_max(x)


def softmax_derivative(x):
    s=soft_max(x)
    d=[i*(1-i) for i in s]
    return d

y_derivative=softmax_derivative(x)

plt.figure(figsize=(10, 5))

# Plot sigmoid
plt.subplot(1, 2, 1)
plt.plot(x, y_soft, label="soft max", color='blue')
plt.title("soft max Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True)
plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(x, y_derivative, label="Derivative", color='red')
# plt.plot(x,y_line,label="Derivative", color='green')
# plt.title("Derivative of soft max")
# plt.xlabel("x")
# plt.ylabel("g'(x)")
# plt.grid(True)
# plt.legend()

plt.tight_layout()
plt.show()
