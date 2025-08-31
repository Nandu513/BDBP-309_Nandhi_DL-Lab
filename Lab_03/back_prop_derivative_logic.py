import numpy as np

def backpropagation(a_list, z_list, w, b, y_true, lr=0.01):
    grads_w = [np.zeros_like(W) for W in w]
    grads_b = [np.zeros_like(B) for B in b]

    delta = a_list[-1] - y_true
    grads_w[-1] = np.outer(delta, a_list[-2])
    grads_b[-1] = delta

    for l in range(len(w) - 2, -1, -1):
        delta = np.dot(w[l+1].T, delta) * relu_derivative(z_list[l])
        grads_w[l] = np.outer(delta, a_list[l])
        grads_b[l] = delta

    for i in range(len(w)):
        print(f"Before update W[{i}]:\n", w[i])
        w[i] -= lr * grads_w[i]
        b[i] -= lr * grads_b[i]
        print(f"After update W[{i}]:\n", w[i])
        print(f"Updated bias[{i}]:\n", b[i])

    return w, b
