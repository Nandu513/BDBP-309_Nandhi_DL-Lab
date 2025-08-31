import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def feed_forward(v, w, b):
    a_list = [v]
    z_list = []

    for i in range(len(w)):
        z = np.dot(w[i], a_list[-1]) + b[i]
        z_list.append(z)

        if i == len(w) - 1:
            a = softmax(z)
        else:
            a = relu(z)

        a_list.append(a)

    return a_list, z_list

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


def train(X, Y, n, epochs=1000, lr=0.1):
    w = [np.random.randn(n[i+1], n[i]) * 0.01 for i in range(len(n)-1)]
    b = [np.zeros(n[i+1]) for i in range(len(n)-1)]

    for epoch in range(epochs):
        total_loss = 0
        for x, y_true in zip(X, Y):
            # Forward
            a_list, z_list = feed_forward(x, w, b)
            loss = cross_entropy(a_list[-1], y_true)
            total_loss += loss

            w, b = backpropagation(a_list, z_list, w, b, y_true, lr)

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(X):.4f}")

    return w, b

def predict(X, w, b):
    preds = []
    for x in X:
        a_list, _ = feed_forward(x, w, b)
        preds.append(np.argmax(a_list[-1]))
    return preds

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[1,0],[0,1],[0,1],[1,0]])

    n = [2, 4, 2]

    w, b = train(X, Y, n, epochs=5, lr=0.1)

    preds = predict(X, w, b)
    print("Predictions:", preds)

if __name__ == "__main__":
    main()
