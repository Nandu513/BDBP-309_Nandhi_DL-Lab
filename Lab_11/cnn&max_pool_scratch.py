import numpy as np

np.random.seed(42)
image = np.random.randint(0, 256, (32, 32))

print("Input Image Shape:", image.shape)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print("Kernel:\n", kernel)
def conv2d(image, kernel, stride=1, padding=0):

    # Apply padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    kernel_size = kernel.shape[0]
    H, W = image.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(0, out_h):
        for j in range(0, out_w):
            region = image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output

conv_output = conv2d(image, kernel, stride=1, padding=0)
print("Convolution Output Shape:", conv_output.shape)

def maxpool2d(image, pool_size=2, stride=2):
    H, W = image.shape
    out_h = H // stride
    out_w = W // stride

    pooled = np.zeros((out_h, out_w))

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            region = image[i:i + pool_size, j:j + pool_size]
            pooled[i // stride, j // stride] = np.max(region)

    return pooled

pool_output = maxpool2d(conv_output, pool_size=2, stride=2)
print("MaxPool Output Shape:", pool_output.shape)
