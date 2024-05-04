import numpy as np
def convolution2d(image, kernel, bias):
    k = kernel.shape[0]
    y, x = image.shape
    y = y - k + 1
    x = x - k + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i, j] = np.sum(image[i : i + k, j : j + k] * kernel)
    return new_image

y = 10
x = 10
image = np.random.randint(0, 255, size=(y, x))
kernel_size = (3, 3)
kernel = np.random.randn(kernel_size[0], kernel_size[1])

bias = np.random.randn()

print(convolution2d(image, kernel, bias))