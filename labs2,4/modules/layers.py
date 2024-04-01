import numpy as np
from typing import List

from numpy.core.multiarray import array as array
from .base import Module
from scipy import signal

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        if self.bias is not None:
            return input @ self.weight.T + self.bias
        return input @ self.weight.T

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis = 0)
        self.grad_weight += grad_output.T @ input
        super().update_grad_parameters(input, grad_output)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]
        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.weight = np.random.uniform(-1, 1, (out_channels, in_channels, kernel_size, kernel_size)) / np.sqrt(in_channels * kernel_size * kernel_size)
        self.bias = np.random.uniform(-1, 1, out_channels) / np.sqrt(in_channels * kernel_size * kernel_size) if bias else None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
    def compute_output(self, input: np.array) -> np.array:
        Y = np.zeros((input.shape[0], self.out_channels, input.shape[2] - self.kernel_size + 1, input.shape[3] - self.kernel_size + 1))
        for i in range(input.shape[0]):
            for out_channel in range(self.out_channels):
                for in_channel in range(input.shape[1]):
                    Y[i, out_channel] += signal.correlate2d(input[i, in_channel], self.weight[out_channel, in_channel], mode='valid')
                if self.bias is not None:
                    Y[i, out_channel] += self.bias[out_channel]
        return Y
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        grad_input = np.zeros_like(input)
        for k in range(input.shape[0]):
            for i in range(self.weight.shape[0]):
                for j in range(self.weight.shape[1]):
                    grad_input[k, j] += signal.convolve2d(grad_output[k][i], self.weight[i][j], mode='full')
        return grad_input

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        for k in range(input.shape[0]):
            for i in range(self.weight.shape[0]):
                for j in range(self.weight.shape[1]):
                    self.grad_weight[i, j] += signal.correlate2d(input[k, j], grad_output[k, i], mode='valid')
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=(0, 2, 3))
    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]
        return [self.grad_weight]

    def __repr__(self) -> str:
        return f'Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size} ' \
               f'bias={not self.bias is None})'

class AvgPool2d(Module):
    def __init__(self, kernel_size : int):
        super().__init__()
        self.kernel_size = kernel_size

    def compute_output(self, input: np.array) -> np.array:
        return input.reshape(input.shape[0], input.shape[1], input.shape[2] // self.kernel_size, self.kernel_size, input.shape[3] // self.kernel_size,
                             self.kernel_size).mean(axis = (3,5))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        desired_shape = np.repeat(grad_output, 2, axis=2)
        return np.repeat(desired_shape, 2, axis=3) * (1 / (self.kernel_size * self.kernel_size))

    def __repr__(self) -> str:
        return f'AvgPool2d(kernel_size={self.kernel_size})'

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.array) -> np.array:
        y = input
        for module in self.modules:
            y = module(y)
        return y

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        grad_input = grad_output
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].backward(self.modules[i-1].output, grad_input)
        return self.modules[0].backward(input, grad_input)

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
    