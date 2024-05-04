import numpy as np
from numpy.core.multiarray import array as array
from scipy.special import expit, softmax, log_softmax
from .base import Module

class ReLU(Module):
    def compute_output(self, input: np.array) -> np.array:
        return np.where(input > 0, input, 0)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return grad_output * np.where(input > 0, 1, 0)

class Tanh(Module):
    def compute_output(self, input: np.array) -> np.array:
        return np.tanh(input)
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return grad_output * (1 - (self.compute_output(input) ** 2))

class Sigmoid(Module):
    def compute_output(self, input: np.array) -> np.array:
        return expit(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return grad_output * self.compute_output(input) * (1 - self.compute_output(input))


class Softmax(Module):
    def compute_output(self, input: np.array) -> np.array:
        return softmax(input, axis = 1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return (grad_output - np.sum(grad_output * self.compute_output(input), axis = 1, keepdims = True)) * self.compute_output(input)


class LogSoftmax(Module):
    def compute_output(self, input: np.array) -> np.array:
        return log_softmax(input, axis = 1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return grad_output - (np.sum(grad_output, axis = 1, keepdims = True) * softmax(input, axis = 1))
