import numpy as np
from .base import Criterion
from .activations import LogSoftmax
from scipy.special import softmax

class MSELoss(Criterion):
    def compute_output(self, input: np.array, target: np.array) -> float:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.sum(np.power((input - target), 2)) / (input.shape[0] * input.shape[1])

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return (2 / (input.shape[0] * input.shape[1])) * (input - target)

class CrossEntropyLoss(Criterion):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        return (-1 / input.shape[0]) * np.sum(input[np.arange(input.shape[0]), target] - np.log(np.sum(np.exp(input), axis = 1)))

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        return (-1 / input.shape[0]) * (np.where(np.arange(input.shape[1]) == target[:, None], 1, 0) - softmax(input, axis = 1))
