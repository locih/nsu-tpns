import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Module(ABC):
    def __init__(self):
        self.output = None
        self.training = True

    @abstractmethod
    def compute_output(self, input: np.array) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        raise NotImplementedError

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        pass

    def __call__(self, input: np.array) -> np.array:
        return self.forward(input)

    def forward(self, input: np.array) -> np.array:
        self.output = self.compute_output(input)
        return self.output

    def backward(self, input: np.array, grad_output: np.array) -> np.array:
        grad_input = self.compute_grad_input(input, grad_output)
        self.update_grad_parameters(input, grad_output)
        return grad_input

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass

    def parameters(self) -> List[np.array]:
        return []

    def parameters_grad(self) -> List[np.array]:
        return []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Criterion(ABC):
    def __init__(self):
        self.output = None

    @abstractmethod
    def compute_output(self, input: np.array, target: np.array) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        raise NotImplementedError

    def __call__(self, input: np.array, target: np.array) -> float:
        return self.forward(input, target)

    def forward(self, input: np.array, target: np.array) -> float:
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input: np.array, target: np.array) -> np.array:
        grad_input = self.compute_grad_input(input, target)
        return grad_input

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Optimizer(ABC):
    def __init__(self, module: Module):
        self.module = module
        self.state = {}

    def zero_grad(self):
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        raise NotImplementedError
