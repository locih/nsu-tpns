import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List

class RNNCell(mm.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'relu'):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, hidden_size) / np.sqrt(hidden_size) if bias else None

        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)

        self.cache = None
    def compute_output(self, input: np.array, hx: np.array) -> np.array:
        relu = mm.ReLU()
        if self.bias_hh is not None: 
            self.cache =  input @ self.weight_ih.T + self.bias_ih + hx @ self.weight_hh.T + self.bias_hh
            return relu(self.cache)
        self.cache = input @ self.weight_ih.T + hx @ self.weight_hh.T
        return relu(self.cache)
    
    def compute_grad_input(self, input: np.array, hx: np.array, grad_output: np.array) -> np.array:
        relu = mm.ReLU()
        ReLU_grad = relu.backward(self.cache, grad_output)
        return ReLU_grad @ self.weight_hh
    def update_grad_parameters(self, input: np.array, hx: np.array, grad_hx: np.array):
        relu = mm.ReLU()
        ReLU_grad = relu.backward(self.cache, grad_hx)
        if self.bias_ih is not None:
            self.grad_bias_ih += np.sum(ReLU_grad, axis = 0)
            self.grad_bias_hh += np.sum(ReLU_grad, axis = 0)
        self.grad_weight_ih += ReLU_grad.T @ input
        self.grad_weight_hh += ReLU_grad.T @ hx
    
    def zero_grad(self):
        self.grad_weight_ih.fill(0)
        self.grad_weight_hh.fill(0)
        if self.bias_ih is not None:
            self.grad_bias_hh.fill(0)
            self.grad_bias_ih.fill(0)
    
    def parameters(self) -> List[np.array]:
        if self.bias_hh is not None:
            return [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        return [self.weight_ih, self.weight_hh]

    def parameters_grad(self) -> List[np.array]:
        if self.bias_hh is not None:
            return [self.grad_weight_ih, self.grad_weight_hh, self.grad_bias_ih, self.grad_bias_hh]
        return [self.grad_weight_ih, self.grad_weight_hh]
    
    def __repr__(self) -> str:
        hidden_size, input_size = self.weight_ih.shape
        return f'RNNCell(input_size={input_size}, out_features={hidden_size}, ' \
               f'bias={not self.bias is None})'
