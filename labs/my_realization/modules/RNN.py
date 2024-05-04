import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List
import copy

class RNN(mm.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.module = mm.RNNCell(self.input_size, self.hidden_size, bias = self.bias, nonlinearity=nonlinearity)
        self.modules = None

        self.weight_ih_l0 = self.module.weight_ih
        self.weight_hh_l0 = self.module.weight_hh
        self.bias_ih_l0 = self.module.bias_ih if bias else None
        self.bias_hh_l0 = self.module.bias_hh if bias else None
        
        self.grad_weight_ih_l0 = np.zeros_like(self.weight_ih_l0)
        self.grad_weight_hh_l0 = np.zeros_like(self.weight_hh_l0)
        self.grad_bias_ih_l0 = np.zeros_like(self.bias_ih_l0)
        self.grad_bias_hh_l0 = np.zeros_like(self.bias_hh_l0)
    def compute_output(self, input: np.array, hx: np.array) -> np.array:
        self.modules = [copy.deepcopy(self.module) for _ in range(input.shape[1])]
        y = hx[0]
        for i in range(len(self.modules)):
            y = self.modules[i](input[:,i,:], y)
        return y
    def compute_grad_input(self, input: np.array, hx: np.array, grad_output: np.array) -> np.array:
        grad_input = grad_output[0]
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].backward(input[:, i, :], self.modules[i-1].output, grad_input)
        return self.modules[0].backward(input[:, 0, :], hx[0], grad_input)
    
    def update_grad_parameters(self, input: np.array, hx: np.array, grad_output: np.array):
        for module in self.modules:
            for rnn_grad, rnn_cell_grad in zip(self.parameters_grad(), module.parameters_grad()):
                rnn_grad += rnn_cell_grad

    def train(self):
        if self.modules is None:
            self.module.train()
            return
        for module in self.modules:
            module.train()

    def eval(self):
        if self.modules is None:
            self.module.eval()
            return
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        if self.modules is None:
            self.module.zero_grad()
            return
        self.grad_weight_ih_l0.fill(0)
        self.grad_weight_hh_l0.fill(0)
        if self.bias_ih_l0 is not None:
            self.grad_bias_hh_l0.fill(0)
            self.grad_bias_ih_l0.fill(0)
    def parameters(self) -> List[np.array]:
        if self.bias_hh_l0 is not None:
            return [self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0]
        return [self.weight_ih_l0, self.weight_hh_l0]

    def parameters_grad(self) -> List[np.array]:
        if self.bias_hh_l0 is not None:
            return [self.grad_weight_ih_l0, self.grad_weight_hh_l0, self.grad_bias_ih_l0, self.grad_bias_hh_l0]
        return [self.grad_weight_ih_l0, self.grad_weight_hh_l0]
    
    def __repr__(self) -> str:
        return f'RNN(input_size={self.input_size}, out_features={self.hidden_size}, ' \
               f'bias={not self.bias is None})'
