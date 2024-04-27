import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List
import copy

class LSTM(mm.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.module = mm.LSTMCell(self.input_size, self.hidden_size, bias = self.bias)
        self.modules = None

        self.weight_ih_l0 = self.module.weight_ih
        self.weight_hh_l0 = self.module.weight_hh
        self.bias_ih_l0 = self.module.bias_ih if bias else None
        self.bias_hh_l0 = self.module.bias_hh if bias else None
        
        self.grad_weight_ih_l0 = np.zeros_like(self.weight_ih_l0)
        self.grad_weight_hh_l0 = np.zeros_like(self.weight_hh_l0)
        self.grad_bias_ih_l0 = np.zeros_like(self.bias_ih_l0)
        self.grad_bias_hh_l0 = np.zeros_like(self.bias_hh_l0)
    def compute_output(self, input: np.array, hx: np.array, cx: np.array) -> np.array:
        self.modules = [copy.deepcopy(self.module) for _ in range(input.shape[1])]
        hx = hx[0]
        cx = cx[0]
        for i in range(len(self.modules)):
            hx, cx = self.modules[i](input[:,i,:], hx, cx)
        return hx, cx
    def compute_grad_input(self, input: np.array, hx: np.array, cx: np.array, grad_hx: np.array) -> np.array:
        grad_hx = grad_hx[0]
        grad_cx = np.zeros_like(grad_hx)
        for i in range(len(self.modules) - 1, 0, -1):
            h_t, c_t = self.modules[i-1].output
            grad_hx, grad_cx = self.modules[i].backward(input[:, i, :], h_t, c_t, grad_hx, grad_cx)
        return self.modules[0].backward(input[:, 0, :], hx[0], cx[0], grad_hx, grad_cx)
    
    def update_grad_parameters(self, input: np.array, hx: np.array, cx: np.array, grad_hx: np.array):
        for module in self.modules:
            for rnn_grad, rnn_cell_grad in zip(self.parameters_grad(), module.parameters_grad()):
                rnn_grad += rnn_cell_grad
    def __getitem__(self, item):
        return self.modules[item]

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
