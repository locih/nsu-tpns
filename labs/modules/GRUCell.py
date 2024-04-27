import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List

class GRUCell(mm.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (3 * hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (3 * hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, 3 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, 3 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.hidden_size = hidden_size
        self.r = None
        self.z = None
        self.n = None
        self.A = None
        self.hidden_vec = None
        self.input_vec = None

        self.grad_A = None
        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)

        self.cache = None
    def compute_output(self, input: np.array, hx: np.array) -> np.array:
        sigmoid = mm.Sigmoid()
        tanh = mm.Tanh()
        self.input_vec = input @ self.weight_ih.T
        self.hidden_vec = hx @ self.weight_hh.T
        if self.bias_hh is not None: 
            self.input_vec += self.bias_ih
            self.hidden_vec += self.bias_hh
        self.A = self.input_vec + self.hidden_vec
        self.r = sigmoid(self.A[:, 0 : self.hidden_size])
        self.z = sigmoid(self.A[:, self.hidden_size : 2 * self.hidden_size])
        self.n = tanh(self.input_vec[:, 2 * self.hidden_size : 3 * self.hidden_size] + 
                      self.r * self.hidden_vec[:, 2 * self.hidden_size : 3 * self.hidden_size])
        return (1 - self.z) * self.n + self.z * hx
    def compute_grad_input(self, input: np.array, hx: np.array, grad_output: np.array) -> np.array:
        grad_n = (grad_output * (1 - self.z) * (1 - (self.n ** 2)))
        grad_z = (grad_output * (-self.n + hx) * self.z * (1 - self.z))
        grad_r = ((grad_output * (1 - self.z) * (1 - (self.n ** 2))) * 
                  self.hidden_vec[:, 2 * self.hidden_size : 3 * self.hidden_size] * self.r * (1 - self.r))
        self.grad_A = np.concatenate([grad_r, grad_z, grad_n], axis = 1)
        grad_A_h = self.grad_A.copy()
        grad_A_h[:, 2 * self.hidden_size : 3 * self.hidden_size] *= self.r
        return grad_A_h @ self.weight_hh + grad_output * self.z


    def update_grad_parameters(self, input: np.array, hx: np.array, grad_hx: np.array):
        self.grad_weight_ih += self.grad_A.T @ input

        grad_A_weight_hh = self.grad_A.copy()
        grad_A_weight_hh[:, 2 * self.hidden_size : 3 * self.hidden_size] *= self.r
        self.grad_weight_hh += grad_A_weight_hh.T @ hx
        if self.bias_hh is not None:
            self.grad_bias_ih += np.sum(self.grad_A, axis = 0)
            self.grad_bias_hh += np.sum(grad_A_weight_hh, axis = 0)
    
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
