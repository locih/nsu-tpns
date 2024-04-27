import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List

class LSTMCell(mm.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.weight_ih = np.random.uniform(-1, 1, (4 * hidden_size, input_size)) / np.sqrt(hidden_size)
        self.weight_hh = np.random.uniform(-1, 1, (4 * hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.bias_ih = np.random.uniform(-1, 1, 4 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.bias_hh = np.random.uniform(-1, 1, 4 * hidden_size) / np.sqrt(hidden_size) if bias else None
        self.hidden_size = hidden_size

        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)
        
        self.ifgo_grad = None
        self.input_gate = None
        self.forget_gate = None
        self.output_gate = None
        self.gain_gate  = None
        self.next_cell_state = None
        self.next_hidden_state = None

    def compute_output(self, input: np.array, hx: np.array, cx: np.array) -> np.array:
        tanh = mm.Tanh()
        sigmoid = mm.Sigmoid()
        A = input @ self.weight_ih.T + hx @ self.weight_hh.T
        if self.bias_hh is not None: 
            A += self.bias_ih + self.bias_hh
        self.input_gate = sigmoid(A[:, 0 : self.hidden_size])
        self.forget_gate = sigmoid(A[:, self.hidden_size : 2 * self.hidden_size])
        self.gain_gate = tanh(A[:, 2 * self.hidden_size : 3 * self.hidden_size])
        self.output_gate = sigmoid(A[:, 3 * self.hidden_size : 4 * self.hidden_size])
        self.next_cell_state = self.forget_gate * cx + self.input_gate * self.gain_gate
        self.next_hidden_state = self.output_gate * tanh(self.next_cell_state)
        return self.next_hidden_state, self.next_cell_state
        
    def compute_grad_input(self, input: np.array, hx: np.array, cx: np.array, grad_hx: np.array, grad_cx: np.array) -> np.array:
        tanh = mm.Tanh()
        tanh_grad = tanh.compute_grad_input(self.next_cell_state, grad_hx * self.output_gate)

        grad_next_cell_state = (grad_cx + tanh_grad) * self.forget_gate

        grad_input_gate = (grad_cx + tanh_grad) * self.gain_gate
        grad_input_gate *= self.input_gate * (1 - self.input_gate)

        grad_forget_gate = (grad_cx + tanh_grad) * cx
        grad_forget_gate *= self.forget_gate * (1 - self.forget_gate)

        grad_output_gate = grad_hx * tanh(self.next_cell_state)
        grad_output_gate *= self.output_gate * (1 - self.output_gate)

        grad_gain_gate = (grad_cx + tanh_grad) * self.input_gate
        grad_gain_gate *= (1 - (self.gain_gate ** 2))

        self.ifgo_grad = np.concatenate([grad_input_gate, grad_forget_gate, grad_gain_gate, grad_output_gate], axis = 1)
        grad_next_hidden_state = self.ifgo_grad @ self.weight_hh
        return grad_next_hidden_state, grad_next_cell_state
    def update_grad_parameters(self, input: np.array, hx: np.array, cx: np.array, grad_hx: np.array, grad_cx: np.array):
        self.grad_weight_ih += self.ifgo_grad.T @ input
        self.grad_weight_hh += self.ifgo_grad.T @ hx
        if self.bias_hh is not None:
            self.grad_bias_ih += np.sum(self.ifgo_grad, axis = 0)
            self.grad_bias_hh += np.sum(self.ifgo_grad, axis = 0)
    
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
