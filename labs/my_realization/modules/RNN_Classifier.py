import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List
import copy

class RNN_Classifier(mm.Module):
    def __init__(self, in_features, num_classes, hidden_size, module):
        super().__init__()
        self.input_size = in_features
        self.hidden_size = hidden_size
        self.h1 = None
        self.c1 = None
        self.encoder = module(in_features, hidden_size)
        self.head = mm.Linear(hidden_size, num_classes)
    def compute_output(self, input: np.array) -> np.array:
        self.h1 = np.zeros((1, input.shape[0], self.hidden_size))
        if isinstance(self.encoder, mm.LSTM):
            self.c1 = np.zeros((1, input.shape[0], self.hidden_size))
            out, _ = self.encoder(input, self.h1, self.c1)
        else:
            out = self.encoder(input, self.h1)
        return self.head(out)
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        if isinstance(self.encoder, mm.LSTM):
            return self.encoder.backward(input, self.h1, self.c1, 
                                         self.head.backward(self.encoder.output[0], grad_output)[np.newaxis, :])[0]
        return self.encoder.backward(input, self.h1,
                                      self.head.backward(self.encoder.output, grad_output)[np.newaxis, :])
    def train(self):
        self.head.train()
        self.encoder.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def zero_grad(self):
        self.encoder.zero_grad()
        self.head.zero_grad()
    
    def parameters(self) -> List[np.array]:
        return self.encoder.parameters() + self.head.parameters()

    def parameters_grad(self) -> List[np.array]:
        return self.encoder.parameters_grad() + self.head.parameters_grad()
