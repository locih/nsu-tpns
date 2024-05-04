import numpy as np
from numpy.core.multiarray import array as array
import modules as mm
from typing import List

class Lenet(mm.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mm.Sequential(
            mm.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5),
            mm.ReLU(),
            mm.AvgPool2d(2),
            mm.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5),
            mm.ReLU(),
            mm.AvgPool2d(2),
            mm.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5)
        )
        self.head = mm.Sequential(
            mm.Linear(in_features = 120, out_features = 84),
            mm.ReLU(),
            mm.Linear(in_features = 84, out_features = 10),
        )
    def compute_output(self, input: np.array) -> np.array:
        out = self.encoder(input).reshape(input.shape[0], -1)
        return self.head(out)
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        return self.encoder.backward(input,
                                      self.head.backward(self.encoder.output.reshape(input.shape[0], -1), grad_output).reshape(self.encoder.output.shape))
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
