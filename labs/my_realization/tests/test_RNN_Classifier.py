import sys
from .test_base import test_module
from torch import nn
import torch
import modules as mm
sys.path.append('..')
from .test_base import assert_almost_equal
import numpy as np

input_shapes = [(64, 5, 9)]
num_tests = 100
random_seed = 1

class RNN_Classifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden_size):
        super().__init__()
        self.encoder = nn.RNN(input_size = in_features, hidden_size = hidden_size, batch_first = True, nonlinearity='relu')
        self.head = nn.Linear(hidden_size, num_classes)
    def forward(self, x, h):
        _, out = self.encoder(x, h)
        return self.head(out[-1])

def _test_RNN_Classifier(custom_module, torch_module, input_shape, outer_iters=100,
                     inner_iters=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    for i in range(outer_iters):
        module1 = custom_module(input_shape[2], 3, 32, mm.RNN)
        module2 = torch_module(input_shape[2], 3, 32)
        debug_msg = f'Error in RNN_Classifier '
        
        for param1, param2 in zip(module1.encoder.parameters(), module2.encoder.parameters()):
            param2.data = torch.from_numpy(param1)
        for param1, param2 in zip(module1.head.parameters(), module2.head.parameters()):
            param2.data = torch.from_numpy(param1)
        
        for _ in range(inner_iters):
            x1 = np.random.randn(*input_shape)
            x2 = torch.from_numpy(x1)
            h1 = np.zeros((1, input_shape[0], 32))
            h2 = torch.from_numpy(h1)
            h2.requires_grad = True
            y1 = module1(x1)
            y2 = module2(x2, h2)
            assert y1.dtype == x1.dtype
            assert_almost_equal(y1, y2.detach().numpy(), debug_msg + 'forward pass: {}')
            grad_output = np.random.randn(*y2.shape)
            
            y2.backward(torch.from_numpy(grad_output))
            grad_input = module1.backward(x1, grad_output)
            assert_almost_equal(h2.grad.numpy()[0], grad_input, debug_msg + 'input grad: {}')

            for grad, param in zip(module1.encoder.parameters_grad(), module2.encoder.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad encoder: {}')
            for grad, param in zip(module1.head.parameters_grad(), module2.head.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad head: {}')


def test_RNN_Classifier():
    print('test_RNN_CLassifier ... ', end='')
    for input_shape in input_shapes:
        _test_RNN_Classifier(
            mm.RNN_Classifier, RNN_Classifier, input_shape,
            outer_iters=num_tests, random_seed=input_shape[0] + random_seed
        )
    print('OK')
