import sys
from .test_base import test_module
from torch import nn
import torch
sys.path.append('..')
from modules.Lenet import Lenet
from .test_base import assert_almost_equal
import numpy as np

input_shapes = [(64, 1, 32, 32)]
num_tests = 5
random_seed = 1

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, dtype=torch.float64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, dtype=torch.float64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, dtype=torch.float64)
        )
        self.head = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(in_features = 84, out_features = 10, dtype=torch.float64),
        )
    def forward(self, x):
        out = self.encoder(x).flatten(start_dim = 1)
        return self.head(out)

def _test_Lenet(custom_module, torch_module, input_shape, outer_iters=100,
                     inner_iters=5, random_seed=None):
    module = LeNet()
    module.head
    if random_seed is not None:
        np.random.seed(random_seed)
    for i in range(outer_iters):
        module1 = custom_module()
        module2 = torch_module()

        debug_msg = f'Error in Lenet '
        
        for param1, param2 in zip(module1.encoder.parameters(), module2.encoder.parameters()):
            param2.data = torch.from_numpy(param1)
        for param1, param2 in zip(module1.head.parameters(), module2.head.parameters()):
            param2.data = torch.from_numpy(param1)
        
        for _ in range(inner_iters):
            x1 = np.random.randn(*input_shape)
            x2 = torch.from_numpy(x1)
            x2.requires_grad = True

            y1 = module1(x1)
            y2 = module2(x2)
            assert y1.dtype == x1.dtype
            assert_almost_equal(y1, y2.detach().numpy(), debug_msg + 'forward pass: {}')

            grad_output = np.random.randn(*y1.shape)
            y2.backward(torch.from_numpy(grad_output))
            grad_input = module1.backward(x1, grad_output)
            assert_almost_equal(x2.grad.numpy(), grad_input, debug_msg + 'input grad: {}')

            for grad, param in zip(module1.encoder.parameters_grad(), module2.encoder.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad encoder: {}')
            for grad, param in zip(module1.head.parameters_grad(), module2.head.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad head: {}')
        print("completed num_test ", i)


def test_Lenet():
    print('test_Lenet ... ', end='')
    for input_shape in input_shapes:
        _test_Lenet(
            Lenet, LeNet, input_shape,
            outer_iters=num_tests, random_seed=input_shape[0] + random_seed
        )
    print('OK')
