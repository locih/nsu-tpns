import sys
from .test_base import test_module
import torch
from torch import nn
import numpy as np
from .test_base import assert_almost_equal

sys.path.append('..')
import modules as mm

input_shapes = [(64, 5, 16), (128, 5, 32), (256, 5, 64)]
num_tests = 50
random_seed = 1


def _test_LSTM(input_shape, bias,
                outer_iters=100, inner_iters=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    for _ in range(outer_iters):
        module1 = mm.LSTM(input_shape[2], input_shape[2] * 2, bias = bias)
        module2 = nn.LSTM(input_shape[2], input_shape[2] * 2, bias = bias, batch_first=True)

        for param1, param2 in zip(module1.parameters(), module2.parameters()):
                    param2.data = torch.from_numpy(param1)

        debug_msg = f'Error in LSTM '
        for _ in range(inner_iters):
            x1 = np.random.randn(*input_shape)
            x2 = torch.from_numpy(x1)
            h1 = np.zeros((1, input_shape[0], input_shape[2] * 2))
            h2 = torch.from_numpy(h1)
            c1 = np.zeros((1, input_shape[0], input_shape[2] * 2))
            c2 = torch.from_numpy(c1)
            h2.requires_grad = True
            c2.requires_grad = True

            next_hidden_cell_1, next_cell_state_1 = module1(x1, h1, c1)
            _, output = module2(x2, (h2, c2))
            next_hidden_cell_2, next_cell_state_2 = output
            assert next_hidden_cell_1.dtype == x1.dtype
            assert next_cell_state_1.dtype == next_hidden_cell_1.dtype
            assert_almost_equal(next_hidden_cell_1, next_hidden_cell_2[0].detach().numpy(), debug_msg + 'forward pass: {}')
            assert_almost_equal(next_cell_state_1, next_cell_state_2[0].detach().numpy(), debug_msg + 'forward pass: {}')

            grad_hx = np.random.randn(*next_hidden_cell_2.shape)
            grad_hidden_cell_state, grad_cell_state = module1.backward(x1, h1, c1, grad_hx)
            next_hidden_cell_2.retain_grad()
            next_hidden_cell_2.backward(torch.from_numpy(grad_hx), retain_graph=True)
            assert_almost_equal(c2.grad.numpy()[0], grad_cell_state, debug_msg + 'input grad: {}')
            assert_almost_equal(h2.grad.numpy()[0], grad_hidden_cell_state, debug_msg + 'input grad: {}')
            for grad, param in zip(module1.parameters_grad(), module2.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad: {}')

def test_LSTM():
    print('test_LSTM ... ', end='')
    for input_shape in input_shapes:
        for bias in (True, False):
            _test_LSTM(
                input_shape, bias,
                outer_iters=num_tests, random_seed=input_shape[0] + random_seed
            )

    print('OK')
