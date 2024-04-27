import sys
from .test_base import test_module
import torch
from torch import nn
import numpy as np
from .test_base import assert_almost_equal

sys.path.append('..')
import modules as mm

input_shapes = [(64, 16), (128, 32), (256, 64)]
num_tests = 50
random_seed = 1


def _test_LSTMCell(custom_module, torch_module, input_shape, module_kwargs=None,
                all_attrs=(), param_attrs=(),
                outer_iters=100, inner_iters=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if module_kwargs is None:
        module_kwargs = {}
    for _ in range(outer_iters):
        module1 = custom_module(**module_kwargs)
        module2 = torch_module(**module_kwargs)

        debug_msg = f'Error in LSTMCell '
        
        for attr in all_attrs:
            param1 = getattr(module1, attr)
            param2 = getattr(module2, attr)
            param1 += 0.1 * np.random.randn(*param1.shape)
            setattr(module1, attr, param1)
            param2.data = torch.from_numpy(param1)
        
        for _ in range(inner_iters):
            x1 = np.random.randn(*input_shape)
            x2 = torch.from_numpy(x1)
            h1 = np.random.randn(input_shape[0], input_shape[1] * 2)
            h2 = torch.from_numpy(h1)
            c1 = np.random.randn(input_shape[0], input_shape[1] * 2)
            c2 = torch.from_numpy(c1)
            h2.requires_grad = True
            c2.requires_grad = True
            next_hidden_cell_1, next_cell_state_1 = module1(x1, h1, c1)
            next_hidden_cell_2, next_cell_state_2 = module2(x2, (h2, c2))
            assert next_hidden_cell_1.dtype == x1.dtype
            assert next_cell_state_1.dtype == next_hidden_cell_1.dtype
            assert_almost_equal(next_hidden_cell_1, next_hidden_cell_2.detach().numpy(), debug_msg + 'forward pass: {}')
            assert_almost_equal(next_cell_state_1, next_cell_state_2.detach().numpy(), debug_msg + 'forward pass: {}')

            grad_hx = np.random.randn(*next_hidden_cell_1.shape)
            grad_cx = np.random.randn(*next_hidden_cell_1.shape)
            grad_hidden_cell_state, grad_cell_state = module1.backward(x1, h1, c1, grad_hx, grad_cx)
            next_cell_state_2.retain_grad()
            next_hidden_cell_2.retain_grad()
            next_hidden_cell_2.backward(torch.from_numpy(grad_hx), retain_graph=True)
            next_cell_state_2.backward(torch.from_numpy(grad_cx), retain_graph=True)

            assert_almost_equal(c2.grad.numpy(), grad_cell_state, debug_msg + 'input grad: {}')
            assert_almost_equal(h2.grad.numpy(), grad_hidden_cell_state, debug_msg + 'input grad: {}')
            for attr in param_attrs:
                assert_almost_equal(
                    getattr(module1, 'grad_' + attr),
                    getattr(module2, attr).grad.numpy(),
                    debug_msg + 'params grad: {}'
                )


def test_LSTMcell():
    print('test_LSTMCell ... ', end='')
    for input_shape in input_shapes:
        for bias in (True, False):
            attrs = ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh') if bias else ('weight_ih', 'weight_hh',)
            module_kwargs = {
                'input_size': input_shape[1],
                'hidden_size': 2 * input_shape[1],
                'bias': bias
            }
            _test_LSTMCell(
                mm.LSTMCell, nn.LSTMCell, input_shape,
                module_kwargs=module_kwargs, all_attrs=attrs,
                param_attrs=attrs,
                outer_iters=num_tests, random_seed=input_shape[0] + random_seed
            )

    print('OK')
