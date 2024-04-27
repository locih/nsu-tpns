import sys
from .test_base import test_module
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(32, 6, 28, 28), (32, 16, 10, 10)]
num_tests = 50
random_seed = 1


def test_avgPool2d():
    print('test_avgPool2d ... ', end='')
    for input_shape in input_shapes:
        module_kwargs = {
            'kernel_size' : 2
        }
        test_module(
            mm.AvgPool2d, nn.AvgPool2d, input_shape,
            module_kwargs=module_kwargs, outer_iters=num_tests,
            random_seed=input_shape[0] + random_seed

        )
    print('OK')
