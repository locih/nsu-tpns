import sys
from .test_base import test_module
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(32, 1, 32, 32), (32, 6, 28, 28), (32, 16, 10, 10)]
output_channels = [6, 16, 120]
num_tests = 50
random_seed = 1


def test_conv2d():
    print('test_conv2d ... ', end='')
    for input_shape, output_channel in zip(input_shapes, output_channels):
        for bias in (True, False):
            attrs = ('weight', 'bias') if bias else ('weight', )
            module_kwargs = {
                'in_channels': input_shape[1],
                'out_channels': output_channel,
                'kernel_size': 5,
                'bias': bias
            }

            test_module(
                mm.Conv2d, nn.Conv2d, input_shape,
                module_kwargs=module_kwargs, all_attrs=attrs,
                param_attrs=attrs, eval_module=False,
                outer_iters=num_tests, random_seed=input_shape[0] + random_seed
            )

    print('OK')
