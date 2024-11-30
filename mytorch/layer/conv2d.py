from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer
import numpy as np
from numpy.lib.stride_tricks import as_strided



class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1),
                 need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, num_channels, input_height, input_width = input_tensor.shape
        filter_height, filter_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding

        output_height = (input_height + 2 * padding_height - filter_height) // stride_height + 1
        output_width = (input_width + 2 * padding_width - filter_width) // stride_width + 1

        padded_input = np.pad(input_tensor.data,
                              ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
                              mode='constant')

        sliding_window_shape = (batch_size, num_channels, output_height, output_width, filter_height, filter_width)
        sliding_window_strides = (
        padded_input.strides[0], padded_input.strides[1], stride_height * padded_input.strides[2],
        stride_width * padded_input.strides[3], padded_input.strides[2], padded_input.strides[3])

        sliding_windows = as_strided(padded_input, shape=sliding_window_shape, strides=sliding_window_strides)

        sliding_windows = sliding_windows.reshape(batch_size, num_channels, output_height * output_width, filter_height,
                                                  filter_width)
        convolution_output = np.tensordot(sliding_windows, self.weight.data, axes=((1, 3, 4), (1, 2, 3)))
        convolution_output = convolution_output.reshape(batch_size, self.out_channels, output_height, output_width)

        if self.need_bias:
            convolution_output += self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(convolution_output, requires_grad=input_tensor.requires_grad)

    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer((self.out_channels, self.in_channels, *self.kernel_size), self.initialize_mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.out_channels,), 'zero'),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
            self.kernel_size[0] * self.kernel_size[1],
            self.kernel_size,
            self.stride, self.padding)
