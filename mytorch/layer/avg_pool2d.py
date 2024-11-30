from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np


class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, in_channels, in_height, in_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        pad_height, pad_width = self.padding

        out_height = (in_height + 2 * pad_height - kernel_height) // stride_height + 1
        out_width = (in_width + 2 * pad_width - kernel_width) // stride_width + 1

        out = np.zeros((batch_size, in_channels, out_height, out_width))

        x_padded = np.pad(x.data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        for n in range(batch_size):
            for c in range(in_channels):
                for h in range(0, in_height, stride_height):
                    for w in range(0, in_width, stride_width):
                        h_end = h + kernel_height
                        w_end = w + kernel_width
                        out[n, c, h // stride_height, w // stride_width] = np.mean(
                            x_padded[n, c, h:h_end, w:w_end]
                        )

        return Tensor(out, requires_grad=x.requires_grad)

    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
