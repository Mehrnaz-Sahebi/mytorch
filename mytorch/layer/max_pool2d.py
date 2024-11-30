from mytorch import Tensor
from mytorch.layer import Layer
from numpy.lib.stride_tricks import as_strided

import numpy as np


class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        out_h = (height + 2 * p_h - k_h) // s_h + 1
        out_w = (width + 2 * p_w - k_w) // s_w + 1

        x_pad = np.pad(x.data, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')

        win_shape = (
            batch, channels, out_h, out_w, k_h, k_w
        )
        win_strides = (
            x_pad.strides[0], x_pad.strides[1], s_h * x_pad.strides[2],
            s_w * x_pad.strides[3], x_pad.strides[2], x_pad.strides[3]
        )

        windows = as_strided(x_pad, shape=win_shape, strides=win_strides)
        windows = windows.reshape(batch, channels, out_h, out_w, -1)

        out = np.max(windows, axis=-1)

        return Tensor(out, requires_grad=x.requires_grad)

    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
