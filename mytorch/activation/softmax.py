import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    # shift_x = x.data - np.max(x.data, axis=1, keepdims=True)
    # tensor_x = Tensor(data=shift_x)
    # exp_x = tensor_x.exp()
    # numerator = exp_x
    # denominator = exp_x.__matmul__(Tensor(np.ones((x.data.shape[-1], 1))))
    # result = numerator.__mul__(denominator.__pow__(-1))
    # return result
    exp_x = x.exp()
    s = exp_x @ (np.ones((exp_x.shape[-1], 1)))
    return exp_x * (s ** -1)
