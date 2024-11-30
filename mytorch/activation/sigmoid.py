import numpy as np
from mytorch import Tensor, Dependency


def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    # exp_negative_x = x.__neg__().exp()
    # denominator = exp_negative_x.__add__(Tensor(np.ones_like(x.data)))
    # return denominator.__pow__(-1)
    return (1 + (-x).exp()) ** -1