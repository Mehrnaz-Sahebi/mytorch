import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    # exp_x = x.exp()
    # exp_negative_x = x.exp(x.__neg__)
    # numerator = exp_x.__sub__(exp_negative_x)
    # denominator = exp_x.__add__(exp_negative_x)
    # return numerator.__mul__(denominator.__pow__(-1))
    return (x.exp() - (-x).exp) * ((x.exp() + (-x).exp) ** -1)