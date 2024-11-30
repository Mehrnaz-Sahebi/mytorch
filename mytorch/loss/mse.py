import numpy as np
from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    sub = preds.__sub__(actual)
    sub_pow_two = sub.__pow__(2)
    sum = sub_pow_two.sum()
    denominator = Tensor(np.array([sub_pow_two.data.size], dtype=np.float64))
    error = sum.__mul__(denominator.__pow__(-1))
    return error
