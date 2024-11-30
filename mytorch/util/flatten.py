import numpy as np
from mytorch import Tensor

def flatten(x: Tensor, axis) -> Tensor:
    """
    TODO: implement flatten.
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """

    shape_before = x.shape[:axis]
    shape_after = (-1,)
    new_shape = shape_before + shape_after
    return Tensor(data=np.reshape(x.data, new_shape), requires_grad=x.requires_grad, depends_on=x.depends_on)
