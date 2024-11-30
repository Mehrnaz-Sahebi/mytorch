import numpy as np
from mytorch import Tensor, Dependency


def relu(x: Tensor) -> Tensor:
    "TODO: implement relu function"

    # use np.maximum
    data = np.maximum(x.data, 0)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # use np.where
            return np.where(x.data < 0, 0, grad)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
