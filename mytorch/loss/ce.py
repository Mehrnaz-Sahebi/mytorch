from mytorch import Tensor
import numpy as np



def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    return - (label * preds.log()).sum()

