from .tensor import *

def MSE(truth: Tensor, pred: Tensor) -> Tensor:
    return Mean(Pow(truth - pred, 2))
