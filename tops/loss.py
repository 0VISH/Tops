from .unary import *
from .tensor import *

class MSE:
    def __init__(self): pass
    def forward(self, truth: Tensor, pred: Tensor) -> Tensor:
        return Mean.forward(Pow.forward(truth - pred, 2))
