from .tensor import *
from .basicOps import *

class MSE(BinaryOp):
    @staticmethod
    def forward(truth: Tensor, pred: Tensor) -> Tensor:
        return Mean.forward(Pow.forward(truth - pred, 2))
