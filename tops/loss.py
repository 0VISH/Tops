from .tensor import *

class MSE(BinaryOp):
    @staticmethod
    def forward(truth: Tensor, pred: Tensor) -> Tensor:
        return Mean(Pow(truth - pred, 2))