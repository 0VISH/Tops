from .unary import *
from .tensor import *
import numpy as np

class MSE:
    def __init__(self): pass
    def forward(self, truth: Tensor, pred: Tensor) -> Tensor:
        return Mean.forward(Pow.forward(truth - pred, 2))     
class CrossEntropy:
    def forward(self, truth: Tensor, predicted: Tensor) -> Tensor:
        return Tensor([-1]) * truth * Log.forward(predicted)
