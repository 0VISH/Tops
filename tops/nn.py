from .tensor import *

class NN:
    def __init__(self): pass
    def getParameters(self):
        v = vars(self)
        parameters = []
        for name in v:
            if type(v[name]) == Tensor: parameters.append(v[name])
        return parameters
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
