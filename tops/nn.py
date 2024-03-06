from .tensor import *
from .basicOps import *

class Linear(UnaryOp):
    def __init__(self, inNeuron: int, outNeuron: int, dtype:Type=Type.f64):
        self.weight = Tensor.rand(inNeuron, outNeuron, dtype=dtype)
        self.bias   = Tensor.rand(1, outNeuron, dtype=dtype)
    def forward(self, x) -> Tensor:
        return x @ self.weight + self.bias
class BatchNormalization(UnaryOp):
    def __init__(self):
        self.alpha = Tensor.rand(1, 1)
        self.beta  = Tensor.rand(1, 1)
    def forward(self, t: Tensor) -> Tensor:
        mean = Mean.forward(t)
        std  = StdDev.forward(t)
        new  = (t - mean)/std
        return self.alpha@new + self.beta
class NN:
    def __init__(self): pass
    def getParameters(self):
        v = vars(self)
        parameters = []
        for name in v:
            param = v[name]
            if type(param) == Linear:
                linear = v[name]
                parameters.append(param.weight)
                parameters.append(param.bias)
            elif type(param) == BatchNormalization:
                parameters.append(param.alpha)
                parameters.append(param.beta)
        return parameters
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
