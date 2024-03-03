from .tensor import *
from .unary import *

class Linear(UnaryOp):
    def __init__(self, inNeuron: int, outNeuron: int, dtype:Type=Type.f64):
        self.weight = Tensor.rand(inNeuron, outNeuron, dtype=dtype)
        self.bias   = Tensor.rand(1, outNeuron, dtype=dtype)
    def forward(self, x) -> Tensor:
        return x @ self.weight + self.bias
        
class NN:
    def __init__(self): pass
    def getParameters(self):
        v = vars(self)
        parameters = []
        for name in v:
            if type(v[name]) == Linear:
                linear = v[name]
                parameters.append(linear.weight)
                parameters.append(linear.bias)
        return parameters
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
