from .tensor import *

class Unary:
    def __init__(self): pass
    def forward(self, t: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, t: Tensor) -> Tensor:
        raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Unary: {self.__class__.__name__}>"
    def printGraph(self, level): print(self)

class Sigmoid(Unary):
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        out = Tensor(1/(1 + np.exp(-t.arr)))
        out.origin = Sigmoid()
        return out
    def backward(*args, **kargs) -> Tensor:
        self = args[0]
        t = args[1]
        x = 1/(1 + np.exp(-t.arr))
        return Tensor(x * (1-x))
