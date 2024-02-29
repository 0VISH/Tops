from .tensor import *

class Unary:
    def __init__(self, input: Tensor): self.input = input
    def forward(self, t: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, t: Tensor) -> Tensor:
        raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Unary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.input.printGraph(level+1)

class Log(Unary):
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        out = Tensor(np.log(t.arr))
        out.origin = Log(t)
        return out
    def backward(*args, **kargs) -> Tensor:
        self = args[0]
        self.input.grad = 1/self.input.arr
class Mean(Unary):
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        out = Tensor(t.arr.mean())
        out.origin = Mean(t)
        return out
    def backward(*args, **kargs) -> Tensor:
        self = args[0]
        self.input.grad = (1/self.input.shape()[0]) * (1/(self.input.arr.sum()**2))
class Pow(Unary):
    def __init__(self, input: Tensor, x: int):
        super().__init__(input)
        self.pow = x
    @staticmethod
    def forward(t: Tensor, x: int) -> Tensor:
        out = Tensor(t.arr ** x)
        out.origin = Pow(t, x)
        return out
    def backward(*args, **kargs) -> Tensor:
        self = args[0]
        self.input.grad = self.pow * (self.input.arr ** (self.pow-1))
class Sigmoid(Unary):
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        out = Tensor(1/(1 + np.exp(-t.arr)))
        out.origin = Sigmoid(t)
        return out
    def backward(*args, **kargs) -> Tensor:
        self = args[0]
        x = 1/(1 + np.exp(-self.input.arr))
        self.input.grad = x * (1-x)
