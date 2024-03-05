class BinaryOp:
    def __init__(self): pass
    def forward(self, lhs, rhs):  raise NotImplementedError(f"forward not implemented for {args[0].__class__.__name__}")
    def backward(self, out): raise NotImplementedError(f"backward not implemented for {args[0].__class__.__name__}")
    def __repr__(self): return f"<Binary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.lhs.printGraph(level+1)
        self.rhs.printGraph(level+1)

class UnaryOp:
    def __init__(self, input): self.input = input
    def forward(self, input):
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, out):
        raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Unary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.input.printGraph(level+1)

class BroadcastOp:
    def __init__(self, input): self.input = input
    def forward(self, t): raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, out): raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Broadcast: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.input.printGraph(level+1)
