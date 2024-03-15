class BinaryOp:
    def reg(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def forward(self, lhs, rhs):  raise NotImplementedError(f"forward not implemented for {args[0].__class__.__name__}")
    def backward(self, out): raise NotImplementedError(f"backward not implemented for {args[0].__class__.__name__}")
    def __repr__(self): return f"<Binary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.lhs.printGraph(level+1)
        self.rhs.printGraph(level+1)

class UnaryOp:
    def forward(self, input):
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, out):
        raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Unary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.input.printGraph(level+1)

class BroadcastOp:
    def forward(self, input): raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def backward(self, out): raise NotImplementedError(f"backward not implemented for {self.__class__.__name__}")
    def __repr__(self): return f"<Broadcast: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.input.printGraph(level+1)

def Mean(input):
    f = input.driver.Mean()
    return f.forward(input)
def StdDev(input):
    f = input.driver.StdDev()
    return f.forward(input)
def Pow(input, x):
    f = input.driver.Pow()
    return f.forward(input, x)
def Sigmoid(input):
    f = input.driver.Sigmoid()
    return f.forward(input)
def ReLu(input):
    f = input.driver.ReLu()
    return f.forward(input)
def Log(input):
    f = input.driver.Log()
    return f.forward(input)
