import numpy as np

DELTA = 1e-6

class Type:
    u8  = np.uint8
    u16 = np.uint16
    u32 = np.uint32
    u64 = np.uint64
    s8  = np.int8
    s16 = np.int16
    s32 = np.int32
    s64 = np.int64
    f32 = np.float32
    f64 = np.float64

class BinaryOp:
    def __init__(self): pass
    def forward(self, lhs, rhs):  raise NotImplementedError(f"forward not implemented for {args[0].__class__.__name__}")
    def backward(self, out): raise NotImplementedError(f"backward not implemented for {args[0].__class__.__name__}")
    def __repr__(self): return f"<Binary: {self.__class__.__name__}>"
    def printGraph(self, level):
        print(self)
        self.lhs.printGraph(level+1)
        self.rhs.printGraph(level+1)

class Add(BinaryOp):
    def forward(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        t = Tensor(lhs.arr + rhs.arr)
        t.origin = self
        return t
    def backward(self, out):
        self.lhs.grad = out.grad
        self.rhs.grad = out.grad
        self.lhs._backward()
        self.rhs._backward()
class Sub(BinaryOp):
    def forward(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        t = Tensor(lhs.arr - rhs.arr)
        t.origin = self
        return t
    def backward(self, out):
        self.lhs.grad = out.grad
        self.rhs.grad = -out.grad
        self.lhs._backward()
        self.rhs._backward()
class Hadamard(BinaryOp):
    def forward(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        t = Tensor(np.multiply(lhs.arr, rhs.arr))
        t.origin = self
        return t
    def backward(self, out):
        self.lhs.grad = self.rhs.arr * out.grad
        self.rhs.grad = self.lhs.arr * out.grad
        self.lhs._backward()
        self.rhs._backward()
class Dot(BinaryOp):
    def forward(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        res = np.dot(lhs.arr, rhs.arr)
        if np.shape(res) == (): res = np.array([res])
        t = Tensor(res)
        t.origin = self
        return t
    def backward(self, out):
        self.lhs.grad = np.dot(out.grad, self.rhs.arr.T)
        self.rhs.grad = np.dot(self.lhs.arr.T, out.grad)
        self.lhs._backward()
        self.rhs._backward()
class Div(BinaryOp):
    def forward(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        res = lhs.arr / (rhs.arr + DELTA)
        #if np.shape(res) == (): res = np.array([res])
        t = Tensor(res)
        t.origin = self
        return t
    def backward(self, out):
        self.lhs.grad = (1/(self.rhs.arr+DELTA)) * out.grad
        self.rhs.grad = self.lhs.arr / ((self.rhs.arr ** 2)+DELTA)
        self.lhs._backward()
        self.rhs._backward()

class Tensor:
    def __init__(self, arr, shape=None, dtype:Type=Type.f64):
        if(type(arr) == np.ndarray): self.arr = arr
        else: self.arr = np.array(arr, dtype=dtype)
        if shape != None: self.arr = np.reshape(self.arr, shape)
        self.grad = np.full(np.shape(self.arr), 0, dtype=dtype)
        self.origin = None
    def shape(self): return np.shape(self.arr)
    def count(self): return np.size(self.arr)
    def sum(self):   return np.sum(self.arr)
    def __repr__(self): return f"<Tensor: {self.shape()}, {str(self.arr.dtype)}>"
    @staticmethod
    def rand(x: int, y: int, dtype:Type=Type.f64):
        return Tensor(np.random.randn(x, y).astype(dtype), dtype=dtype)
    @staticmethod
    def fill(shape, val, dtype:Type=Type.f64):
        return Tensor(np.full(shape, val, dtype=dtype), dtype=dtype)
    def __add__(self, other):
        f = Add()
        return f.forward(self, other)
    def __sub__(self, other):
        f = Sub()
        return f.forward(self, other)
    def __matmul__(self, other):
        f = Dot()
        return f.forward(self, other)
    def __mul__(self, other):
        f = Hadamard()
        return f.forward(self, other)
    def __truediv__(self, other):
        f = Div()
        return f.forward(self, other)
    def _backward(self):
        origin = self.origin
        if origin != None: origin.backward(self)
    def backward(self):
        self.grad = np.full(self.shape(), 1)
        self._backward()
    def printGraph(self, level=0):
        origin = self.origin
        print(level*"    ", end="")
        if origin != None: origin.printGraph(level)
        else: print(self)
