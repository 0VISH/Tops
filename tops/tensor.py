import numpy as np

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
    
    @staticmethod
    def getName(t: type):
        if t == Type.f64: return "f64"
        if t == Type.f32: return "f32"
        if t == Type.s64: return "s64"
        if t == Type.s32: return "s32"
        if t == Type.s16: return "s16"
        if t == Type.u64: return "u64"
        if t == Type.u32: return "u32"
        if t == Type.u16: return "u16"
        if t == Type.s8:  return "s8"
        if t == Type.u8:  return "u8"

class BinaryOp:
    def __init__(self): pass
    def forward(*args, **kargs):  raise NotImplementedError(f"forward not implemented for {args[0].__class__.__name__}")
    def backward(*args, **kargs): raise NotImplementedError(f"backward not implemented for {args[0].__class__.__name__}")
    def __repr__(self): raise NotImplementedError(f"__repr__ not implemented for {self.__class__.__name__}")
    def printGraph(self, level):
        print(self)
        self.lhs.printGraph(level+1)
        self.rhs.printGraph(level+1)

class Add(BinaryOp):
    def forward(*args, **kargs):
        self = args[0]
        self.lhs = args[1]
        self.rhs = args[2]
        t = Tensor(self.lhs.arr + self.rhs.arr, dtype=args[1].type)
        t.origin = self
        return t
    def backward(*args, **kargs):
        self = args[0]
        out  = args[1]
        self.lhs.grad = out.grad
        self.rhs.grad = out.grad
        self.lhs._backward()
        self.rhs._backward()
    def __repr__(self): return "<Function: add>"
class Sub(BinaryOp):
    def forward(*args, **kargs):
        self = args[0]
        self.lhs = args[1]
        self.rhs = args[2]
        t = Tensor(self.lhs.arr - self.rhs.arr, dtype=args[1].type)
        t.origin = self
        return t
    def backward(*args, **kargs):
        self = args[0]
        out  = args[1]
        self.lhs.grad = out.grad
        self.rhs.grad = -1 * out.grad
        self.lhs._backward()
        self.rhs._backward()
    def __repr__(self): return "<Function: sub>"
class Hadamard(BinaryOp):
    def forward(*args, **kargs):
        self = args[0]
        self.lhs = args[1]
        self.rhs = args[2]
        t = Tensor(np.multiply(self.lhs.arr, self.rhs.arr), dtype=args[1].type)
        t.origin = self
        return t
    def backward(*args, **kargs):
        self = args[0]
        out  = args[1]
        self.lhs.grad = self.rhs.arr * out.grad
        self.rhs.grad = self.lhs.arr * out.grad
        self.lhs._backward()
        self.rhs._backward()
    def __repr__(self): return "<Function: hadamard>"
class Dot(BinaryOp):
    def forward(*args, **kargs):
        self = args[0]
        self.lhs = args[1]
        self.rhs = args[2]
        res = np.dot(self.lhs.arr, self.rhs.arr)
        if np.shape(res) == (): res = np.array([res])
        t = Tensor(res, dtype=args[1].type)
        t.origin = self
        return t
    def backward(*args, **kargs):
        self = args[0]
        out  = args[1]
        self.lhs.grad = np.dot(self.rhs.arr, out.grad)
        self.rhs.grad = np.dot(self.lhs.arr.T, out.grad)
        self.lhs._backward()
        self.rhs._backward()
    def __repr__(self): return "<Function: dot>"

class Tensor:
    def __init__(self, arr: list, shape=None, dtype:Type=Type.f64):
        self.type = dtype
        self.arr  = np.array(arr, dtype=self.type)
        if shape != None: self.arr = np.reshape(self.arr, shape)
        self.grad = np.full(np.shape(self.arr), 0, dtype=Type.f64)
        self.origin = None
    def shape(self): return np.shape(self.arr)
    def __repr__(self): return f"<tensor: {self.shape()}, {Type.getName(self.type)}>"
    @staticmethod
    def rand(x: int, y: int, dtype:Type=Type.f64):
        return Tensor(np.random.randn(x, y).astype(dtype), dtype=dtype)
    def fill(self, shape, val, dtype:Type=Type.f64):
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
