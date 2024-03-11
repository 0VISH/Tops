import numpy as np
from .ops import *
from .driver import *

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

class Echo(BroadcastOp):
    def __init__(self, outputShape): self.outputShape = outputShape
    def forward(self, t):
        self.input = t
        out = Tensor.fill(self.outputShape, t.arr, dtype=t.type())
        out.origin = self
        return out
    def backward(self, grad):
        newGrad = np.array([np.sum(grad)])
        self.input.grad += newGrad
        self.input._backward(newGrad)
class Flatten(BroadcastOp):
    def forward(self, t):
        self.input = t
        return Tensor(t.arr.flatten(), origin=self)
    def backward(self, grad):
        newGrad = np.reshape(grad, self.input.shape())
        self.input.grad += newGrad
        self.input._backward(newGrad)

class Tensor:
    def __init__(self, arr, shape=None, dtype:Type=Type.f64, driver=CPUDriver(), origin=None):
        if(type(arr) == np.ndarray): self.arr = arr
        else: self.arr = np.array(arr, dtype=dtype)
        if shape != None: self.arr = np.reshape(self.arr, shape)
        self.grad = np.zeros_like(self.arr, dtype=np.float64)
        self.origin = origin
        self.driver = driver
    def zeroGrad(self): self.grad = np.zeros_like(self.arr, dtype=np.float64)
    def type(self):  return self.arr.dtype
    def shape(self): return np.shape(self.arr)
    def count(self): return np.size(self.arr)
    def sum(self):   return np.sum(self.arr)
    def __repr__(self): return f"<Tensor: {self.shape()}, {str(self.arr.dtype)}>"
    @staticmethod
    def rand(shape, dtype:Type=Type.f64):
        return Tensor(np.random.randn(*shape).astype(dtype), dtype=dtype)
    @staticmethod
    def fill(shape, val, dtype:Type=Type.f64):
        return Tensor(np.full(shape, val, dtype=dtype), dtype=dtype)
    def flatten(self):
        f = Flatten()
        return f.forward(self)
    def __add__(self, other):
        lhs = self
        rhs = other
        if lhs.shape() == (1,1) or lhs.shape() == (1,):
            b = Echo(rhs.shape())
            lhs = b.forward(lhs)
        if rhs.shape() == (1,1) or rhs.shape() == (1,):
            b = Echo(lhs.shape())
            rhs = b.forward(rhs)
        f = self.driver.Add()
        return f.forward(lhs, rhs)
    def __sub__(self, other):
        lhs = self
        rhs = other
        if lhs.shape() == (1,1) or lhs.shape() == (1,):
            b = Echo(rhs.shape())
            lhs = b.forward(lhs)
        if rhs.shape() == (1,1) or rhs.shape() == (1,):
            b = Echo(lhs.shape())
            rhs = b.forward(rhs)
        f = self.driver.Sub()
        return f.forward(lhs, rhs)
    def __mul__(self, other):
        lhs = self
        rhs = other
        if lhs.shape() == (1,1) or lhs.shape() == (1,):
            b = Echo(rhs.shape())
            lhs = b.forward(lhs)
        if rhs.shape() == (1,1) or rhs.shape() == (1,):
            b = Echo(lhs.shape())
            rhs = b.forward(rhs)
        f = self.driver.Mul()
        return f.forward(lhs, rhs)
    def __truediv__(self, other):
        lhs = self
        rhs = other
        if lhs.shape() == (1,1) or lhs.shape() == (1,):
            b = Echo(rhs.shape())
            lhs = b.forward(lhs)
        if rhs.shape() == (1,1) or rhs.shape() == (1,):
            b = Echo(lhs.shape())
            rhs = b.forward(rhs)
        f = self.driver.Div()
        return f.forward(lhs, rhs)
    def __matmul__(self, other):
        f = self.driver.Dot()
        return f.forward(self, other)
    def _backward(self, grad):
        origin = self.origin
        if origin != None: origin.backward(grad)
    def backward(self):
        self.grad = np.full(self.shape(), 1)
        self._backward(self.grad)
    def printGraph(self, level=0):
        origin = self.origin
        print(level*"    ", end="")
        if origin != None: origin.printGraph(level)
        else: print(self)
