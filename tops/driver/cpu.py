from tops import tensor
from tops import ops
import numpy as np

class CPUDriver:
    class Add(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr + rhs.arr, origin=self)
        def backward(self, out):
            self.lhs.grad = out.grad
            self.rhs.grad = out.grad
            self.lhs._backward()
            self.rhs._backward()
    class Sub(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr - rhs.arr, origin=self)
        def backward(self, out):
            self.lhs.grad = out.grad
            self.rhs.grad = -out.grad
            self.lhs._backward()
            self.rhs._backward()
    class Mul(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(np.multiply(lhs.arr, rhs.arr), origin=self)
        def backward(self, out):
            self.lhs.grad = self.rhs.arr * out.grad
            self.rhs.grad = self.lhs.arr * out.grad
            self.lhs._backward()
            self.rhs._backward()
    class Div(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr / (rhs.arr + tensor.DELTA), origin=self)
        def backward(self, out):
            self.lhs.grad = (1/(self.rhs.arr+tensor.DELTA)) * out.grad
            self.rhs.grad = self.lhs.arr / ((self.rhs.arr ** 2)+tensor.DELTA)
            self.lhs._backward()
            self.rhs._backward()
    class Dot(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            res = np.dot(lhs.arr, rhs.arr)
            if np.shape(res) == (): res = np.array([res])
            return tensor.Tensor(res, origin=self)
        def backward(self, out):
            self.lhs.grad = np.dot(out.grad, self.rhs.arr.T)
            self.rhs.grad = np.dot(self.lhs.arr.T, out.grad)
            self.lhs._backward()
            self.rhs._backward()
    class Mean(ops.BroadcastOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor([input.arr.mean()], dtype=input.type(), origin=self)
        def backward(self, out):
            self.input.grad = np.full(self.input.shape(), out.grad/self.input.count())
            self.input._backward()
    class StdDev(ops.BroadcastOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor([input.arr.std()], dtype=input.type(), origin=self)
        def backward(self, out):
            mean = self.input.arr.mean()
            res  = (1/(out.arr+tensor.DELTA)) * ((self.input.arr - mean) / np.shape(self.input.arr)[1]) * out.grad
            self.input.grad = res
    class Pow(ops.UnaryOp):
        def forward(self, input, x: int):
            self.input = input
            self.pow = x
            return tensor.Tensor(input.arr ** x, origin=self)
        def backward(self, out):
            self.input.grad = self.pow * (self.input.arr ** (self.pow-1)) * out.grad
            self.input._backward()
    class Sigmoid(ops.UnaryOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor(1/(1 + np.exp(-input.arr)), origin=self)
        def backward(self, out):
            x = 1/(1 + np.exp(-self.input.arr))
            self.input.grad = x * (1-x) * out.grad
            self.input._backward()
