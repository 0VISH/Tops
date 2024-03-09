from tops import tensor
from tops import ops
import numpy as np

class CPUDriver:
    class Add(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr + rhs.arr, origin=self)
        def backward(self, grad):
            self.lhs.grad = grad
            self.rhs.grad = grad
            self.lhs._backward(grad)
            self.rhs._backward(grad)
    class Sub(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr - rhs.arr, origin=self)
        def backward(self, grad):
            lhsGrad = grad
            rhsGrad = -grad
            self.lhs.grad = lhsGrad
            self.rhs.grad = rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Mul(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(np.multiply(lhs.arr, rhs.arr), origin=self)
        def backward(self, grad):
            lhsGrad = self.rhs.arr * grad
            rhsGrad = self.lhs.arr * grad
            self.lhs.grad = lhsGrad
            self.rhs.grad = rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Div(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr / (rhs.arr + tensor.DELTA), origin=self)
        def backward(self, grad):
            lhsGrad = (1/(self.rhs.arr+tensor.DELTA)) * grad
            rhsGrad = self.lhs.arr / ((self.rhs.arr ** 2)+tensor.DELTA) * grad
            self.lhs.grad = lhsGrad
            self.rhs.grad = rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Dot(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            res = np.dot(lhs.arr, rhs.arr)
            if np.shape(res) == (): res = np.array([res])
            return tensor.Tensor(res, origin=self)
        def backward(self, grad):
            lhsGrad = np.dot(grad, self.rhs.arr.T)
            rhsGrad = np.dot(self.lhs.arr.T, grad)
            self.lhs.grad = lhsGrad
            self.rhs.grad = rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Mean(ops.BroadcastOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor([input.arr.mean()], dtype=input.type(), origin=self)
        def backward(self, grad):
            newGrad = np.full(self.input.shape(), grad/self.input.count())
            self.input.grad = newGrad
            self.input._backward(newGrad)
    class StdDev(ops.BroadcastOp):
        def forward(self, input):
            self.input = input
            out = tensor.Tensor([input.arr.std()], dtype=input.type(), origin=self)
            self.outArr = out.arr
            return out
        def backward(self, grad):
            mean = self.input.arr.mean()
            newGrad  = (1/(self.outArr+tensor.DELTA)) * ((self.input.arr - mean) / np.shape(self.input.arr)[1]) * grad
            self.input.grad = newGrad
            self.input._backward(newGrad)
    class Pow(ops.UnaryOp):
        def forward(self, input, x: int):
            self.input = input
            self.pow = x
            return tensor.Tensor(input.arr ** x, origin=self)
        def backward(self, grad):
            newGrad = self.pow * (self.input.arr ** (self.pow-1)) * grad
            self.input.grad = newGrad
            self.input._backward(newGrad)
    class Sigmoid(ops.UnaryOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor(1/(1 + np.exp(-input.arr)), origin=self)
        def backward(self, grad):
            x = 1/(1 + np.exp(-self.input.arr))
            newGrad = x * (1-x) * grad
            self.input.grad = newGrad
            self.input._backward(newGrad)
