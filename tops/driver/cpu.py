from tops import tensor
from tops import ops
import numpy as np

class CPUDriver:
    class Add(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr + rhs.arr, origin=self)
        def backward(self, grad):
            self.lhs.grad += grad
            self.rhs.grad += grad
            self.lhs._backward(grad)
            self.rhs._backward(grad)
    class Sub(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr - rhs.arr, origin=self)
        def backward(self, grad):
            lhsGrad = grad
            rhsGrad = -grad
            self.lhs.grad += lhsGrad
            self.rhs.grad += rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Mul(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(np.multiply(lhs.arr, rhs.arr), origin=self)
        def backward(self, grad):
            lhsGrad = self.rhs.arr * grad
            rhsGrad = self.lhs.arr * grad
            self.lhs.grad += lhsGrad
            self.rhs.grad += rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Div(ops.BinaryOp):
        def forward(self, lhs, rhs):
            super().reg(lhs, rhs)
            return tensor.Tensor(lhs.arr / (rhs.arr + tensor.DELTA), origin=self)
        def backward(self, grad):
            lhsGrad = (1/(self.rhs.arr+tensor.DELTA)) * grad
            rhsGrad = self.lhs.arr / ((self.rhs.arr ** 2)+tensor.DELTA) * grad
            self.lhs.grad += lhsGrad
            self.rhs.grad += rhsGrad
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
            self.lhs.grad += lhsGrad
            self.rhs.grad += rhsGrad
            self.lhs._backward(lhsGrad)
            self.rhs._backward(rhsGrad)
    class Mean(ops.BroadcastOp):
        def __init__(self, dim, keepdim):
            self.dim = dim
            self.keepdim=keepdim
        def forward(self, input):
            self.input = input
            return tensor.Tensor(input.arr.mean(axis=self.dim, keepdims=self.keepdim), dtype=input.type(), origin=self)
        def backward(self, grad):
            if self.keepdim and self.dim == 0: grad = grad[0]
            elif self.keepdim==False and self.dim == 1: grad = np.array([grad]).T
            split = np.split(grad, grad.shape[self.dim], axis=self.dim)
            d = []
            for i in range(0, len(split)):
                d.append(np.full(split[i].shape, grad[i]/self.input.shape()[self.dim]))
            newGrad = np.concatenate(d, axis=self.dim)
            self.input.grad += newGrad
            self.input._backward(newGrad)
    class StdDev(ops.BroadcastOp):
        def __init__(self, dim, keepdim):
            self.dim = dim
            self.keepdim=keepdim
        def forward(self, input):
            self.input = input
            out = tensor.Tensor(input.arr.std(axis=self.dim, keepdims=self.keepdim), dtype=input.type(), origin=self)
            self.outArr = out.arr
            return out
        def backward(self, grad):
            if self.keepdim and self.dim == 0: grad = grad[0]
            elif self.keepdim==False and self.dim == 1:
                grad = np.array([grad]).T
                self.outArr = np.array([self.outArr]).T
            split = np.split(grad, grad.shape[self.dim], axis=self.dim)
            splitInput = np.split(self.input.numpy(), self.input.shape()[self.dim], axis=self.dim)
            splitOutput = np.split(self.outArr, self.outArr.shape[self.dim], axis=self.dim)
            d = []
            for i in range(0, len(splitInput)):
                mean = splitInput[i].mean()
                stdVar = splitInput[i].std()
                d.append((splitInput[i] - mean)/(splitInput[i].size * stdVar))
            newGrad = np.concatenate(d, axis=self.dim) 
            self.input.grad += newGrad
            self.input._backward(newGrad)
    class Pow(ops.UnaryOp):
        def forward(self, input, x: int):
            self.input = input
            self.pow = x
            return tensor.Tensor(input.arr ** x, origin=self)
        def backward(self, grad):
            newGrad = self.pow * (self.input.arr ** (self.pow-1)) * grad
            self.input.grad += newGrad
            self.input._backward(newGrad)
    class Sigmoid(ops.UnaryOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor(1/(1 + np.exp(-input.arr)), origin=self)
        def backward(self, grad):
            x = 1/(1 + np.exp(-self.input.arr))
            newGrad = x * (1-x) * grad
            self.input.grad += newGrad
            self.input._backward(newGrad)
    class ReLu(ops.UnaryOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor(np.maximum(0, input.grad), origin=self)
        def backward(self, grad):
            newGrad = np.where(self.input.arr <= 0, 0, 1) * grad
            self.input.grad += newGrad
            self.input._backward(newGrad)
    class Log(ops.UnaryOp):
        def forward(self, input):
            self.input = input
            return tensor.Tensor(np.log(input.arr+tensor.DELTA), origin=self)
        def backward(self, grad):
            newGrad = (1/(self.input.arr+tensor.DELTA)) * grad
            self.input.grad += newGrad
            self.input._backward(newGrad)
    @staticmethod
    def Conv2DForward(input, kernel, stride):
        inputX, inputY = input.shape()
        kernelX, kernelY = kernel.shape()

        outputY = (inputY - kernelY) // stride + 1
        outputX = (inputX - kernelX) // stride + 1
        output = np.zeros((outputY, outputX))
        
        for y in range(0, inputY - kernelY + 1, stride):
            for x in range(0, inputX - kernelX + 1, stride):
                region = input.arr[y:y+kernelY, x:x+kernelX]
                output[y // stride, x // stride] = np.sum(region * kernel.arr)
        return output
    @staticmethod
    def Conv2DBackward(input, kernel, grad, stride):
        inputY, inputX = np.shape(input)
        kernelY, kernelX = np.shape(kernel)
        outputY, outputX = np.shape(grad)

        gradInput = np.zeros_like(input)
        gradKernel = np.zeros_like(kernel)
    
        for i in range(0, inputY - kernelY + 1, stride):
            for j in range(0, inputX - kernelX + 1, stride):
                gradInput[i:i+kernelY, j:j+kernelX] += kernel  * grad[i // stride, j // stride]
                gradKernel += input[i:i+kernelY, j:j+kernelX]  * grad[i // stride, j // stride]
            
        return gradInput, gradKernel
