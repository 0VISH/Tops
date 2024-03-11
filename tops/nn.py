from .tensor import *

#outputSize: (inputSize - filterSize + 2*padding)/stride + 1
class Conv2D(BinaryOp):
    def __init__(self, kernelSize, stride=1):
        self.kernel = Tensor.rand(kernelSize)
        self.stride = stride
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        arr = input.driver.Conv2DForward(input, self.kernel, self.stride)
        return Tensor(arr, origin = self)
    def backward(self, grad):
        newGrad, gradKernel = self.input.driver.Conv2DBackward(self.input.arr, self.kernel.arr, grad, self.stride)
        self.input.grad  += newGrad
        self.kernel.grad += gradKernel
        self.input._backward(newGrad)
        
class Linear(UnaryOp):
    def __init__(self, inNeuron: int, outNeuron: int, dtype:Type=Type.f64):
        self.weight = Tensor.rand((inNeuron, outNeuron), dtype=dtype)
        self.bias   = Tensor.rand((1, outNeuron), dtype=dtype)
    def forward(self, x) -> Tensor:
        return x @ self.weight + self.bias
class BatchNormalization(UnaryOp):
    def __init__(self):
        self.alpha = Tensor.rand((1, 1))
        self.beta  = Tensor.rand((1, 1))
    def forward(self, t: Tensor) -> Tensor:
        mean = Mean(t)
        std  = StdDev(t)
        new  = (t - mean)/std
        return self.alpha@new + self.beta
class NN:
    def __init__(self): pass
    def getParameters(self):
        v = vars(self)
        parameters = []
        for name in v:
            param = v[name]
            if type(param) == Linear:
                linear = v[name]
                parameters.append(param.weight)
                parameters.append(param.bias)
            elif type(param) == BatchNormalization:
                parameters.append(param.alpha)
                parameters.append(param.beta)
            elif type(param) == Conv2D:
                parameters.append(param.kernel)
        return parameters
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
