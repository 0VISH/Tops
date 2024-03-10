from .tensor import *

class Convolve(BinaryOp):
    def __init__(self, kernelSize, stride=1):
        self.kernel = Tensor.rand(kernelSize)
        self.stride = stride
    def forward(self, input: Tensor) -> Tensor:
        inputX, inputY = input.shape()
        kernelX, kernelY = self.kernel.shape()

        outputY = (inputY - kernelY) // self.stride + 1
        outputX = (inputX - kernelX) // self.stride + 1
        output = np.zeros((outputY, outputX))
        
        for y in range(0, inputY - kernelY + 1, self.stride):
            for x in range(0, inputX - kernelX + 1, self.stride):
                region = input.arr[y:y+kernelY, x:x+kernelX]
                output[y // self.stride, x // self.stride] = np.sum(region * self.kernel.arr)
        return output
        
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
        return parameters
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
