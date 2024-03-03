import numpy as np

class SGD:
    def __init__(self, parameters: list, batchSize=1):
        self.parameters = parameters
        self.batchSize = batchSize
    def zeroGrad(self):
        for i in self.parameters: i.fill(i.shape(), 0)
    def step(self, lr=0.001):
        for i in self.parameters:
            if(i.shape() == (1,1)): i.grad = np.sum(i.grad)
            i.arr -= lr * i.grad
