class SGD:
    def __init__(self, parameters: list, lr=0.001, batchSize=1):
        self.parameters = parameters
        self.lr = lr
        self.batchSize = batchSize
    def zeroGrad(self):
        for i in self.parameters: i.fill(i.shape(), 0)
    def step(self):
        for i in self.parameters: i.arr -= self.lr * (i.grad/self.batchSize)
