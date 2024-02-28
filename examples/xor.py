from tops import *

class net(NN):
    def __init__(self):
        self.w1 = Tensor.rand(2, 5)
        self.b1 = Tensor.rand(1, 5)
        self.w2 = Tensor.rand(5, 1)
        self.b2 = Tensor.rand(1, 1)
    def forward(self, x):
        z1 = Sigmoid.forward(x @ self.w1 + self.b1)
        z2 = z1 @ self.w2 + self.b2
        return z2
