from tops import *

inputs = (
    (0., 0.),
    (0., 1.),
    (1., 0.),
    (1., 1.),
)
outputs = (
    (0.),
    (1.),
    (1.),
    (0.),
)

class net(NN):
    def __init__(self):
        self.w1 = Tensor.rand(2, 3)
        self.b1 = Tensor.rand(1, 3)
        self.w2 = Tensor.rand(3, 1)
        self.b2 = Tensor.rand(1, 1)
    def forward(self, x):
        z1 = Sigmoid.forward(x @ self.w1 + self.b1)
        z2 = Sigmoid.forward(z1 @ self.w2 + self.b2)
        return z2

n = net()
lossFunc = MSE()
optim = SGD(n.getParameters())
EPOCHS = 10000
for i in range(EPOCHS):
    for j in range(4):
        optim.zeroGrad()
        input = Tensor(inputs[j], shape=(1,2), dtype=Type.f32)
        predicted = n.forward(input)
        truth = Tensor([[outputs[j]]])
        loss = lossFunc.forward(truth, predicted)
        loss.backward()
        optim.step(0.1)

for i in range(len(inputs)):
    input = Tensor(inputs[i], shape=(1,2))
    predicted = n.forward(input)
    truth = Tensor([[outputs[i]]])
    print(input.arr, "->", predicted.arr, "|", truth.arr)
