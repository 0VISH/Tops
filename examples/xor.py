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
    (1.),
)

class net(NN):
    def __init__(self):
        self.w1 = Tensor.rand(2, 2)
        self.b1 = Tensor.rand(1, 2)
        self.w2 = Tensor.rand(2, 1)
        self.b2 = Tensor.rand(1, 1)
        print(self.w2.arr)
    def forward(self, x):
        z1 = Sigmoid.forward(x @ self.w1 + self.b1)
        z2 = z1 @ self.w2 + self.b2
        return z2


n = net()
lossFunc = MSE()
optim = SGD(n.getParameters())
EPOCHS = 10000
for i in range(EPOCHS):
    for j in range(len(inputs)):
        optim.zeroGrad()
        input = Tensor(inputs[j], shape=(1,2))
        predicted = n.forward(input)
        truth = Tensor([[outputs[j]]])
        loss = lossFunc.forward(truth, predicted)
        loss.backward()
        optim.step()

for i in range(len(inputs)):
    input = Tensor(inputs[i], shape=(1,2))
    predicted = n.forward(input)
    truth = Tensor([[outputs[i]]])
    print(input.arr, "->", predicted.arr, "|", truth.arr)
