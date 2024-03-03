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
        self.l1 = Linear(2, 3, dtype=Type.f32)
        self.l2 = Linear(3, 1, dtype=Type.f32)
    def forward(self, x):
        z1 = Sigmoid.forward(self.l1.forward(x))
        z2 = Sigmoid.forward(self.l2.forward(z1))
        return z2

n = net()
optim = SGD(n.getParameters())
EPOCHS = 10000

for i in range(EPOCHS):
    for j in range(4):
        optim.zeroGrad()
        input = Tensor(inputs[j], shape=(1,2), dtype=Type.f32)
        predicted = n.forward(input)
        truth = Tensor([[outputs[j]]], dtype=Type.f32)
        loss = MSE.forward(truth, predicted)
        loss.backward()
        optim.step(0.1)

for i in range(len(inputs)):
    input = Tensor(inputs[i], shape=(1,2), dtype=Type.f32)
    predicted = n.forward(input)
    truth = Tensor([[outputs[i]]])
    print(input.arr, "->", predicted.arr, "|", truth.arr)
