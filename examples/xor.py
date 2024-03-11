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
        z1 = Sigmoid(self.l1.forward(x))
        z2 = Sigmoid(self.l2.forward(z1))
        return z2

n = net()
optim = SGD(n.getParameters())
EPOCHS = 3000
runningLoss = 0

for i in range(EPOCHS):
    for j in range(len(inputs)):
        optim.zeroGrad()
        input = Tensor([inputs[j]], dtype=Type.f32)
        predicted = n.forward(input)
        truth = Tensor([[outputs[j]]], dtype=Type.f32)
        loss = MSE(truth, predicted)
        runningLoss += loss.arr
        print(f"epoch: {i}/{EPOCHS} loss: {loss.arr}", end="\r")
        loss.backward()
        optim.step(0.1)

print("\nloss:", runningLoss/(EPOCHS*len(inputs)))
    
for i in range(len(inputs)):
    input = Tensor(inputs[i], shape=(1,2), dtype=Type.f32)
    predicted = n.forward(input)
    truth = Tensor([[outputs[i]]])
    print(input.arr, "->", predicted.arr, "|", truth.arr)
