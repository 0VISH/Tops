from tops import *

file = open("extra/data/MnistHandWritten.csv")
next(file)

data = []
lineCount = 0
for line in file:
    line = line.split(',')
    output = int(line[0])
    pixels = [float(i) for i in line[1:]]
    oneHot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    oneHot[output] = 1
    data.append([pixels, oneHot])
    if lineCount > 1000: break
    lineCount += 1

file.close()
    
class net(NN):
    def __init__(self):
        self.c1 = Conv2D((10, 10))
        self.c2 = Conv2D((5, 5))
        self.l1 = Linear(15*15, 150)
        self.l2 = Linear(150, 50)
        self.l3 = Linear(50, 10)
    def forward(self, x):
        z1 = self.c1.forward(x)
        z2 = Sigmoid(z1)
        z3 = self.c2.forward(z2)
        z4 = Sigmoid(z3)
        z5 = z4.flatten()
        z6 = self.l1.forward(z5)
        z7 = self.l2.forward(z6)
        z8 = self.l3.forward(z7)
        return z8

n = net()
optim = SGD(n.getParameters())
EPOCHS = 4
runningLoss = 0

for i in range(EPOCHS):
    for j in range(len(data)):
        optim.zeroGrad()
        input = Tensor([data[j][0]], shape=(28, 28))
        truth = Tensor([data[j][1]])
        predicted = n.forward(input)
        loss = MSE(truth, predicted)
        runningLoss += loss.arr
        print(f"epoch: {i}/{EPOCHS} loss: {loss.arr}", end="\r")
        loss.backward()
        optim.step(0.1)

print("\nloss:", runningLoss/(EPOCHS*len(inputs)))
