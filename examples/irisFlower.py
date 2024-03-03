from tops import *
from random import shuffle

file = open("examples/data/IrisFlower.csv", "r")
next(file) #skip header

nameToId = {
    "Iris-versicolor\n":0,
    "Iris-setosa\n"    :1,
    "Iris-virginica\n" :2,
}
data = []
for line in file:
    line = line.split(",")
    values = line[:4]
    values = [float(i) for i in values]
    truth = nameToId[line[4]]
    output = [0, 0, 0]  #one hot encoding
    output[truth-1] = 1
    data.append([values, output])
    
file.close()
shuffle(data)

class net(NN):
    def __init__(self):
        self.l1 = Linear(4, 10)
        self.l2 = Linear(10, 5)
        self.l3 = Linear(5, 3)
    def forward(self, x):
        z1 = Sigmoid.forward(self.l1.forward(x))
        z2 = Sigmoid.forward(self.l2.forward(z1))
        z3 = Sigmoid.forward(self.l3.forward(z2))
        return z3

n = net()
optim = SGD(n.getParameters())
EPOCHS = 100

for i in range(EPOCHS):
    for j in range(len(data)):
        optim.zeroGrad()
        input = Tensor(data[j][0], shape=(1, 4))
        predicted = n.forward(input)
        truth = Tensor(data[j][1])
        loss = MSE.forward(truth, predicted)
        loss.backward()
        optim.step(0.1)

for i in range(10):
    input = Tensor(data[i][0], shape=(1, 4))
    predicted = n.forward(input)
    truth = Tensor(data[i][1])
    print(input.arr, "->", predicted.arr, "|", truth.arr)
