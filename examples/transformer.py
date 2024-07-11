from tops import *
import math

# word1 word2 word3       ^
#   |     |     |         |
#   |     |     |        dModel
#   |     |     |         |

def LayerNormBlock(input): return (input - Mean(input, dim=1, keepdim=True)) / (StdDev(input, dim=1, keepdim=True) + DELTA)
def PositionEncodingBlock(input):
    d = []
    for i in range(input.shape()[1]):
        val = i / (math.pow(10000, (2*i)/dModel))
        if i%2 == 0: d.append(Tensor.fill((dModel, 1), math.sin(val)))
        else: d.append(Tensor.fill((dModel, 1), math.cos(val)))
    pos = concat(d, axis=1)
    return input + pos
class FeedForwardBlock(NN):
    def __init__(self, density = 200):
        self.l1 = Linear(dModel, density)
        self.l2 = Linear(density, dModel)
    def forward(self, input):
        i = ReLu(self.l1.forward(input))
        return self.l2.forward(i)
class SingleHeadAttention(NN):
    def __init__(self, h):
        assert dModel % h == 0, "dModel is not divisible by h"
        self.q = Linear(dModel, dModel)
        self.k = Linear(dModel, dModel)
        self.v = Linear(dModel, dModel)
        self.h = h
    def forward(self, input, q, k, v, mask):
        return Softmax((q @ k.transpose())/(dModel ** 0.5)) @ v
