#This file contains all the blocks required to build a transformer

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
    def __init__(self, dQOut, dVOut):
        self.q = Linear(dModel, dQOut)
        self.k = Linear(dModel, dQOut)
        self.v = Linear(dModel, dVOut)
    def forward(self, input, q, k, v, mask=None):
        # NOTE: R + -inf = -inf
        if mask: input += mask
        q = input @ q
        k = input @ k
        v = input @ v
        return Softmax((q @ k.transpose())/(dModel ** 0.5)) @ v
