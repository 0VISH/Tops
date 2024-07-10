from tops import *
import math

def LayerNormBlock(input): return (input - Mean(input, dim=1, keepdim=True)) / (StdDev(input, dim=1, keepdim=True) + DELTA)
def PositionEncodingBlock(input):
    d = []
    for i in range(input.shape()[1]):
        y = input.shape()[0]
        val = i / (math.pow(10000, (2*i)/y))
        if i%2 == 0: d.append(Tensor.fill((y, 1), math.sin(val)))
        else: d.append(Tensor.fill((y, 1), math.cos(val)))
    pos = concat(d, axis=1)
    return input + pos
