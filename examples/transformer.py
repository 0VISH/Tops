from tops import *
from sys import argv

def LayerNormBlock(input): return (input - Mean(input, dim=1, keepdim=True)) / (StdDev(input, dim=1, keepdim=True) + DELTA)

d = 0
k = False

tens = Tensor.rand((5,4))
print(tens.numpy())
out = Mean(tens, dim=d, keepdim=k)
print(out.numpy(), out.shape())
out.backward()
print(tens.gradient())

from torch import tensor
import torch

tens = tensor(tens.numpy(), requires_grad=True)
out = tens.mean(dim=d, keepdim=k)
print(out.detach().numpy(), out.shape)
out.backward(torch.ones_like(out))
print(tens.grad)
