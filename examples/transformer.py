from tops import *
from sys import argv

def LayerNormBlock(input): return (input - Mean(input, dim=1, keepdim=True)) / (StdDev(input, dim=1, keepdim=True) + DELTA)
