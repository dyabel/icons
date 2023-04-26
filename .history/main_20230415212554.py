import math
import torch.nn as nn
from typing import List
import torch
from layers import LIFSpike
def func(x, y):
    if x<y:
        y = math.sin(x)
    else:
        y = math.cos(x)
    return y

def build_mlp(dims: List[int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)

class SNN(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1_s = nn.Linear(2,32)
        self.fc2_s =nn.Linear(32, 1)

        self.spike = LIFSpike()
        
    def forward(self, x):
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        # out  # [N, neurons, steps]
        return x


if __name__ == '__main__':
    # sin_func = build_mlp([1,16,1])
    # cos_func = build_mlp([1,16,1])
    from approximator import get_approximator, UnaryApproximator
    from domain import Domain
    domain =  Domain(-10, 10)

    sin_approximator = UnaryApproximator(torch.sin, 64, domain, 3)
    sin_approximator.load_state_dict(torch.load('funcs_new/sin_64.pkl'))
    cos_approximator = UnaryApproximator(torch.cos, 64, domain, 3)
    sin_approximator.load_state_dict(torch.load('funcs_new/cos_64.pkl'))
    