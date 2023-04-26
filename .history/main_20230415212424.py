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

class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(2,32)
        self.fc2 =nn.Linear(32, 1)

        self.spike = LIFSpike()
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


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
    