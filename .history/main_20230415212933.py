import math
import torch.nn as nn
from typing import List
import torch
from layers import LIFSpike
from func_sample import FuncSample
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
    
def construct_batches(func, domain, sample_step, batch_num, cuda=True):
    func_sample = FuncSample(func, domain, sample_step, cuda)
    batch = 0
    batches = []
    input_batch = torch.empty(0, func_sample.dim()).cuda()
    output_batch = torch.empty(0).cuda()
    dim = func_sample.dim()
    if not cuda:
        input_batch = torch.empty(0, dim)
        output_batch = torch.empty(0)

    last_batch_num = len(func_sample) % batch_num
    batch_iter = len(func_sample) // batch_num
    iter = 0
    for x, y in func_sample:
        batch += 1
        if batch == batch_num or (iter == batch_iter and batch == last_batch_num):
            iter += 1
            batch = 0
            batches.append((input_batch, output_batch))
            input_batch = torch.empty(0, dim).cuda()
            output_batch = torch.empty(0).cuda()
            if not cuda:
                input_batch = torch.empty(0, dim)
                output_batch = torch.empty(0)
        input_batch = torch.cat((input_batch, x.view(1, -1)), 0)
        output_batch = torch.cat((output_batch, y.view(-1)), 0)

    return batches, len(func_sample)

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
    batches, sample_num = construct_batches(torch.sin, domain, 0.01, 100000)

    