import math
import torch.nn as nn
from typing import List
import torch
from layers import LIFSpike
from func_sample import FuncSample
import torch.optim as optim

def bool_func(x, y):
    return x<y

def func(x, y):
    if bool_func(x, y):
        z = torch.sin(x)
    else:
        z = torch.cos(x)
    return z

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
    import time
    domain =  Domain(-10, 10)

    sin_approximator = UnaryApproximator(torch.sin, 64, domain, 3).cuda()
    sin_approximator.load_state_dict(torch.load('funcs_new/sin_64.pkl'))
    cos_approximator = UnaryApproximator(torch.cos, 64, domain, 3).cuda()
    sin_approximator.load_state_dict(torch.load('funcs_new/cos_64.pkl'))
    snn = SNN().cuda()
    criterion = nn.MSELoss()
    batches, sample_num = construct_batches(torch.sin, domain, 0.01, 100000)
    optimizer = optim.Adam(snn.parameters(), lr=0.1)
    x_batch = batches[0][0]
    # print(len(x_batch))
    x_batch = torch.cat((20*torch.rand(len(x_batch),1).cuda()-10, 20*torch.rand(len(x_batch),1).cuda()-10), dim=-1)
    # x_batch = torch.cat((x_batch.cuda(), 20*torch.rand(len(x_batch),1).cuda()-10), dim=-1)
    num = len(x_batch)
    y_batch = torch.stack([func(x[0],x[1]) for x in x_batch])
    batch_size = 1999
    for epoch in range(10000):
        loss_batch = 0
        ori_error_batch = 0.
        switch_loss_batch = 0.
        for i in range(num//batch_size):
            x = x_batch[i*batch_size:(i+1)*batch_size]
            labels = y_batch[i*batch_size:(i+1)*batch_size]
            # print(x.shape, labels.shape)
            # y = torch.rand(1) * 20 - 10
            switch = snn(x).squeeze()
            with torch.no_grad():
                sin_x = sin_approximator(x[:,0].unsqueeze(1))
            with torch.no_grad():
                cos_x = cos_approximator(x[:,0].unsqueeze(1))
            # print(switch.shape, sin_x.shape)
            prediction = switch * sin_x + (1-switch) * cos_x
            # print(prediction.shape, labels.shape)
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            loss_batch += loss.item()
            switch_loss_batch += ((bool_func(x[:,0], x[:,1])).float() - switch).mean().item()
            ori_error_batch += (((bool_func(x[:,0], x[:,1]).float()) * sin_x + (1-bool_func(x[:,0], x[:,1]).float()) * cos_x - labels)**2).mean()
            optimizer.step()
        print('epoch', epoch)
        print('average switch loss', switch_loss_batch/num*batch_size)
        print('average loss: ', loss_batch/num*batch_size)
        print('average original error:', ori_error_batch/num*batch_size)




    