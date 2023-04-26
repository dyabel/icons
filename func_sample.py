import torch
from torch.utils.data import Dataset
import numpy as np

from domain import Domain


class FuncSample(Dataset):
    def __init__(self, func, domain: Domain, step, cuda=True):
        step = float(step)

        self._func = func
        self._dim = domain.dim

        if self._dim == 1:
            max_value, min_value = domain.upper_bound[0], domain.lower_bound[0]
            num = int((max_value - min_value) / step) + 1
            self._domain = []
            for i in range(num):
                x1 = i * step + domain.lower_bound[0]
                if abs(x1) > 0.001:
                    self._domain.append((x1,))
            self._domain = torch.tensor(self._domain)
            # self._domain = torch.tensor([(i * step + min_value,) for i in range(num)])
        elif self._dim == 2:
            num_0 = int((domain.upper_bound[0] - domain.lower_bound[0]) / step) + 1
            num_1 = int((domain.upper_bound[1] - domain.lower_bound[1]) / step) + 1
            self._domain = []
            for i in range(num_0):
                for j in range(num_1):
                    x1, x2 = i * step + domain.lower_bound[0], j * step + domain.lower_bound[1]
                    if abs(x1) > 0.01 and abs(x2) > 0.01:
                        self._domain.append([x1, x2])
            self._domain = torch.tensor(self._domain)
            # self._domain = torch.tensor(
            #     [[i * step + domain.lower_bound[0], j * step + domain.lower_bound[1]]
            #      for i in range(num_0) for j in range(num_1)])
        elif self._dim == 3:
            num_0 = int((domain.upper_bound[0] - domain.lower_bound[0]) / step) + 1
            num_1 = int((domain.upper_bound[1] - domain.lower_bound[1]) / step) + 1
            num_2 = int((domain.upper_bound[2] - domain.lower_bound[2]) / step) + 1
            lower = domain.lower_bound

            self._domain = []
            for i in range(num_0):
                for j in range(num_1):
                    for k in range(num_2):
                        x1, x2, x3 = i * step + lower[0], j * step + lower[1], k * step + lower[2]
                        if abs(x1) > 0.01 and abs(x2) > 0.01 and abs(x3) > 0.01:
                            self._domain.append((x1, x2, x3))

            self._domain = torch.tensor(self._domain)
            # self._domain = torch.tensor(
            #     [[i*step + lower[0], j*step + lower[1], k*step + lower[2]]
            #      for i in range(num_0) for j in range(num_1) for k in range(num_2)])

        if cuda:
            self._domain = self._domain.cuda()

    def __getitem__(self, idx):
        value_x = self._domain[idx]
        value_y = self._func(*value_x)
        return value_x, value_y

    def __len__(self):
        return len(self._domain)

    def dim(self):
        return self._dim


#
# class BinaryFuncSample(Dataset):
#     def __init__(self, func, min_value, max_value, step, cuda=False):
#         self.func = func
#         num_0 = int((max_value[0] - min_value[0]) / step) + 1
#         num_1 = int((max_value[1] - min_value[1]) / step) + 1
#         self.domain = torch.tensor(
#             [[i * step + min_value[0], j * step + min_value[1]] for i in range(num_0) for j in range(num_1)]).float()
#         if cuda:
#             self.domain = self.domain.cuda()
#
#         self.cuda = cuda
#
#         self._dim = 2
#
#     def __getitem__(self, idx):
#         value_x = self.domain[idx]
#         value_y = self.func(*value_x)
#         return value_x, value_y
#
#     def __len__(self):
#         return len(self.domain)
#
#     def dim(self):
#         return self._dim


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
