import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from func_sample import FuncSample, construct_batches
from ua import UniversalApproximator
from domain import Domain


class UnaryApproximator(nn.Module):
    def __init__(self, func, num, domain: Domain, optimize_level, cuda=True):
        super(UnaryApproximator, self).__init__()
        self.func = func
        self.num = num

        optimize_w2 = True if optimize_level >= 2 else False
        optimize_w3 = True if optimize_level >= 3 else False

        min_value, max_value = domain.lower_bound[0], domain.upper_bound[0]
        self.A = torch.nn.Parameter(torch.from_numpy(np.linspace(min_value, max_value, num, endpoint=True,
                                                                 dtype=np.float32)), requires_grad=True)
        # self.W1 = (func(self.A[1:]) - func(self.A[:-1])) / (self.A[1:] - self.A[:-1])  # f(A[i+1]) / (A[i+1] - A[i])
        # self.W1 = self.W1.cuda()
        self.W2 = torch.nn.Parameter(torch.ones(1, num - 1), requires_grad=optimize_w2)
        self.W3 = torch.nn.Parameter(torch.ones(1, num - 1), requires_grad=optimize_w3)

        self.use_cuda = cuda

    def forward(self, x: torch.Tensor):
        if self.use_cuda:
            x = x * torch.ones(1, self.num - 1).cuda()
        else:
            x = x * torch.ones(1, self.num - 1)
        x = (self.func(self.A[1:]) - self.func(self.A[:-1])) / (self.A[1:] - self.A[:-1]) * self.W2 * (
                    F.relu(x - self.A[:-1]) - F.relu(x - self.A[1:]))
        output = x.mm(self.W3.t()) + self.func(self.A[0])
        return output.view(-1)


class PolyApproximator(nn.Module):
    def __init__(self, func, num, domain: Domain):
        super(PolyApproximator, self).__init__()
        self.func = func
        self.num = num

        # step1 = (domain.upper_bound[0] - domain.lower_bound[0]) / num
        # step2 = (domain.upper_bound[1] - domain.lower_bound[1]) / num
        # print("Grid info: ", num, step1, step2)
        #
        # xs = [[i * step1 + domain.lower_bound[0] + random.uniform(0, step1), j * step2 + domain.lower_bound[1] + random.uniform(0, step2)]
        #       for i in range(num) for j in range(num)]

        x = []
        seg_num = 20
        nums = [num // seg_num] * seg_num
        for i in range(num % seg_num):
            nums[i] += 1
        for i in range(domain.dim):
            if i == 0:
                segment = np.linspace(domain.lower_bound[i], domain.upper_bound[i], seg_num + 1, endpoint=True,
                                      dtype=np.float32)
                x_seg = []
                for j in range(seg_num):
                    low = segment[j]
                    high = segment[j + 1]
                    x_seg.append(low + (high - low) * np.random.rand(nums[j], 1))
                x.append(np.row_stack(x_seg))
            else:
                low = domain.lower_bound[i]
                high = domain.upper_bound[i]
                x.append(low + (high - low) * np.random.rand(num, 1))
        xs = np.column_stack(x)

        ua, param_dict = UniversalApproximator(func, xs)
        # error = np.mean([abs(ua(*x) - func(*x)) for x in xs])
        # print(error)

        self.W1 = torch.nn.Parameter(torch.from_numpy(param_dict['W1']).float().t(), requires_grad=True)
        self.B1 = torch.nn.Parameter(torch.from_numpy(param_dict['B1']).float(), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.from_numpy(param_dict['W2']).float().t(), requires_grad=True)
        self.B2 = torch.nn.Parameter(torch.tensor([param_dict['B2']]).float(), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = x.mm(self.W1)
        x.add_(self.B1)
        F.relu(x, inplace=True)
        x = x.mm(self.W2)
        x.add_(self.B2)
        return x.view(-1)

    def second_stage(self):
        self.W1.requires_grad = True
        self.B1.requires_grad = True


def get_approximator(func, domain, segment_num, info, batches=None, sample_num=0):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    approximator = None
    func_arg_num = 1 # func.__code__.co_argcount
    if func_arg_num == 1:
        approximator = UnaryApproximator(func, segment_num, domain, info['optimize_level'], info['cuda'])
        if info['cuda']:
            approximator = approximator.cuda()
    elif func_arg_num >= 2:
        approximator = PolyApproximator(func, segment_num, domain).cuda()

    if batches is None:
        batches, sample_num = construct_batches(func, domain, info['test_step'], info['batch_num'], info['cuda'])

    torch.cuda.empty_cache()

    if func_arg_num >= 2:
        now_max = sum(torch.sum(torch.abs((y - approximator(x)) / (y))) for x, y in batches) / sample_num
        for i in range(info['find_init_approx_time']):
            approximator_now = PolyApproximator(func, segment_num, domain).cuda()
            torch.cuda.empty_cache()
            e_now = sum(torch.sum(torch.abs((y - approximator_now(x)) / (y))) for x, y in batches) / sample_num
            print("now_max, e_now: ", now_max.item(), e_now.item())
            if e_now < now_max:
                now_max = e_now
                approximator = approximator_now
                print("Replace with better approximator")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, approximator.parameters()), lr=info['lr'])

    satisfy_error = False
    second_stage = False
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for it in range(info['max_iteration']):
        print(len(batches))
        loss = sum(torch.sum(torch.abs((y - approximator(x)) / (y))) for x, y in batches) / sample_num + \
               sum(loss_fn(y, approximator(x)) for x, y in batches)
        square_error = loss.item()  # torch.sqrt(loss).item()

        if (info['print_log']):
            print(it, square_error)
        if square_error < info['e_max']:
            satisfy_error = True
            print('Early break: ', it, square_error, info['e_max'])
            break

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if square_error < 0.1 and not second_stage:
            print('Start second stage')
            second_stage = True
            # approximator.second_stage()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, approximator.parameters()), lr=1e-4)

    return approximator, satisfy_error

#
# def test():
#     function = funcs.cos_sin
#     nums = [4, 8, 16, 32]
#     errors = []
#     plt.figure()
#     count = 0
#     for i in nums:
#         count += 1
#         approximator, sample = get_approximator(function, -10, 10, i, max_iteration=100, sample_step=0.01)
#         error = sum((y - approximator(x)) ** 2 for x, y in sample) / len(sample)
#         print("{}".format(error))
#         errors.append(error)
#         plt.subplot(2, 2, count)
#         # plt.plot(sample.get_x(), sample.get_y())
#         # plt.plot(sample.get_x(), [approximator(x) for x in sample.get_x()])
#         plt.title(str(i))
#         plt.grid()
#
#     plt.show()
#     plt.plot(nums, errors)
#     plt.grid()
#     plt.show()
