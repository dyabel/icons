import torch

from func_sample import construct_batches
from approximator import get_approximator
import os


def find_approx_func_under_emax(func, domain, info, func_name="", num_min=2, num_max=128):
    batch_num = info['batch_num']

    num = (num_max + num_min) // 2
    best_now = (num, None)
    log = []
    batches, sample_num = construct_batches(func, domain, info['test_step'], batch_num)
    print("Finish constructing the batches")

    while 1:
        print('Try', num, 'for', func_name)
        approximator, satisfy = get_approximator(func, domain, num, info, batches=batches, sample_num=sample_num)

        e_now = torch.sqrt(sum(torch.sum((y - approximator(x)) ** 2) for x, y in batches) / sample_num)
        print(func_name, num_min, num, num_max, e_now.item())
        log.append((func_name, num, num_min, num_max, e_now.item()))
        if not satisfy:
            num_min = num
            if abs(num_max - num_min) <= 1:
                break
            num = (num_max + num_min) // 2
        else:
            best_now = (num, approximator)
            num_max = num
            if abs(num_max - num_min) <= 1:
                break
            num = (num_max + num_min) // 2

    if info['save'] and best_now[1] is not None:
        torch.save(best_now[1].state_dict(), func_name + '_' + str(best_now[0]) + '.pkl')
        with open(func_name + '_' + str(best_now[0]) + '.txt', 'w') as wf:
            wf.write(str(log))

    return best_now


def fix_hidden_layer_approx(func, domain, info, batches, sample_num, func_name="", num=64):
    approximator, _ = get_approximator(func, domain, num, info, batches=batches, sample_num=sample_num)
    e_now = sum(torch.sum(torch.abs((y - approximator(x)) / (y))) for x, y in batches) / sample_num

    if info['save']:
        os.makedirs('funcs_new', exist_ok=True)
        torch.save(approximator.state_dict(), 'funcs_new/' + func_name + '_' + str(num) + '.pkl')
    return e_now
