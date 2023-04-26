import numpy
from convex import sort as csort
import numpy as np
import torch


def relu(x):
    return numpy.where(x > 0, x, 0)


def UniversalApproximator(f, xs):
    """

    Arguments:
        f {function} -- N个输入1个输出.
        xs {List[List]} -- M x N的二重列表
    """
    point_equations = csort(numpy.array(xs))
    xs, equations = zip(*point_equations)
    xs_torch = torch.tensor(xs)
    n = xs[0].shape[0]
    ys = [f(*x).item() for x in xs_torch]
    param_dict = dict()
    W1 = numpy.zeros((0, n))
    B1 = numpy.zeros((0,))
    W2 = numpy.zeros((1, 0))
    B2 = ys[0]

    for i in range(1, len(xs)):
        x = xs[i]
        y = ys[i]
        equation = equations[i]
        y_ = W2.dot(relu(W1.dot(x) + B1)) + B2

        fix = equation[:-1].dot(x) + equation[-1]
        if abs(fix) <= 0:
            # 这里应该是浮点精度溢出导致的，好像也没什么好办法。。。
            print("Warning: point {}: {} omit".format(i, x), fix)
            continue
        # 确保x这边equation取值是正数
        if fix < 0:
            fix = -fix
            equation = -equation
        W1 = numpy.concatenate([W1, equation[:-1].reshape(1, n)], axis=0)
        B1 = numpy.concatenate([B1, equation[-1:]])
        adjust = 0
        # if abs((y - y_) / (equation[:-1].dot(x) + equation[-1])) > 1:
        #    adjust = 1
        new_W2 = (numpy.array((y - y_ + adjust) / ((fix + adjust))).reshape((1, 1)))

        # if new_W2 > 1:
        #     new_W2 = numpy.array(1).reshape((1, 1))
        # if new_W2 < -1:

        W2 = numpy.concatenate([W2, new_W2], axis=1)

        # print('values: ', equation[:-1].dot(x) + equation[-1], y_)

    def approximator(*x):
        x = numpy.array(x)
        return (W2.dot(relu(W1.dot(x) + B1)) + B2).item()

    param_dict['W1'] = W1
    param_dict['B1'] = B1
    param_dict['W2'] = W2
    param_dict['B2'] = B2

    # print(W2)
    return approximator, param_dict


def mul(a, b, c):
    return (1/torch.sqrt(a + b))*c


def test(f, xs, domain):
    ua, _ = UniversalApproximator(f, xs)
    xs = torch.from_numpy(xs)
    train_error = torch.mean(torch.tensor([torch.abs(ua(*x) - f(*x)) for x in xs]))

    # for x in xs:
    #    print(x, ua(*x), ua(x[0] + 0.01, x[1] + 0.01))

    # func_sample = FuncSample(f, domain, 1.1)
    # for x, y in func_sample:
    #     print(x, y, ua(*x))
    # print(len(func_sample))
    # test_error = numpy.mean([abs(ua(*x) - y.item()) for x, y in func_sample])
    print(train_error)
    # print(test_error)


def get_xs(num, domain):
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
    return np.column_stack(x)

#
# d = Domain((0, -10, -8), (64, 64, 8))
#
# test(mul, get_xs(500, d), d)
