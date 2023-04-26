import math
import torch.nn as nn
def func(x, y):
    if x<y:
        y = math.sin(x)
    else:
        y = math.cos(x)
    return y