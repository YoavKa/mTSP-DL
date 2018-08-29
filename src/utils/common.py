import numbers
import random

import numpy as np

import torch
import torch.backends.cudnn


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = deterministic


def is_number(obj):
    return isinstance(obj, numbers.Number)
