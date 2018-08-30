import itertools
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


# taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def round_robin(*iterables, index=False, index_sequence=False):
    # Recipe credited to George Sakkis
    assert not index or not index_sequence
    pending = len(iterables)
    nexts = itertools.cycle(((i, iter(it).__next__) for i, it in enumerate(iterables)))
    while pending:
        try:
            for i, next_ in nexts:
                if index:
                    yield i, next_()
                elif index_sequence:
                    yield [(i, n) for n in next_()]
                else:
                    yield next_()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def random_merge(*iterables, index=False, index_sequence=False):
    assert not index or not index_sequence
    nexts = [iter(it).__next__ for it in iterables]
    while len(nexts) > 0:
        i = random.randrange(len(nexts))
        try:
            if index:
                yield i, nexts[i]()
            elif index_sequence:
                yield [(i, n) for n in nexts[i]()]
            else:
                yield nexts[i]()
        except StopIteration:
            nexts.pop(i)
