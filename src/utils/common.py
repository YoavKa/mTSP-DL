import datetime
import itertools
import numbers
import random
import re

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


def is_ok(func, args, err):
    try:
        func(*args)
        return True
    except err:
        return False


def is_int(str_):
    return is_ok(int, [str_], ValueError)


def is_float(str_):
    return is_ok(float, [str_], ValueError)


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


def assert_result(result, *test_args, **test_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        actual_res = func(*test_args, **test_kwargs)
        cmp = actual_res == result
        if hasattr(cmp, 'all'):
            cmp = cmp.all()
        if not cmp:
            message = 'Result assertion failed!\n'
            message += 'Arguments: ' + str(test_args) + '\n'
            message += 'Keyword arguments: ' + str(test_kwargs) + '\n'
            message += 'Expected result: ' + str(result) + '\n'
            message += 'Actual result: ' + str(actual_res)
            raise AssertionError(message)

        return wrapper
    return decorator


# taken from https://stackoverflow.com/a/4628148
TIME_REGEX = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')


# taken from https://stackoverflow.com/a/4628148
def parse_time(time_str):
    # USAGE: parser.add_argument('delta', type=parse_time, default=parse_time('0s'), nargs='?')
    #        delta.total_seconds()
    parts = TIME_REGEX.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return datetime.timedelta(**time_params)
