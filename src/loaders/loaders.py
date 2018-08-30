import itertools

import torch
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from ..utils import round_robin, random_merge


class RepeatIterator:
    def __init__(self, obj):
        self.__obj = obj
        self.__iterator = itertools.repeat(iter(obj))

    def get(self, k=0):
        if k <= 0:
            k = len(self.__obj)

        yield from itertools.islice(self.__iterator, k)

    def __len__(self):
        return len(self.__obj)


class TorchPickleWrapper(object):
    def __init__(self, obj):
        self.obj = obj

    @classmethod
    def wrap(cls, obj):
        if hasattr(obj, 'numpy'):
            obj = cls(obj.numpy())
        return obj

    @classmethod
    def unwrap(cls, obj):
        if obj.__class__ == cls:
            obj = torch.from_numpy(obj.obj)
        return obj


class AlternatingBatchSampler(object):
    # noinspection PyShadowingNames
    def __init__(self, indices, batch_size, shuffle, drop_last, round_robin=False):
        self.round_robin = round_robin

        self.batch_samplers = []
        for idx in indices:
            if shuffle:
                sampler = SubsetRandomSampler(idx)
            else:
                sampler = idx
            self.batch_samplers.append(BatchSampler(sampler, batch_size, drop_last))

    def __iter__(self):
        if not self.round_robin:
            return random_merge(*self.batch_samplers)
        else:
            return round_robin(*self.batch_samplers)

    def __len__(self):
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)
