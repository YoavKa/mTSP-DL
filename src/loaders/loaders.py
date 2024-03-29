import torch
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, Sampler

from ..utils import round_robin, random_merge


class RepeatIterator:
    def __init__(self, obj):
        self.__obj = obj
        self.__iterator = iter(obj)

    def get(self, k=0):
        if k <= 0:
            k = len(self.__obj)

        for _ in range(k):
            try:
                yield next(self.__iterator)
            except StopIteration:
                self.__iterator = iter(self.__obj)
                try:
                    yield next(self.__iterator)
                except StopIteration:
                    raise RuntimeError("Couldn't restart the iterator!")

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
    def __init__(self, indices, batch_size, shuffle, drop_last, round_robin=False, dataset_index=False):
        self.round_robin = round_robin
        self.dataset_index = dataset_index

        self.batch_samplers = []
        for idx in indices:
            if shuffle:
                sampler = SubsetRandomSampler(idx)
            else:
                sampler = idx
            self.batch_samplers.append(BatchSampler(sampler, batch_size, drop_last))

    def __iter__(self):
        if not self.round_robin:
            return random_merge(*self.batch_samplers, index_sequence=self.dataset_index)
        else:
            return round_robin(*self.batch_samplers, index_sequence=self.dataset_index)

    def __len__(self):
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)


class RangeSampler(Sampler):
    def __init__(self, count):
        super().__init__(None)
        self.count = count

    def __iter__(self):
        yield from range(self.count)

    def __len__(self):
        return self.count
