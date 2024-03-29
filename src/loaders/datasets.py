import abc
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from ..utils import apply_traverse_async
from .loaders import TorchPickleWrapper, AlternatingBatchSampler
from .transforms import IdentityTransform


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, files, max_workers=None, load_transform=IdentityTransform(),
                 runtime_transform=IdentityTransform()):
        self.transform = runtime_transform

        self.samples = []
        stats_sum = defaultdict(float)
        type_to_idx = defaultdict(list)

        for cur_samples, cur_stats_sum, cur_type_to_idx in \
                apply_traverse_async(max_workers, files, self.handle_file, load_transform).values():
            for key, value in cur_type_to_idx.items():
                # noinspection PyArgumentList
                type_to_idx[key].append(torch.LongTensor(value) + len(self.samples))

            self.samples.extend(cur_samples)

            for key, value in cur_stats_sum.items():
                stats_sum[key] += value

        self.type_to_idx = {
            key: torch.cat(value, dim=0) for key, value in type_to_idx.items()
        }

        if len(self.samples) > 0:
            self.stats = {
                key: value / len(self.samples) for key, value in stats_sum.items()
            }
        else:
            self.stats = {}

    @abc.abstractmethod
    def parse_file(self, file_path):
        # yields [sample, stats, type]
        raise NotImplementedError()

    def handle_file(self, file_path, load_transform):
        samples = []
        stats_sum = defaultdict(float)
        type_to_idx = defaultdict(list)
        for sample, stats, type_ in self.parse_file(file_path):
            sample = load_transform(*sample)
            sample = tuple(map(TorchPickleWrapper.wrap, sample))
            samples.append(sample)

            for key, value in stats.items():
                stats_sum[key] += value

            type_to_idx[type_].append(len(samples) - 1)

        return samples, stats_sum, type_to_idx

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = tuple(map(TorchPickleWrapper.unwrap, sample))
        sample = self.transform(*sample)
        return sample

    def __len__(self):
        return len(self.samples)

    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False, round_robin=False):
        return AlternatingBatchSampler(self.type_to_idx.values(), batch_size, shuffle, drop_last,
                                       round_robin=round_robin)
