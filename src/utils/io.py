import os

import torch

from .gpu import USE_GPU


def prepare_write(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def torch_save(path, data):
    prepare_write(path)
    torch.save(data, path)


def torch_load(path):
    map_location = None if USE_GPU else (lambda storage, loc: storage)
    return torch.load(path, map_location=map_location)
