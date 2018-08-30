from concurrent.futures import ProcessPoolExecutor as PoolExecutor
import itertools
from glob import iglob
import os

import torch

from .tensors import USE_GPU


def prepare_write(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def torch_save(path, data):
    prepare_write(path)
    torch.save(data, path)


def torch_load(path):
    map_location = None if USE_GPU else (lambda storage, loc: storage)
    return torch.load(path, map_location=map_location)


def traverse(files):
    if isinstance(files, str):
        files = [files]

    files = [iglob(file) for file in files]
    files = list(itertools.chain(*files))

    while len(files) > 0:
        file = files.pop()

        if os.path.isdir(file):
            for sub_file in os.listdir(file):
                files.append(os.path.join(file, sub_file))

        else:
            yield file


def apply_traverse(files, func, *args, **kwargs):
    if isinstance(files, str):
        def get_name(file):
            return os.path.basename(file) if file == files else os.path.relpath(file, files)
    else:
        def get_name(file):
            return file

    result = {get_name(file): func(file, *args, **kwargs)
              for file in traverse(files)}
    return result


def apply_traverse_async(workers, files, func, *args, **kwargs):
    if workers is not None and workers == 0:
        return apply_traverse(files, func, *args, **kwargs)

    futures = {}
    with PoolExecutor(max_workers=workers) as excutor:
        for file in traverse(files):
            if file == files:
                name = os.path.basename(file)
            elif isinstance(files, str):
                name = os.path.relpath(file, files)
            else:
                name = file
            futures[name] = excutor.submit(func, file, *args, **kwargs)

    result = {}
    for file, future in futures.items():
        result[file] = future.result()
    return result
