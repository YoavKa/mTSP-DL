import itertools


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
