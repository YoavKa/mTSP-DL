import abc


class Transform(object):
    def __init__(self, indices=None):
        self.indices = indices
        if indices is not None:
            assert len(self.indices) == len(set(self.indices)), 'the indices must be unique!'

    def __call__(self, *parts):
        if self.indices is None:
            return self.apply(*parts)

        else:
            parts = list(parts)
            new_parts = self.apply(*[parts[index] for index in self.indices])
            for i, index in enumerate(self.indices):
                parts[index] = new_parts[i]
            parts.extend(new_parts[len(self.indices):])
            return parts

    @abc.abstractmethod
    def apply(self, *parts):
        raise NotImplementedError()


class IdentityTransform(Transform):
    @staticmethod
    def apply(*parts):
        return parts
