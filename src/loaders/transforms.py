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


class ComposeTransforms(Transform):
    def __init__(self, *transforms):
        super(ComposeTransforms, self).__init__()
        self.transforms = transforms

    def apply(self, *parts):
        for t in self.transforms:
            parts = t(*parts)
        return parts


class FilterParts(Transform):
    def __init__(self, *valid_indices):
        super(FilterParts, self).__init__()
        self.valid_indices = valid_indices

    def __call__(self, *parts):
        return self.apply(*parts)

    def apply(self, *parts):
        parts = [parts[idx] for idx in self.valid_indices]
        return parts
