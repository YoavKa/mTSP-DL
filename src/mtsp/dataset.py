import torch

from ..loaders import BaseDataset, Transform


class CitiesDistances(Transform):
    def __init__(self, cities_idx):
        super().__init__((cities_idx,))

    def apply(self, cities):
        # cities:   Float(n x d)
        # res:      Float(n x d), Float(n x n)
        n, d = cities.size()

        from_cities = cities.unsqueeze(1).expand(n, n, d)  # Float(n x n x d)
        to_cities = cities.unsqueeze(0).expand(n, n, d)  # Float(n x n x d)
        dists = (from_cities - to_cities).norm(p=2, dim=2)  # Float(n x n)

        return cities, dists


class SVDCities(Transform):
    def __init__(self, cities_idx, dists_idx, svd_dim):
        super().__init__((cities_idx, dists_idx))
        self.svd_dim = svd_dim

    def apply(self, cities, dists):
        # cities:   Float(n x 2)
        # dists:    Float(n x n)
        # res:      Float(n x k), Float(n x n)
        if self.svd_dim <= 0:
            return cities, dists

        # normalize distances
        dists = dists / dists.mean()

        u, s, v = dists.svd()
        top_indices = s.topk(self.svd_dim)[1]  # Long(k)
        u_approx = u.t()[top_indices].t()  # Float(n x k)
        s_approx = s[top_indices]  # Float(k)

        cities_approx = u_approx * s_approx.unsqueeze(0)  # Float(n x k)

        return cities_approx, dists


class SplitConnectivityMatrix(Transform):
    def __init__(self, con_matrix_idx):
        super().__init__((con_matrix_idx,))

    @staticmethod
    def apply(con_matrix):
        n = con_matrix.size(0)
        starts_idx = con_matrix[0].nonzero().squeeze(1)  # Float(m)
        target = torch.zeros(starts_idx.size(0), n, n).byte()
        for group, cur in enumerate(starts_idx):
            target[group, 0, cur] = 1
            while cur != 0:
                target[group, cur, :] = con_matrix[cur, :]
                cur = con_matrix[cur, :].nonzero()[0, 0]
        return target,


class SalesmenEncoding(Transform):
    def __init__(self, count_idx):
        super().__init__((count_idx,))

    def apply(self, count):
        # noinspection PyArgumentList
        return torch.stack([torch.FloatTensor([(i + 1.0) / count, count]) for i in range(count)]),


class MTSPDataset(BaseDataset):
    def parse_file(self, file_path):
        with open(file_path) as f:
            f_iter = iter(f)

            while True:
                try:
                    # read cities
                    coords = list(map(float, next(f_iter).strip().split(' ')))
                    # noinspection PyArgumentList
                    cities = torch.FloatTensor(coords).view(2, -1).t()

                    # read connectivity matrix
                    n = cities.size(0)
                    con_matrix = torch.zeros(n, n).byte()
                    for row in range(n):
                        line = next(f_iter).strip().split(' ')
                        # noinspection PyArgumentList
                        con_matrix[row] = torch.ByteTensor(list(map(int, line)))

                    # get groups count
                    m = con_matrix[0].nonzero().size(0)

                    # read constraints if any
                    label_length = float(next(f_iter).strip())

                    # skip delimiter
                    next(f_iter)

                    # normalize cities to the range (-1, 1) from (0, 1)
                    cities = cities * 2 - 1
                    label_length *= 2

                    yield (cities, m, con_matrix, label_length), {'length': label_length}, (n, m)

                except StopIteration:
                    # we reached end of file
                    break


class PointerNetworkDataset(BaseDataset):
    def parse_file(self, file_path):
        with open(file_path) as f:
            for line in iter(f):
                parts = line.split()
                split_idx = parts.index('output')

                # read cities
                coords = list(map(float, parts[:split_idx]))
                # noinspection PyArgumentList
                cities = torch.FloatTensor(coords).view(-1, 2)
                # normalize cities to the range (-1, 1) from (0, 1)
                cities = cities * 2 - 1

                # read path
                n = cities.size(0)
                con_matrix = torch.zeros(n, n).byte()
                for index in range(split_idx + 1, len(parts) - 1):
                    con_matrix[int(parts[index]) - 1, int(parts[index + 1]) - 1] = 1

                # calculate optimal length
                from_cities = cities.unsqueeze(1).expand(n, n, 2)  # Float(n x n x 2)
                to_cities = cities.unsqueeze(0).expand(n, n, 2)  # Float(n x n x 2)
                dists = (from_cities - to_cities).norm(p=2, dim=2)  # Float(n x n)
                label_length = (con_matrix.float() * dists).sum()

                yield (cities, 1, con_matrix, label_length), {'length': label_length}, n
