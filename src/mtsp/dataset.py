import torch

from ..loaders import BaseDataset


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
