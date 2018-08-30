from collections import defaultdict

import torch

from ..loaders import ComposeTransforms, FilterParts
from ..models import CustomModel
from ..utils import to_variable, get_data
from .beam_search import MTSPBeamSearch
from .dataset import PointerNetworkDataset, MTSPDataset, SplitConnectivityMatrix, SVDCities, CitiesDistances
from .dataset import SalesmenEncoding
from .loss import MTSPLoss
from .net import MTSPNet


class MTSPModel(CustomModel):
    @classmethod
    def default_kwargs(cls):
        result = super().default_kwargs()
        result.update({
            # data settings
            'ptrnet': False,

            # augmentation settings
            'svd': 4,

            # net settings
            'layers_count': 7,
            'main_dim': 256,
            'avg_pool': False,
            'no_residual': False,
            'no_layer_norm': False,
            'ff_hidden_dim': 1024,
            'dropout': 0.0,
            'self_pool': False,
            'no_embedding_norm': False,
            'softassign_layers': 100,
            'no_weighting': False,

            # loss settings
            'starts_weight': 0.5,
            'simple_loss': False,
            'no_perms_loss': False,
        })
        return result

    @classmethod
    def add_args(cls, parser):
        parser = super().add_args(parser)
        parser.add_argument('--beam-sizes', type=int, nargs='*', default=[])
        return parser

    def get_dataset(self, is_train):
        if self.start_args['ptrnet']:
            return PointerNetworkDataset
        else:
            return MTSPDataset

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_load_transform(self, is_train):
        transform = ComposeTransforms(  # (cities, groups, con_matrix, length)
            SplitConnectivityMatrix(2),  # (cities, groups, target, length)
            FilterParts(0, 1, 2),  # (cities, groups, target)
        )
        return transform

    # noinspection PyUnusedLocal
    def get_runtime_transform(self, is_train):
        transform = ComposeTransforms(  # (cities, groups, target)
            SalesmenEncoding(1),  # (cities, groups, target)
            CitiesDistances(0),  # (cities, groups, target, dists)
            SVDCities(0, 3, self.start_args['svd']),  # (cities, groups, target, dists)
        )
        return transform

    def init_net(self):
        layers = self.start_args['layers_count']
        main_dim = self.start_args['main_dim']
        avg_pool = self.start_args['avg_pool']
        residual = not self.start_args['no_residual']
        norm = not self.start_args['no_layer_norm']
        ff_hidden_dim = self.start_args['ff_hidden_dim']
        dropout = self.start_args['dropout']
        self_pool = self.start_args['self_pool']
        embedding_norm = not self.start_args['no_embedding_norm']
        softassign_layers = self.start_args['softassign_layers']
        weighting = not self.start_args['no_weighting']

        svd = self.start_args['svd']

        groups_dim = 2

        if svd > 0:
            cities_dim = svd
        else:
            cities_dim = 2

        return MTSPNet(layers, cities_dim, groups_dim, main_dim, avg_pool, residual, norm, ff_hidden_dim, dropout,
                       self_pool, embedding_norm, softassign_layers, weighting)

    def init_criterion(self):
        starts_weight = self.start_args['starts_weight']
        simple_loss = self.start_args['simple_loss']
        no_perms = self.start_args['no_perms_loss']

        return MTSPLoss(starts_weight=starts_weight, simple_loss=simple_loss, no_perms=no_perms, size_average=True)

    def get_stats_constants(self):
        constants = super().get_stats_constants()

        if len(self.start_args['beam_sizes']) > 0:
            if self.train_dataset is not None:
                constants['Train/Length/Best'] = self.train_dataset.stats['length']

            if self.val_datasets is not None:
                sum_of_avg_lengths = 0.0
                for dataset_name, dataset in self.val_datasets.items():
                    constants['Validate/{}/Length/Best'.format(dataset_name)] = dataset.stats['length']
                    sum_of_avg_lengths += dataset.stats['length']
                constants['Validate/Length/Best'] = sum_of_avg_lengths / len(self.val_datasets)

        return constants

    def get_train_scores(self, cities, groups, target, dists):
        cities, groups, target, dists = to_variable(cities, groups, target, dists)
        probs = self.net(cities, groups, dists)
        loss = self.criterion(probs, target, dists)

        result = {}

        if len(self.start_args['beam_sizes']) > 0:
            for beam, length in self.beam_search(dists, probs).items():
                result[f'Length/Beam {beam}'] = length

        return loss, result

    def beam_search(self, dists, probs, beam_sizes=None, raw_solutions=False, sum_lengths=True):
        if beam_sizes is None:
            beam_sizes = self.start_args['beam_sizes']

        if raw_solutions or not sum_lengths:
            results = defaultdict(list)
        else:
            results = defaultdict(float)

        for prob, distances in zip(probs, dists):
            # prob:         Float(m x n x n)
            # distances:    Float(n x n)
            m, n, _ = prob.size()

            for beam_size in beam_sizes:
                beam_search = MTSPBeamSearch(get_data(prob), get_data(distances))
                matches, lengths = beam_search.run(beam_size)  # Byte(beam x n x n), Float(beam x groups)
                paths_lengths = lengths.sum(dim=1)  # Float(beam)

                if raw_solutions:
                    results[beam_size].append(matches[paths_lengths.min(dim=0)[1][0]])  # Float(n x n)
                elif not sum_lengths:
                    results[beam_size].append(paths_lengths.min(dim=0)[0])
                else:
                    results[beam_size] += paths_lengths.min()

        if raw_solutions or not sum_lengths:
            return {beam: torch.stack(lengths) for beam, lengths in results.items()}
        else:
            return results

    def solve(self, cities, groups_count, beam_size=None, raw_solution=True, no_beam_search=False):
        cities, distances = CitiesDistances(0)(cities)
        cities, distances = SVDCities(0, 1, self.start_args['svd'])(cities, distances)
        groups = SalesmenEncoding(0)(groups_count)[0]  # Float(1 x m x d)

        cities, groups, distances = cities.unsqueeze(0), groups.unsqueeze(0), distances.unsqueeze(0)
        cities, groups, distances = to_variable(cities, groups, distances)

        probs = self.net(cities, groups, distances)  # Float(1 x m x n x n)

        if no_beam_search:
            return get_data(probs).squeeze(0)  # Float(m x n x n)

        if beam_size is not None and isinstance(beam_size, int):
            beam_size = [beam_size]
        solution = self.beam_search(distances, probs, beam_sizes=beam_size, raw_solutions=raw_solution)
        if raw_solution:
            solution = {beam: sol[0] for beam, sol in solution.items()}
        if beam_size is not None and len(beam_size) == 1:
            solution = list(solution.values())[0]
        return solution
