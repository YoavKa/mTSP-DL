import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import PairwiseLinear, PermutationInvariantNet, normalize_dims
from ..utils import chunk_at, to_variable


class DistancesToWeights(nn.Module):
    def __init__(self, self_pool, main_dim):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(main_dim), requires_grad=True)

        self.self_pool = self_pool
        self.main_dim = main_dim

    def forward(self, dists):
        # dists:   Float(b x n x n)
        # res:      Float(b x n x n x main_dim)
        b, n, _ = dists.size()

        # normalize distances
        dists_mean = dists.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / (n * n)
        dists = dists / dists_mean

        # apply temperature
        # below: Float(b x n x n x main_dim)
        dists = dists.unsqueeze(3).expand(b, n, n, self.main_dim)
        dists = dists * self.temperature.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # calculate weights
        weights = (-dists).exp()  # Float(b x n x n x main_dim)

        # remove self pool
        if not self.self_pool:
            # noinspection PyArgumentList
            n_range = to_variable(torch.arange(n).long())  # Long(n)
            # Float(b x n x n x main_dim)
            idx = n_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, n, 1, self.main_dim)
            weights.scatter_(dim=2, index=idx, value=0)

        # shift
        # Float(b x n x n x main_dim)
        weights = weights * self.gamma.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
            self.beta.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        return weights


class MTSPSoftassign(nn.Module):
    def __init__(self, layers, eps=1e-8):
        super(MTSPSoftassign, self).__init__()
        self.eps = eps
        self.layers = layers

    def forward(self, output):
        # output:           Float(batch x groups x from_city x to_city)
        # res:              Float(batch x groups x from_city x to_city)

        if self.layers > 0:
            # calculate the maximum per each row in the first softmax (of both depot and others)
            depots_max = output[:, :, :1, :].max(dim=3, keepdim=True)[0]  # Float(batch x groups x 1 x 1)
            # Float(batch x groups x 1 x to_city)
            depots_max = depots_max.expand(output.size(0), output.size(1), 1, output.size(3))
            # Float(batch x 1 x (from_city - 1) x to_city)
            others_max = output[:, :, 1:, :].max(dim=1, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            # Float(batch x groups x (from_city - 1) x to_city)
            others_max = others_max.expand(output.size(0), output.size(1), output.size(2) - 1, output.size(3))
            output_max = torch.cat((depots_max, others_max), dim=2)  # Float(batch x groups x from_city x to_city)

            # subtract the maximum from the output
            output = output - output_max

        # calculate exponents and save intermediate results
        output = output.exp()

        # normalize the three dimensions in each layer
        for i in range(self.layers):
            output = output.clamp(min=self.eps)

            # normalize the depot and others in the (normal / inverse) matrix

            # below: Float(batch x groups x 1 x to_city)
            starts_output = output[:, :, :1, :]
            starts_output = normalize_dims(starts_output, (3,))

            # below: Float(batch x groups x (from_city - 1) x to_city)
            nexts_output = output[:, :, 1:, :]
            nexts_output = normalize_dims(nexts_output, (1, 3))

            # below: Float(batch x groups x from_city x to_city)
            output = torch.cat((starts_output, nexts_output), dim=2)
            output = output.transpose(2, 3)

        # fix the transposition
        if self.layers % 2 != 0:
            output = output.transpose(2, 3)

        return output


class MTSPNet(nn.Module):
    def __init__(self, layers, cities_in_dim, groups_in_dim, main_dim, avg_pool, residual, norm, ff_hidden_dim, dropout,
                 self_pool, embedding_norm, softassign_layers, weighting, memory_efficient=False):
        super().__init__()

        if weighting:
            self.distance_to_weights = DistancesToWeights(self_pool, main_dim)
        else:
            self.distance_to_weights = lambda dists: None
        self.perm_inv_net = PermutationInvariantNet(layers, [cities_in_dim, cities_in_dim, groups_in_dim],
                                                    [main_dim] * 3, [ff_hidden_dim] * 3, avg_pool, residual, norm,
                                                    dropout, True, self_pool, embedding_norm)
        self.linear_out_1 = PairwiseLinear(main_dim * 3, main_dim)
        self.linear_out_2 = PairwiseLinear(main_dim, 1)
        self.softassign = MTSPSoftassign(softassign_layers)

        self.memory_efficient = memory_efficient

    def forward(self, cities, groups, dists):
        # cities:   Float(batch x cities_length x d)
        # groups:   Float(batch x groups_length x groups_in_dim)
        # dists:    Float(batch x from_city x to_city)
        # res:      Float(batch x groups x from_city x to_city)

        # returns probs

        b = cities.size(0)
        n = cities.size(1)
        m = groups.size(1)

        depots = cities[:, :1, :]  # Float(batch x 1 x cities_in_dim)
        others = cities[:, 1:, :]  # Float(batch x (cities_length - 1) x cities_in_dim)

        # None | Float(batch x cities_length x cities_length x main_dim)
        # noinspection PyNoneFunctionAssignment
        weights = self.distance_to_weights(dists)
        if weights is None:
            sets_weights = None
        else:
            depots_weights = [weights[:, :1, :1, :], weights[:, :1, 1:, :], None]
            others_weights = [weights[:, 1:, :1, :], weights[:, 1:, 1:, :], None]
            groups_weights = None
            sets_weights = [depots_weights, others_weights, groups_weights]

        # run the network
        depots, others, groups = self.perm_inv_net([depots, others, groups], weights=sets_weights)
        cities = torch.cat((depots, others), dim=1)  # Float(batch x cities_length x main_dim)

        main_dim = cities.size(2)

        # combine sets to output tensor

        if self.memory_efficient:
            # below: Float(batch x cities_length x main_dim)
            from_cities = cities.unsqueeze(2).expand(b, n, n, main_dim)
            to_cities = cities.unsqueeze(1).expand_as(from_cities)
            # Float(batch x cities_length x cities_length x (main_dim*2))
            base_nodes = torch.cat((from_cities, to_cities), dim=3)
            # Float(batch x (cities_length*cities_length) x (main_dim*2))
            base_nodes = base_nodes.view(base_nodes.size(0), n * n, main_dim * 2)

            res = []
            for i, group in enumerate(chunk_at(groups, dim=1, squeeze=False)):  # Float(batch x 1 x main_dim)
                group = group.expand(b, n * n, group.size(2))  # Float(batch x (cities_length*cities_length) x main_dim)
                # Float(batch x (cities_length*cities_length) x (main_dim*3))
                nodes = torch.cat((group, base_nodes), dim=2)
                nodes = self.linear_out_1(nodes)
                nodes = F.relu(nodes)
                nodes = self.linear_out_2(nodes)  # Float(batch x (cities_length*cities_length) x 1)
                nodes = nodes.view(b, n, n)  # Float(batch x cities_length x cities_length)
                res.append(nodes)

            output = torch.stack(res, dim=1)  # Float(batch x groups_length x cities_length x cities_length)

        else:
            # below: Float(batch x groups_length x cities_length x cities_length x main_dim)
            at_groups = groups.unsqueeze(2).unsqueeze(3).expand(b, m, n, n, main_dim)
            from_cities = cities.unsqueeze(1).unsqueeze(3).expand_as(at_groups)
            to_cities = cities.unsqueeze(1).unsqueeze(2).expand_as(at_groups)

            # Float(batch x groups_length x cities_length x cities_length x (3 * main_dim))
            nodes = torch.cat((at_groups, from_cities, to_cities), dim=4)
            # Float(batch x (groups_length*cities_length*cities_length) x (3 * main_dim))
            nodes = nodes.view(b, m * n * n, main_dim * 3)
            nodes = self.linear_out_1(nodes)
            nodes = F.relu(nodes)
            nodes = self.linear_out_2(nodes)

            output = nodes.view(b, m, n, n)  # Float(batch x groups_length x cities_length x cities_length)

        # run softassign on the output tensor

        result = self.softassign(output)

        return result
