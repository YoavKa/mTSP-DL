import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from ..utils import strip_main_diagonal
from .common import Identity, PairwiseLinear, distribute, Distributed, FeedForwardLayer, FeatureDropout1d, LayerNorm


def pool(nodes, from_length, is_avg=False, self_pool=True, weights=None):
    # nodes:        Float(batch x to_length x to_dim)
    # weights:      None | Float(batch x from_length x to_length x to_dim)
    # res:          Float(batch x from_length x to_dim

    # weighted pooling
    if weights is not None:
        # Float(batch x from_length x to_length x to_dim)
        nodes = nodes.unsqueeze(1).expand(nodes.size(0), from_length, nodes.size(1), nodes.size(2))
        # Float(batch x from_length x to_length x to_dim)
        new_nodes = nodes * weights  # Float(batch x from_length x to_length x to_dim)

        # below: Float(batch x from_length x to_dim)
        if is_avg:
            return new_nodes.mean(dim=2)
        else:
            return new_nodes.max(dim=2)[0]

    # LOO pooling
    elif not self_pool and nodes.size(1) > 1:
        assert nodes.size(1) == from_length, 'LOO pooling is possible only when the nodes pool over themselves!'

        # Float(batch x from_length x to_length x to_dim)
        new_nodes = nodes.unsqueeze(1).repeat(1, from_length, 1, 1)
        # Float(batch x from_length x (to_length - 1) x to_dim)
        new_nodes = strip_main_diagonal(new_nodes.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if is_avg:
            return new_nodes.mean(dim=2)  # Float(batch x from_length x to_dim)
        else:
            return new_nodes.max(dim=2)[0]  # Float(batch x from_length x to_dim)

    # regular pooling
    else:
        if is_avg:
            res = nodes.mean(dim=1)  # Float(batch x to_dim)
        else:
            res = nodes.max(dim=1)[0]  # Float(batch x to_dim)

        res = res.unsqueeze(1).expand(nodes.size(0), from_length, nodes.size(2))  # Float(batch x from_length x to_dim)
        return res


class PoolLayer(nn.Module):
    def __init__(self, in_dims, out_dims, is_avg=False, self_pool=True):
        super(PoolLayer, self).__init__()
        assert len(in_dims) == len(out_dims), 'the number of groups must be consistent across a pooling layer!'

        self.linears = nn.ModuleList([
            PairwiseLinear(in_dim + sum(in_dims), out_dim)
            for in_dim, out_dim
            in zip(in_dims, out_dims)
        ])
        self.is_avg = is_avg
        self.self_pool = self_pool

    def forward(self, xs, weights=None):
        # xs:       list[Float(batch x length_i x in_dim_i)]
        # weights:  None | list[None | list[None | Float(batch x length_i x length_j x in_dim_i)]]
        # res:      list[Float(batch x length_i x out_dim_i)]

        if weights is None:
            weights = [None] * len(xs)

        res = []
        for i, (x, weight) in enumerate(zip(xs, weights)):
            if weight is None:
                weight = [None] * len(xs)
            pooled_contexts = [
                pool(y, x.size(1), is_avg=self.is_avg, self_pool=(self.self_pool or j != i), weights=w,)
                for j, (y, w) in enumerate(zip(xs, weight))
            ]
            # below: Float(batch x length i x (in_dim_i + sum(in_dims)))
            x_concat = torch.cat((x, *pooled_contexts), dim=2)
            new_x = self.linears[i](x_concat)  # Float(batch x length_i x out_dim_i)
            res.append(new_x)
        return res


class PermutationInvariantNet(nn.Module):
    def __init__(self, layers, in_dims, main_dims, ff_hidden_dims, avg_pool, residual, norm, dropout, embeddings,
                 self_pool, embedding_norm):
        super().__init__()

        if embeddings:
            self.embeddings = Distributed(PairwiseLinear, in_dims, main_dims)
        else:
            assert in_dims == main_dims, 'in_dims must equal main_dims if no embeddings are used'
            self.embeddings = Identity()

        self.pool_layers = nn.ModuleList([
            PoolLayer(main_dims, main_dims, is_avg=avg_pool, self_pool=self_pool)
            for _ in range(layers)
        ])
        self.feed_forwards = nn.ModuleList([
            FeedForwardLayer(main_dims, ff_hidden_dims, main_dims) for _ in range(layers)
        ])

        if residual:
            self.apply_layer = lambda olds, news: [old + new for old, new in zip(olds, news)]
        else:
            self.apply_layer = lambda olds, news: distribute(F.relu, news)

        if embedding_norm:
            self.embedding_norms = Distributed(LayerNorm, main_dims)
        else:
            self.embedding_norms = Identity()

        self.pool_norms = nn.ModuleList([
            Distributed(LayerNorm, main_dims)
            if norm
            else Identity()
            for _ in range(layers)
        ])
        self.feed_forward_norms = nn.ModuleList([
            Distributed(LayerNorm, main_dims)
            if norm
            else Identity()
            for _ in range(layers)
        ])

        if dropout > 0:
            self.dropout = Distributed(FeatureDropout1d, [dropout] * len(main_dims))
        else:
            self.dropout = Identity()

        self.layers = layers

    def forward(self, xs, weights=None):
        # xs:       list[Float(batch x length_i x in_dim_i)]
        # weights:  None | list[None | list[None | Float(batch x length_i x length_i x main_dim_i)]]
        # res:      list[Float(batch x length_i x main_dim_i)]
        xs = self.embeddings(xs)
        xs = self.embedding_norms(xs)

        for i in range(self.layers):
            new_xs = self.pool_layers[i](xs, weights=weights)
            new_xs = self.dropout(new_xs)
            xs = self.apply_layer(xs, new_xs)
            xs = self.pool_norms[i](xs)

            new_xs = self.feed_forwards[i](xs)
            new_xs = self.dropout(new_xs)
            xs = self.apply_layer(xs, new_xs)
            xs = self.feed_forward_norms[i](xs)

        return xs
