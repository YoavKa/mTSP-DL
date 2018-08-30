import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
# noinspection PyProtectedMember
from torch.nn._functions.dropout import FeatureDropout


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        elif len(args) > 1:
            return args


class PairwiseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(PairwiseLinear, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x:    Float(batch x length x in_dim)
        # res:  Float(batch x length x out_dim)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


def distribute(nets, xs):
    if isinstance(nets, (list, tuple, nn.ModuleList)):
        return [nets[i](x) for i, x in enumerate(xs)]
    else:
        return [nets(x) for x in xs]


class Distributed(nn.Module):
    def __init__(self, net, *args):
        super(Distributed, self).__init__()
        self.nets = nn.ModuleList([
            net(*arg) for arg in zip(*args)
        ])

    def forward(self, xs):
        return distribute(self.nets, xs)


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(FeedForwardLayer, self).__init__()
        assert len(in_dims) == len(hidden_dims) == len(out_dims), \
            'the number of groups muts be consistent across a feed forward layer!'

        self.linear_1 = Distributed(PairwiseLinear, in_dims, hidden_dims)
        self.linear_2 = Distributed(PairwiseLinear, hidden_dims, out_dims)

    def forward(self, xs):
        # xs:   list[batch x length_i x in_dim_i]
        # res:  list[batch x length_i x out_dim_i]
        xs = self.linear_1(xs)  # list[Float(batch x length_i x hidden_dim_i]]
        xs = distribute(F.relu, xs)  # list[Float(batch x length_i x hidden_dim_i]]
        xs = self.linear_2(xs)  # list[Float(batch x length_i x out_dim_i]]
        return xs


class FeatureDropout1d(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(FeatureDropout1d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, '
                             'but got {}'.format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        # x:    Float(batch x length x channels)
        # res:  Float(batch x length x channels)
        x = x.transpose(1, 2)  # Float(batch x channels x length)
        x = FeatureDropout.apply(x, self.p, self.training, self.inplace)  # Float(batch x channels x length)
        x = x.transpose(1, 2)  # Float(batch x length x channels)
        return x


# TODO change with standard implementation in torch 0.4.0
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        # x:    Float(batch x length x dim)
        # res:  Float(batch x length x dim)

        # calculate std in a weird way, due to:
        # https://github.com/pytorch/pytorch/pull/2019#issuecomment-342787832

        mean = x.mean(dim=2, keepdim=True)  # batch x length x 1
        std = ((x - mean).pow(2).sum(dim=2, keepdim=True).div(x.size(2) - 1) + self.eps).sqrt()  # batch x length x 1

        normalized_x = (x - mean.expand_as(x)) / std.expand_as(x)
        return normalized_x * self.gamma + self.beta
