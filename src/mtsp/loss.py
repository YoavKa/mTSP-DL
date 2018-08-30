import itertools

import torch
import torch.nn as nn

from ..utils import to_variable


class MTSPLoss(nn.Module):
    def __init__(self, starts_weight=0.5, simple_loss=False, no_perms=False, size_average=True):
        super(MTSPLoss, self).__init__()
        assert 0 <= starts_weight <= 1, 'starts_weight must be in [0,1]!'

        self.starts_weight = starts_weight
        self.simple_loss = simple_loss
        self.no_perms = no_perms
        self.size_average = size_average

        self._permutations = {}

    def get_permutations(self, groups):
        # res:  Long(perms x groups)
        if groups in self._permutations:
            return self._permutations[groups]

        # noinspection PyArgumentList
        permutations = torch.LongTensor(list(itertools.permutations(range(groups))))
        permutations = to_variable(permutations)

        self._permutations[groups] = permutations
        return permutations

    def forward(self, probs, target, dists):
        # log_probs:        Float(batch x groups x from_city x to_city)
        # target:           Byte(batch x groups x from_city x to_city)
        # dists:            Float(batch x from_city x to_city)
        b, m, n, _ = probs.size()
        target = target.float()  # Float(batch x groups x from_city x to_city)
        log_probs = probs.log()

        # calculate loss without permutations
        if self.no_perms:
            losses = -log_probs * target  # Float(batch x groups x from_city x to_city)

            # calculate simple loss without permutations
            if self.simple_loss:
                weighted_losses = losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n + m - 1)  # Float(batch)

            # calculate weighted loss without permutations
            else:
                starts_losses = losses[:, :, 0, :]  # Float(batch x groups x to_city)
                nexts_losses = losses[:, :, 1:, :]  # Float(batch x groups x (from_city - 1) x to_city)

                starts_losses = starts_losses.sum(dim=2).sum(dim=1) / m  # Float(batch)
                nexts_losses = nexts_losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n - 1)  # Float(batch)

                # Float(batch)
                weighted_losses = starts_losses * self.starts_weight + nexts_losses * (1 - self.starts_weight)

        # calculate permutation invariant loss
        else:
            # preprocess target and log_probs

            # Float(batch x 2 x groups x from_city x to_city)
            stacked_target = torch.stack((target, target.transpose(2, 3)), dim=1)
            # Float(batch x from_groups x 2 x to_groups x from_city x to_city)
            all_target = stacked_target.unsqueeze(1).expand(b, m, 2, m, n, n)
            # Float(batch x from_groups x 2 x to_groups x from_city x to_city)
            log_probs = log_probs.unsqueeze(2).unsqueeze(3).expand_as(all_target)

            # calculate the nll loss based on the target
            losses = -log_probs * all_target  # Float(batch x from_groups x 2 x to_groups x from_city x to_city)

            # calculate simple permutation invariant loss
            if self.simple_loss:
                # Float(batch x from_groups x 2 x to_groups)
                weighted_losses = losses.sum(dim=5).sum(dim=4) / (n + m - 1)

            # calculate weighted permutation invariant loss
            else:
                starts_losses = losses[:, :, :, :, 0, :]  # Float(batch x from_groups x 2 x to_groups x to_city)
                # Float(batch x from_groups x 2 x to_groups x (from_city - 1) x to_city)
                nexts_losses = losses[:, :, :, :, 1:, :]

                # below: Float(batch x from_groups x 2 x to_groups)
                starts_losses = starts_losses.sum(dim=4) / m
                nexts_losses = nexts_losses.sum(dim=5).sum(dim=4) / (n - 1)

                # Float(batch x from_groups x 2 x to_groups)
                weighted_losses = starts_losses * self.starts_weight + nexts_losses * (1 - self.starts_weight)

            # choose best direction for each match
            weighted_losses = weighted_losses.min(dim=2)[0]  # Float(batch x from_groups x to_groups)

            # choose best permutation
            perms = self.get_permutations(m)  # Long(perms x groups)
            perms = perms.unsqueeze(0).expand(b, perms.size(0), perms.size(1))  # Long(batch x perms x groups)
            weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
            weighted_losses = weighted_losses.sum(dim=2).min(dim=1)[0]  # Float(batch)

        if self.size_average:
            weighted_losses = weighted_losses.mean(dim=0)  # Float(1)

        return weighted_losses
